import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import jigsaw.modules.conv as conv
from jigsaw.utils_aggregate import gather_1d_results, gather_results_as_list
from tests.process_mesh import *

pytestmark = [pytest.mark.dist_required]

Pair = int | tuple[int, int]


def _pair(value: Pair) -> tuple[int, int]:
    """Normalise an ``int`` or 2-tuple to a ``(h, w)`` tuple."""
    return (value, value) if isinstance(value, int) else tuple(value)


# (kernel, stride, padding, dilation) -- each entry is an int or an (h, w)
# tuple, so asymmetric kernels/strides/paddings/dilations are covered. Sweeps
# the no-halo regime, stride < / > kernel, unit-stride, dilated, stride ==
# kernel forced onto the halo path by padding/dilation, and combos whose global
# output is not divisible by the spatial group (which must be rejected cleanly).
MULTI_CASES_2D = [
    (2, 2, 0, 1),
    (3, 2, 1, 1),
    (2, 3, 1, 1),
    (3, 1, 1, 1),
    (3, 1, 2, 2),
    ((3, 4), (2, 2), (1, 1), 1),
    ((4, 2), (2, 3), (1, 0), 1),
    (2, 2, 1, 1),
    ((3, 3), (3, 3), (0, 0), (1, 2)),
    (3, 2, 0, 1),
]
MULTI_IDS_2D = [
    f"k{k}_s{s}_p{p}_d{d}".replace(" ", "") for (k, s, p, d) in MULTI_CASES_2D
]


def _conv2d_reference_check(
    device: torch.device,
    mesh: dict,
    in_channels: int,
    out_channels: int,
    kernel_size: Pair,
    height: int,
    width: int,
    batch: int = 2,
    stride: Pair | None = None,
    padding: Pair = 0,
    dilation: Pair = 1,
) -> None:
    """Compare a distributed ``DistributedConv2D`` against a single-process conv2d.

    Works for any ``row.size()`` (spatial) x ``col.size()`` (channel) mesh and
    for both the no-halo regime (``stride == kernel_size``, the default) and the
    halo regime (any other stride). ``kernel_size`` / ``stride`` / ``padding`` /
    ``dilation`` accept an ``int`` or a ``(h, w)`` tuple, so asymmetric kernels
    and strides are covered. Rebuilds the equivalent non-distributed convolution
    from the sharded weight, bias and input, runs both forward and backward, and
    checks the output plus every gradient (input, weight, bias) after gathering
    the distributed shards.
    """
    row_group, col_group = mesh["row"], mesh["col"]
    n_spatial, n_channel = row_group.size(), col_group.size()
    model_group = mesh["model"]
    kernel_size = _pair(kernel_size)
    stride = kernel_size if stride is None else _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    torch.manual_seed(model_group.rank())
    x_local = torch.randn(
        batch, in_channels // n_channel, height, width // n_spatial, device=device
    )

    conv2d = conv.DistributedConv2D(
        in_channels,
        out_channels,
        kernel_size,
        device=device,
        spatial_group=row_group,
        channel_group=col_group,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,
        bias=True,
        parallelism=model_group.size(),
        bayesian=False,
        dtype=torch.float32,
    ).to(device)

    # --- reconstruct the equivalent global convolution -----------------------
    weight_global = gather_1d_results(
        conv2d.weight.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    bias_global = gather_1d_results(
        conv2d.bias.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    x_global = gather_1d_results(x_local, mesh, dim=-3, pm_key="col")
    x_global = gather_1d_results(x_global, mesh, dim=-1, pm_key="row").clone()

    x_global.requires_grad_(True)
    weight_global.requires_grad_(True)
    bias_global.requires_grad_(True)

    y_ref = F.conv2d(
        x_global,
        weight_global,
        bias_global,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    torch.manual_seed(2024)
    grad_global = torch.randn(y_ref.shape, device=device)
    y_ref.backward(grad_global)

    # --- distributed forward / backward on this rank's shard ----------------
    x_dist = x_local.clone().requires_grad_(True)
    y = conv2d(x_dist)

    out_local = out_channels // n_channel
    assert y_ref.shape[-1] % n_spatial == 0, (
        f"global output width {y_ref.shape[-1]} not divisible by spatial size {n_spatial}"
    )
    ow_local = y_ref.shape[-1] // n_spatial
    grad_local = grad_global[
        ...,
        col_group.rank() * out_local : (col_group.rank() + 1) * out_local,
        :,
        row_group.rank() * ow_local : (row_group.rank() + 1) * ow_local,
    ].contiguous()
    y.backward(grad_local)

    # --- gather distributed results back to global tensors ------------------
    def _to_global(t: torch.Tensor, channel_dim: int) -> torch.Tensor:
        t = gather_1d_results(t, mesh, dim=channel_dim, pm_key="col")
        return gather_1d_results(t, mesh, dim=-1, pm_key="row")

    y_global = _to_global(y.detach(), -3)
    x_grad_global = _to_global(x_dist.grad, -3)
    w_grad_global = gather_1d_results(conv2d.weight.grad, mesh, dim=0, pm_key="col")
    b_grad_global = gather_1d_results(conv2d.bias.grad, mesh, dim=0, pm_key="col")

    assert torch.allclose(y_global, y_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(x_grad_global, x_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(w_grad_global, weight_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(b_grad_global, bias_global.grad, atol=1e-4, rtol=1e-4)


class TestConv2d:
    def test_param_init_conv2d(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Verify initial weights for Conv2d are correctly synchronized."""
        local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
        torch.manual_seed(local_rank)  # each process gets a different initial seed
        row_comm_size, col_comm_size = (
            process_mesh_dict_2x2["row"].size(),
            process_mesh_dict_2x2["col"].size(),
        )
        in_dim, out_dim = 5, 10
        kernel_size = 4
        conv2d = conv.DistributedConv2D(
            in_dim,
            out_dim,
            kernel_size,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=kernel_size,
            padding=(0,),
            dilation=1,
            groups=1,
            bias=True,
            parallelism=4,
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        weights_gathered_as_list = gather_results_as_list(
            conv2d.weight, process_mesh_dict_2x2
        )

        for i in range(process_mesh_dict_2x2["model"].size() // row_comm_size):
            assert torch.allclose(
                weights_gathered_as_list[i],
                weights_gathered_as_list[i + col_comm_size],
                atol=1e-5,
            )

    def test_conv2d_forward(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """2-way spatial x 2-way channel matches a single-process conv2d."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
        )

    def test_stride_lt_kernel_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Overlapping windows (stride < kernel_size) match a single-process conv2d."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
        )

    def test_stride_gt_kernel_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Strided-past-kernel windows (stride > kernel_size) match a plain conv2d."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=2,
            height=14,
            width=14,
            stride=4,
            padding=0,
        )

    def test_dilated_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Dilated 'same'-style conv (stride 1) matches a plain conv2d."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=3,
            height=16,
            width=16,
            stride=1,
            padding=2,
            dilation=2,
        )

    def test_asymmetric_kernel_stride_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """2x2 mesh, non-square kernel and stride, stride != kernel on both axes."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=(3, 4),
            height=16,
            width=16,
            stride=(2, 2),
            padding=(1, 1),
        )

    def test_string_padding_rejected_with_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """String padding is not supported once a halo exchange is needed."""
        conv2d = conv.DistributedConv2D(
            8,
            12,
            4,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=2,
            padding="same",
            dilation=1,
            groups=1,
            bias=True,
            parallelism=process_mesh_dict_2x2["model"].size(),
            bayesian=False,
            dtype=torch.float32,
        ).to(device)
        x = torch.randn(2, 4, 16, 8, device=device)
        with pytest.raises(NotImplementedError):
            conv2d(x)


class TestConv2dParallel:
    """Distributed Conv2d across arbitrary spatial and channel group sizes."""

    @pytest.mark.dist(4)
    def test_forward_backward_1x4(
        self, device: torch.device, process_mesh_dict_1x4: dict
    ) -> None:
        """Pure channel parallelism (4 channel ranks, no spatial split)."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
        )

    @pytest.mark.dist(4)
    def test_forward_backward_4x1(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        """Pure spatial parallelism (4 spatial ranks, no channel split)."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
        )

    @pytest.mark.dist(8)
    def test_forward_backward_2x4(
        self, device: torch.device, process_mesh_dict_2x4_rowcol: dict
    ) -> None:
        """Combined 2-way spatial and 4-way channel parallelism."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
        )

    @pytest.mark.dist(4)
    def test_halo_1x4(self, device: torch.device, process_mesh_dict_1x4: dict) -> None:
        """Halo regime with no spatial split (channel-only, group size 1 path)."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(4)
    def test_halo_4x1(self, device: torch.device, process_mesh_dict_4x1: dict) -> None:
        """Halo regime with 4-way spatial split (real cross-rank exchange)."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=32,
            width=32,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(8)
    def test_halo_2x4(
        self, device: torch.device, process_mesh_dict_2x4_rowcol: dict
    ) -> None:
        """Halo regime combined with 4-way channel parallelism."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(8)
    def test_halo_2x4_asymmetric_kernel_stride(
        self, device: torch.device, process_mesh_dict_2x4_rowcol: dict
    ) -> None:
        """2x4 mesh, non-square kernel and stride, stride < kernel on both axes."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=(3, 4),
            height=17,
            width=16,
            stride=(2, 3),
            padding=(1, 2),
        )

    @pytest.mark.dist(8)
    def test_halo_2x4_asymmetric_stride_gt_kernel(
        self, device: torch.device, process_mesh_dict_2x4_rowcol: dict
    ) -> None:
        """2x4 mesh, non-square kernel and stride, width stride > kernel."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=(4, 2),
            height=16,
            width=18,
            stride=(2, 3),
            padding=(1, 0),
        )

    @pytest.mark.dist(8)
    def test_halo_2x4_asymmetric_dilation(
        self, device: torch.device, process_mesh_dict_2x4_rowcol: dict
    ) -> None:
        """2x4 mesh, non-square stride and dilation over the halo exchange."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=(3, 3),
            height=16,
            width=16,
            stride=(1, 2),
            padding=(2, 1),
            dilation=(2, 1),
        )


class TestConv2dTranspose:
    def test_param_init_conv2d_transpose(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Verify initial weights for Conv2d are correctly synchronized."""
        local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
        torch.manual_seed(local_rank)  # each process gets a different initial seed
        row_comm_size, col_comm_size = (
            process_mesh_dict_2x2["row"].size(),
            process_mesh_dict_2x2["col"].size(),
        )
        in_dim, out_dim = 6, 10
        kernel_size = 4
        conv2d = conv_transpose.DistributedConvTranspose2D(
            in_dim,
            out_dim,
            kernel_size,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=kernel_size,
            padding=(0,),
            dilation=1,
            groups=1,
            bias=True,
            parallelism=4,
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        weights_gathered_as_list = gather_results_as_list(
            conv2d.weight, process_mesh_dict_2x2
        )

        for i in range(process_mesh_dict_2x2["model"].size() // row_comm_size):
            assert torch.allclose(
                weights_gathered_as_list[i],
                weights_gathered_as_list[i + col_comm_size],
                atol=1e-5,
            )

    def test_conv2d_transpose_forwardbackward(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Verify initial weights for Conv2d are correctly synchronized."""
        local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
        torch.manual_seed(local_rank)  # each process gets a different initial seed
        row_comm_size, col_comm_size = (
            process_mesh_dict_2x2["row"].size(),
            process_mesh_dict_2x2["col"].size(),
        )
        in_dim, out_dim = 6, 10
        kernel_size = 4
        h, w = 16, 16

        conv2d = conv_transpose.DistributedConvTranspose2D(
            in_dim,
            out_dim,
            kernel_size,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=kernel_size,
            padding=(0,),
            dilation=1,
            groups=1,
            bias=True,
            parallelism=4,
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        weights_gathered_as_list = gather_results_as_list(
            conv2d.weight, process_mesh_dict_2x2
        )

        for i in range(process_mesh_dict_2x2["model"].size() // row_comm_size):
            assert torch.allclose(
                weights_gathered_as_list[i],
                weights_gathered_as_list[i + col_comm_size],
                atol=1e-5,
            )

        x = torch.randn(4, in_dim // 2, h, w // 2, device=device)
        y = conv2d(x)
        y.backward(torch.ones_like(y))
        assert isinstance(y, torch.Tensor)


# ---------------------------------------------------------------------------
# multi-spatial-dim parallelism: the domain is split along height *and* width
# ---------------------------------------------------------------------------
def _conv2d_multi_spatial_reference_check(
    device: torch.device,
    mesh: dict,
    in_channels: int,
    out_channels: int,
    kernel_size: Pair,
    height: int,
    width: int,
    batch: int = 2,
    stride: Pair | None = None,
    padding: Pair = 0,
    dilation: Pair = 1,
) -> None:
    """Compare a ``DistributedConv2D`` split along H *and* W to a plain conv2d.

    ``mesh`` carries ``sh`` (height), ``sw`` (width) and ``col`` (channel)
    groups plus a ``model`` group over every rank. Either spatial group may be
    size 1 (that axis is then not split). Covers the no-halo regime
    (``stride == kernel_size``) and the halo regime (any other stride, padding
    or dilation), and checks the output plus every gradient after gathering the
    sharded results.
    """
    h_group, w_group, c_group = mesh["sh"], mesh["sw"], mesh["col"]
    model_group = mesh["model"]
    nh, nw, nc = h_group.size(), w_group.size(), c_group.size()
    kernel_size = _pair(kernel_size)
    stride = kernel_size if stride is None else _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    torch.manual_seed(model_group.rank())
    x_local = torch.randn(
        batch,
        in_channels // nc,
        height // nh,
        width // nw,
        device=device,
    )

    conv2d = conv.DistributedConv2D(
        in_channels,
        out_channels,
        kernel_size,
        device=device,
        spatial_group=[h_group, w_group],
        channel_group=c_group,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,
        bias=True,
        parallelism=model_group.size(),
        bayesian=False,
        dtype=torch.float32,
    ).to(device)

    # When a halo exchange is needed, every split axis's global output must be
    # divisible by that axis's group size; otherwise the module rejects the call
    # cleanly rather than returning a wrong result. Predict that and check the
    # error path instead of the numeric one.
    split_idxs = [i for i, n in enumerate((nh, nw)) if n > 1]
    groups_per_axis = (nh, nw)
    global_sizes = (height, width)
    halo = not all(k == s for k, s in zip(kernel_size, stride)) or any(
        padding[i] != 0 or dilation[i] != 1 for i in split_idxs
    )
    if halo:
        for i in split_idxs:
            ext = dilation[i] * (kernel_size[i] - 1) + 1
            g_out = (global_sizes[i] + 2 * padding[i] - ext) // stride[i] + 1
            if g_out % groups_per_axis[i] != 0:
                with pytest.raises(NotImplementedError):
                    conv2d(x_local.clone().requires_grad_(True))
                return

    # --- reconstruct the equivalent global convolution ---------------------
    def _gather_spatial(t: torch.Tensor) -> torch.Tensor:
        t = gather_1d_results(t, mesh, dim=-3, pm_key="col")
        t = gather_1d_results(t, mesh, dim=-2, pm_key="sh")
        return gather_1d_results(t, mesh, dim=-1, pm_key="sw")

    weight_global = gather_1d_results(
        conv2d.weight.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    bias_global = gather_1d_results(
        conv2d.bias.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    x_global = _gather_spatial(x_local).clone()

    x_global.requires_grad_(True)
    weight_global.requires_grad_(True)
    bias_global.requires_grad_(True)

    y_ref = F.conv2d(
        x_global,
        weight_global,
        bias_global,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    assert y_ref.shape[-2] % nh == 0 and y_ref.shape[-1] % nw == 0, (
        f"global output {tuple(y_ref.shape[-2:])} not divisible by spatial "
        f"grid ({nh}, {nw})"
    )

    torch.manual_seed(2024)
    grad_global = torch.randn(y_ref.shape, device=device)
    y_ref.backward(grad_global)

    # --- distributed forward / backward on this rank's shard --------------
    x_dist = x_local.clone().requires_grad_(True)
    y = conv2d(x_dist)

    oc_local = out_channels // nc
    oh_local = y_ref.shape[-2] // nh
    ow_local = y_ref.shape[-1] // nw
    grad_local = grad_global[
        ...,
        c_group.rank() * oc_local : (c_group.rank() + 1) * oc_local,
        h_group.rank() * oh_local : (h_group.rank() + 1) * oh_local,
        w_group.rank() * ow_local : (w_group.rank() + 1) * ow_local,
    ].contiguous()
    y.backward(grad_local)

    # --- gather distributed results back to global tensors ---------------
    y_global = _gather_spatial(y.detach())
    x_grad_global = _gather_spatial(x_dist.grad)
    w_grad_global = gather_1d_results(conv2d.weight.grad, mesh, dim=0, pm_key="col")
    b_grad_global = gather_1d_results(conv2d.bias.grad, mesh, dim=0, pm_key="col")

    assert torch.allclose(y_global, y_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(x_grad_global, x_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(w_grad_global, weight_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(b_grad_global, bias_global.grad, atol=1e-4, rtol=1e-4)


class TestConv2dMultiSpatial:
    """Distributed Conv2d with the domain split along several spatial axes."""

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), MULTI_CASES_2D, ids=MULTI_IDS_2D
    )
    def test_hw_split_arbitrary_params(
        self,
        device: torch.device,
        spatial_mesh_2x2_hw: dict,
        kernel: Pair,
        stride: Pair,
        padding: Pair,
        dilation: Pair,
    ) -> None:
        """2x2 height/width split: every kernel/stride/padding/dilation combo.

        Each combo is either matched against a single-process conv2d or (when
        its global output is not divisible by the spatial group) asserted to be
        rejected with ``NotImplementedError`` -- never silently wrong.
        """
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            height=24,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(4)
    def test_hw_split_no_halo(
        self, device: torch.device, spatial_mesh_2x2_hw: dict
    ) -> None:
        """2x2 height/width split, stride == kernel (no communication)."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw,
            in_channels=6,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
        )

    @pytest.mark.dist(4)
    def test_hw_split_halo_stride_lt_kernel(
        self, device: torch.device, spatial_mesh_2x2_hw: dict
    ) -> None:
        """2x2 height/width split, overlapping windows on both axes."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw,
            in_channels=6,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(4)
    def test_hw_split_halo_asymmetric(
        self, device: torch.device, spatial_mesh_2x2_hw: dict
    ) -> None:
        """2x2 split, non-square kernel/stride/padding, stride != kernel."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw,
            in_channels=6,
            out_channels=12,
            kernel_size=(3, 4),
            height=16,
            width=16,
            stride=(2, 2),
            padding=(1, 1),
        )

    @pytest.mark.dist(4)
    def test_hw_split_halo_dilated(
        self, device: torch.device, spatial_mesh_2x2_hw: dict
    ) -> None:
        """2x2 split, dilated 'same'-style conv (stride 1) on both axes."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw,
            in_channels=6,
            out_channels=12,
            kernel_size=3,
            height=16,
            width=16,
            stride=1,
            padding=2,
            dilation=2,
        )

    @pytest.mark.dist(4)
    def test_hw_split_stride_eq_kernel_with_padding(
        self, device: torch.device, spatial_mesh_2x2_hw: dict
    ) -> None:
        """stride == kernel but split-axis padding forces the halo path."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw,
            in_channels=6,
            out_channels=12,
            kernel_size=2,
            height=14,
            width=14,
            stride=2,
            padding=(1, 1),
        )

    @pytest.mark.dist(4)
    def test_one_axis_split_plus_channel(
        self, device: torch.device, spatial_mesh_2x1_hw_x2c: dict
    ) -> None:
        """Height split only (width group size 1) combined with channel split."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x1_hw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(8)
    def test_hw_split_plus_channel_no_halo(
        self, device: torch.device, spatial_mesh_2x2_hw_x2c: dict
    ) -> None:
        """2x2 height/width split combined with 2-way channel parallelism."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
        )

    @pytest.mark.dist(8)
    def test_hw_split_plus_channel_halo(
        self, device: torch.device, spatial_mesh_2x2_hw_x2c: dict
    ) -> None:
        """2x2 height/width split + channel split, overlapping windows."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=24,
            width=24,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(4)
    def test_list_of_len_one_rejected(
        self, device: torch.device, spatial_mesh_2x2_hw: dict
    ) -> None:
        """A spatial_group sequence must have one entry per spatial axis."""
        with pytest.raises(ValueError):
            conv.DistributedConv2D(
                6,
                12,
                4,
                device=device,
                spatial_group=[spatial_mesh_2x2_hw["sw"]],
                channel_group=spatial_mesh_2x2_hw["col"],
                stride=4,
                parallelism=spatial_mesh_2x2_hw["model"].size(),
            ).to(device)
