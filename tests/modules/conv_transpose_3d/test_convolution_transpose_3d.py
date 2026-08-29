import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import jigsaw.modules.conv_transpose_3d as conv_transpose_3d
from jigsaw.utils_aggregate import gather_1d_results, gather_results_as_list
from tests.process_mesh import *

pytestmark = [pytest.mark.dist_required]

Triple = int | tuple[int, int, int]


def _triple(value: Triple) -> tuple[int, int, int]:
    """Normalise an ``int`` or 3-tuple to a ``(d, h, w)`` tuple."""
    return (value, value, value) if isinstance(value, int) else tuple(value)


# (kernel, stride, padding, dilation) -- each entry is an int or a (d, h, w)
# tuple, so asymmetric kernels/strides/paddings/dilations are covered. The split
# axis is the width (last) axis. The last few entries exercise the
# ``stride == kernel`` config branch with non-trivial split-axis padding/dilation
# (must still route through the halo path) and a combo whose global output width
# is not divisible by an even spatial group (must raise ``NotImplementedError``).
ARBITRARY_CASES = [
    (2, 2, 0, 1),
    (4, 2, 1, 1),
    (2, 4, 1, 1),
    (3, 1, 2, 2),
    ((2, 3, 5), (2, 2, 3), (0, 1, 1), 1),
    ((3, 3, 6), (2, 2, 2), (1, 1, 2), 1),
    (2, 2, 1, 1),
    ((3, 3, 3), (3, 3, 3), (0, 0, 0), (1, 1, 2)),
    (3, 2, 1, 1),
]
ARBITRARY_IDS = [f"k{k}_s{s}_p{p}_d{d}".replace(" ", "") for (k, s, p, d) in ARBITRARY_CASES]


def _conv3d_transpose_reference_check(
    device: torch.device,
    mesh: dict,
    in_channels: int,
    out_channels: int,
    kernel_size: Triple,
    depth: int,
    height: int,
    width: int,
    batch: int = 2,
    stride: Triple | None = None,
    padding: Triple = 0,
    dilation: Triple = 1,
) -> None:
    """Compare ``DistributedConvTranspose3D`` against a single-process op.

    Works for any ``row.size()`` (spatial, splits the width axis) x ``col.size()``
    (channel) mesh and for both the no-halo regime (``stride == kernel_size``) and
    the halo regime (any other stride, or ``stride == kernel_size`` with
    split-axis padding/dilation). Rebuilds the equivalent non-distributed
    transpose convolution from the sharded weight, bias and input, runs forward
    and backward on both, and checks the output plus every gradient after
    gathering the distributed shards.

    When the global output width is not divisible by the spatial group size the
    distributed layer cannot shard the output evenly; this asserts it raises
    ``NotImplementedError`` instead of returning a wrong result.
    """
    row_group, col_group = mesh["row"], mesh["col"]
    n_spatial, n_channel = row_group.size(), col_group.size()
    model_group = mesh["model"]
    kernel_size = _triple(kernel_size)
    stride = kernel_size if stride is None else _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    torch.manual_seed(model_group.rank())
    x_local = torch.randn(
        batch,
        in_channels // n_channel,
        depth,
        height,
        width // n_spatial,
        device=device,
    )

    conv = conv_transpose_3d.DistributedConvTranspose3D(
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

    # --- reconstruct the equivalent global transpose convolution -----------
    # weight is (C_in, C_out // groups, kD, kH, kW): the channel group shards dim 1.
    weight_global = gather_1d_results(
        conv.weight.detach(), mesh, dim=1, pm_key="col"
    ).clone()
    bias_global = gather_1d_results(
        conv.bias.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    x_global = gather_1d_results(x_local, mesh, dim=-4, pm_key="col")
    x_global = gather_1d_results(x_global, mesh, dim=-1, pm_key="row").clone()

    x_global.requires_grad_(True)
    weight_global.requires_grad_(True)
    bias_global.requires_grad_(True)

    y_ref = F.conv_transpose3d(
        x_global,
        weight_global,
        bias_global,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    # global output not divisible by the spatial ranks -> unsupported, cleanly.
    if n_spatial > 1 and y_ref.shape[-1] % n_spatial != 0:
        with pytest.raises(NotImplementedError):
            conv(x_local.clone().requires_grad_(True))
        return

    torch.manual_seed(2024)
    grad_global = torch.randn(y_ref.shape, device=device)
    y_ref.backward(grad_global)

    # --- distributed forward / backward on this rank's shard --------------
    x_dist = x_local.clone().requires_grad_(True)
    y = conv(x_dist)

    out_local = out_channels // n_channel
    ow_local = y_ref.shape[-1] // n_spatial
    grad_local = grad_global[
        ...,
        col_group.rank() * out_local : (col_group.rank() + 1) * out_local,
        :,
        :,
        row_group.rank() * ow_local : (row_group.rank() + 1) * ow_local,
    ].contiguous()
    y.backward(grad_local)

    # --- gather distributed results back to global tensors ---------------
    def _to_global(t: torch.Tensor, channel_dim: int) -> torch.Tensor:
        t = gather_1d_results(t, mesh, dim=channel_dim, pm_key="col")
        return gather_1d_results(t, mesh, dim=-1, pm_key="row")

    y_global = _to_global(y.detach(), -4)
    x_grad_global = _to_global(x_dist.grad, -4)
    w_grad_global = gather_1d_results(conv.weight.grad, mesh, dim=1, pm_key="col")
    b_grad_global = gather_1d_results(conv.bias.grad, mesh, dim=0, pm_key="col")

    assert torch.allclose(y_global, y_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(x_grad_global, x_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(w_grad_global, weight_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(b_grad_global, bias_global.grad, atol=1e-4, rtol=1e-4)


class TestConv3dTranspose:
    def test_param_init(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Initial weights are synchronized across model-parallel replicas."""
        local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
        torch.manual_seed(local_rank)
        row_comm_size, col_comm_size = (
            process_mesh_dict_2x2["row"].size(),
            process_mesh_dict_2x2["col"].size(),
        )
        conv = conv_transpose_3d.DistributedConvTranspose3D(
            6,
            12,
            2,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=2,
            padding=(0,),
            dilation=1,
            groups=1,
            bias=True,
            parallelism=4,
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        weights_gathered = gather_results_as_list(conv.weight, process_mesh_dict_2x2)
        for i in range(process_mesh_dict_2x2["model"].size() // row_comm_size):
            assert torch.allclose(
                weights_gathered[i], weights_gathered[i + col_comm_size], atol=1e-5
            )

    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_arbitrary_params(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """2x2 mesh: arbitrary kernel/stride/padding/dilation, incl. asymmetric."""
        _conv3d_transpose_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            depth=4,
            height=4,
            width=12,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def test_string_padding_rejected_with_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """String padding is not supported once a halo exchange is needed."""
        conv = conv_transpose_3d.DistributedConvTranspose3D(
            8,
            12,
            3,
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
        x = torch.randn(2, 4, 4, 4, 8, device=device)
        with pytest.raises(NotImplementedError):
            conv(x)


class TestConv3dTransposeParallel:
    """DistributedConvTranspose3d across arbitrary spatial and channel sizes."""

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_arbitrary_params_1x4(
        self,
        device: torch.device,
        process_mesh_dict_1x4: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """Pure channel parallelism: every param combo works (no spatial split)."""
        _conv3d_transpose_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            depth=4,
            height=4,
            width=12,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_arbitrary_params_4x1(
        self,
        device: torch.device,
        process_mesh_dict_4x1: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """Pure 4-way spatial split with arbitrary params (real halo exchange)."""
        _conv3d_transpose_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            depth=4,
            height=4,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_arbitrary_params_2x4(
        self,
        device: torch.device,
        process_mesh_dict_2x4_rowcol: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """2-way spatial x 4-way channel with arbitrary params."""
        _conv3d_transpose_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            depth=4,
            height=4,
            width=12,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )


# ---------------------------------------------------------------------------
# multi-spatial-dim parallelism: the domain is split along depth / height /
# width simultaneously.
# ---------------------------------------------------------------------------
def _conv3d_transpose_multi_spatial_reference_check(
    device: torch.device,
    mesh: dict,
    in_channels: int,
    out_channels: int,
    kernel_size: Triple,
    depth: int,
    height: int,
    width: int,
    batch: int = 2,
    stride: Triple | None = None,
    padding: Triple = 0,
    dilation: Triple = 1,
) -> None:
    """Compare a ``DistributedConvTranspose3D`` split along D/H/W to a plain op.

    ``mesh`` carries ``sd`` (depth), ``sh`` (height), ``sw`` (width) and ``col``
    (channel) groups plus a ``model`` group over every rank. Any spatial group
    may be size 1 (that axis is then not split). Covers the no-halo regime
    (``stride == kernel_size``) and the halo regime, and checks the output plus
    every gradient after gathering the sharded results.
    """
    d_group, h_group, w_group, c_group = (
        mesh["sd"],
        mesh["sh"],
        mesh["sw"],
        mesh["col"],
    )
    model_group = mesh["model"]
    nd, nh, nw, nc = (
        d_group.size(),
        h_group.size(),
        w_group.size(),
        c_group.size(),
    )
    kernel_size = _triple(kernel_size)
    stride = kernel_size if stride is None else _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    torch.manual_seed(model_group.rank())
    x_local = torch.randn(
        batch,
        in_channels // nc,
        depth // nd,
        height // nh,
        width // nw,
        device=device,
    )

    conv = conv_transpose_3d.DistributedConvTranspose3D(
        in_channels,
        out_channels,
        kernel_size,
        device=device,
        spatial_group=[d_group, h_group, w_group],
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
    split_idxs = [i for i, n in enumerate((nd, nh, nw)) if n > 1]
    groups_per_axis = (nd, nh, nw)
    global_sizes = (depth, height, width)
    halo = not all(k == s for k, s in zip(kernel_size, stride)) or any(
        padding[i] != 0 or dilation[i] != 1 for i in split_idxs
    )
    if halo:
        for i in split_idxs:
            ext = dilation[i] * (kernel_size[i] - 1) + 1
            g_out = (global_sizes[i] - 1) * stride[i] - 2 * padding[i] + ext
            if g_out % groups_per_axis[i] != 0:
                with pytest.raises(NotImplementedError):
                    conv(x_local.clone().requires_grad_(True))
                return

    def _gather_spatial(t: torch.Tensor) -> torch.Tensor:
        t = gather_1d_results(t, mesh, dim=-4, pm_key="col")
        t = gather_1d_results(t, mesh, dim=-3, pm_key="sd")
        t = gather_1d_results(t, mesh, dim=-2, pm_key="sh")
        return gather_1d_results(t, mesh, dim=-1, pm_key="sw")

    # weight is (C_in, C_out // groups, kD, kH, kW): the channel group shards dim 1.
    weight_global = gather_1d_results(
        conv.weight.detach(), mesh, dim=1, pm_key="col"
    ).clone()
    bias_global = gather_1d_results(
        conv.bias.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    x_global = _gather_spatial(x_local).clone()

    x_global.requires_grad_(True)
    weight_global.requires_grad_(True)
    bias_global.requires_grad_(True)

    y_ref = F.conv_transpose3d(
        x_global,
        weight_global,
        bias_global,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    assert (
        y_ref.shape[-3] % nd == 0
        and y_ref.shape[-2] % nh == 0
        and y_ref.shape[-1] % nw == 0
    ), (
        f"global output {tuple(y_ref.shape[-3:])} not divisible by spatial "
        f"grid ({nd}, {nh}, {nw})"
    )

    torch.manual_seed(2024)
    grad_global = torch.randn(y_ref.shape, device=device)
    y_ref.backward(grad_global)

    x_dist = x_local.clone().requires_grad_(True)
    y = conv(x_dist)

    oc_local = out_channels // nc
    od_local = y_ref.shape[-3] // nd
    oh_local = y_ref.shape[-2] // nh
    ow_local = y_ref.shape[-1] // nw
    grad_local = grad_global[
        ...,
        c_group.rank() * oc_local : (c_group.rank() + 1) * oc_local,
        d_group.rank() * od_local : (d_group.rank() + 1) * od_local,
        h_group.rank() * oh_local : (h_group.rank() + 1) * oh_local,
        w_group.rank() * ow_local : (w_group.rank() + 1) * ow_local,
    ].contiguous()
    y.backward(grad_local)

    y_global = _gather_spatial(y.detach())
    x_grad_global = _gather_spatial(x_dist.grad)
    w_grad_global = gather_1d_results(conv.weight.grad, mesh, dim=1, pm_key="col")
    b_grad_global = gather_1d_results(conv.bias.grad, mesh, dim=0, pm_key="col")

    assert torch.allclose(y_global, y_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(x_grad_global, x_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(w_grad_global, weight_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(b_grad_global, bias_global.grad, atol=1e-4, rtol=1e-4)


class TestConvTranspose3dMultiSpatial:
    """Distributed ConvTranspose3d with the domain split along several axes."""

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_dh_split_arbitrary_params(
        self,
        device: torch.device,
        spatial_mesh_2x2x1_dhw: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """2x2 (depth, height) split: every kernel/stride/padding/dilation combo.

        Each combo is either matched against a single-process conv_transpose3d
        or (when its global output is not divisible by the spatial group)
        asserted to be rejected with ``NotImplementedError`` -- never silently
        wrong.
        """
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            depth=12,
            height=12,
            width=6,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(4)
    def test_dh_split_no_halo(
        self, device: torch.device, spatial_mesh_2x2x1_dhw: dict
    ) -> None:
        """2 (depth) x 2 (height) split, stride == kernel (no communication)."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=2,
            depth=8,
            height=8,
            width=8,
        )

    @pytest.mark.dist(4)
    def test_dh_split_halo(
        self, device: torch.device, spatial_mesh_2x2x1_dhw: dict
    ) -> None:
        """2 (depth) x 2 (height) split, overlapping raw outputs on both axes."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=3,
            depth=12,
            height=12,
            width=12,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(4)
    def test_dw_split_halo(
        self, device: torch.device, spatial_mesh_2x1x2_dhw: dict
    ) -> None:
        """Depth and width split, height not split, overlapping raw outputs."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x1x2_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=3,
            depth=12,
            height=6,
            width=12,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(4)
    def test_dh_split_halo_asymmetric(
        self, device: torch.device, spatial_mesh_2x2x1_dhw: dict
    ) -> None:
        """2x2 (D, H) split, non-square kernel/stride/padding, stride != kernel."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=(2, 3, 3),
            depth=12,
            height=12,
            width=12,
            stride=(2, 2, 1),
            padding=(0, 1, 1),
        )

    @pytest.mark.dist(8)
    def test_dhw_split_no_halo(
        self, device: torch.device, spatial_mesh_2x2x2_dhw: dict
    ) -> None:
        """Full 2x2x2 depth/height/width split, stride == kernel."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x2_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=2,
            depth=8,
            height=8,
            width=8,
        )

    @pytest.mark.dist(8)
    def test_dhw_split_halo(
        self, device: torch.device, spatial_mesh_2x2x2_dhw: dict
    ) -> None:
        """Full 2x2x2 split, overlapping raw outputs on all axes."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x2_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=3,
            depth=12,
            height=12,
            width=12,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(8)
    def test_dh_split_plus_channel_halo(
        self, device: torch.device, spatial_mesh_2x2x1_dhw_x2c: dict
    ) -> None:
        """2x2 (D, H) split combined with 2-way channel parallelism."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            depth=16,
            height=16,
            width=8,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(4)
    def test_wrong_length_sequence_rejected(
        self, device: torch.device, spatial_mesh_2x2x1_dhw: dict
    ) -> None:
        """A spatial_group sequence must have one entry per spatial axis."""
        with pytest.raises(ValueError):
            conv_transpose_3d.DistributedConvTranspose3D(
                6,
                12,
                2,
                device=device,
                spatial_group=[
                    spatial_mesh_2x2x1_dhw["sd"],
                    spatial_mesh_2x2x1_dhw["sh"],
                ],
                channel_group=spatial_mesh_2x2x1_dhw["col"],
                stride=2,
                parallelism=spatial_mesh_2x2x1_dhw["model"].size(),
            ).to(device)
