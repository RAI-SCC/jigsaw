"""Distributed Conv2dTranspose with the domain split along several spatial axes.

Split out of ``test_conv2dtranspose.py``: the single-spatial-axis (width-only)
distributed transpose conv lives there, everything that splits height -- or
height *and* width, optionally combined with a channel split -- lives here.
"""

import pytest
import torch
import torch.nn.functional as F

import jigsaw.modules.conv2dtranspose as conv2dtranspose
from jigsaw.utils_aggregate import gather_1d_results
from tests.process_mesh import *  # noqa: F401,F403

pytestmark = [pytest.mark.dist_required]

Pair = int | tuple[int, int]


def _pair(value: Pair) -> tuple[int, int]:
    """Normalise an ``int`` or 2-tuple to a ``(h, w)`` tuple."""
    return (value, value) if isinstance(value, int) else tuple(value)


# (kernel, stride, padding, dilation) -- each entry is an int or an (h, w) tuple,
# so asymmetric kernels/strides/paddings/dilations are covered. Sweeps the
# no-halo regime, stride < / > kernel, unit-stride, dilated, stride == kernel
# forced onto the halo path by padding/dilation, and combos whose global output
# is not divisible by the spatial group (which must be rejected cleanly).
MULTI_CASES_2D = [
    (2, 2, 0, 1),
    (4, 2, 1, 1),
    (2, 4, 1, 1),
    (3, 1, 2, 2),
    ((3, 4), (2, 2), (1, 1), 1),
    ((4, 2), (2, 3), (1, 0), 1),
    (2, 2, 1, 1),
    ((3, 3), (3, 3), (0, 0), (1, 2)),
    (3, 2, 1, 1),
]
MULTI_IDS_2D = [
    f"k{k}_s{s}_p{p}_d{d}".replace(" ", "") for (k, s, p, d) in MULTI_CASES_2D
]


def _conv2d_transpose_multi_spatial_reference_check(
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
    channel_comm: str = "all_gather",
) -> None:
    """Compare a ``DistributedConv2DTranspose`` split along H *and* W to a plain op.

    ``mesh`` carries ``sh`` (height), ``sw`` (width) and ``col`` (channel) groups
    plus a ``model`` group over every rank. Either spatial group may be size 1
    (that axis is then not split). Covers the no-halo regime
    (``stride == kernel_size``) and the halo regime (any other stride, padding or
    dilation), and checks the output plus every gradient after gathering the
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

    conv = conv2dtranspose.DistributedConv2DTranspose(
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
        channel_comm=channel_comm,
    ).to(device)

    # --- reconstruct the equivalent global transpose convolution ----------
    # weight is (C_in, C_out // groups, kH, kW): the channel group shards dim 1.
    def _gather_spatial(t: torch.Tensor, channel_dim: int) -> torch.Tensor:
        t = gather_1d_results(t, mesh, dim=channel_dim, pm_key="col")
        t = gather_1d_results(t, mesh, dim=-2, pm_key="sh")
        return gather_1d_results(t, mesh, dim=-1, pm_key="sw")

    weight_global = gather_1d_results(
        conv.weight.detach(), mesh, dim=1, pm_key="col"
    ).clone()
    bias_global = gather_1d_results(
        conv.bias.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    x_global = _gather_spatial(x_local, -3).clone()

    x_global.requires_grad_(True)
    weight_global.requires_grad_(True)
    bias_global.requires_grad_(True)

    y_ref = F.conv_transpose2d(
        x_global,
        weight_global,
        bias_global,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    # When a halo exchange is needed, every split axis's global output must be
    # divisible by that axis's group size; otherwise the module rejects the call
    # cleanly rather than returning a wrong result.
    split_idxs = [i for i, n in enumerate((nh, nw)) if n > 1]
    groups_per_axis = (nh, nw)
    halo = not all(k == s for k, s in zip(kernel_size, stride)) or any(
        padding[i] != 0 or dilation[i] != 1 for i in split_idxs
    )
    if halo and any(y_ref.shape[-2:][i] % groups_per_axis[i] != 0 for i in split_idxs):
        with pytest.raises(NotImplementedError):
            conv(x_local.clone().requires_grad_(True))
        return

    torch.manual_seed(2024)
    grad_global = torch.randn(y_ref.shape, device=device)
    y_ref.backward(grad_global)

    # --- distributed forward / backward on this rank's shard -------------
    x_dist = x_local.clone().requires_grad_(True)
    y = conv(x_dist)

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

    # --- gather distributed results back to global tensors --------------
    y_global = _gather_spatial(y.detach(), -3)
    x_grad_global = _gather_spatial(x_dist.grad, -3)
    w_grad_global = gather_1d_results(conv.weight.grad, mesh, dim=1, pm_key="col")
    b_grad_global = gather_1d_results(conv.bias.grad, mesh, dim=0, pm_key="col")

    assert torch.allclose(y_global, y_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(x_grad_global, x_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(w_grad_global, weight_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(b_grad_global, bias_global.grad, atol=1e-4, rtol=1e-4)


class TestConv2dTransposeMultiSpatial:
    """Distributed Conv2dTranspose with the domain split along several spatial axes."""

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("channel_comm", ["all_gather", "p2p"])
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
        channel_comm: str,
    ) -> None:
        """2x2 height/width split: every kernel/stride/padding/dilation combo.

        Each combo is either matched against a single-process
        ``conv_transpose2d`` or (when its global output is not divisible by the
        spatial group) asserted to be rejected with ``NotImplementedError``.
        """
        _conv2d_transpose_multi_spatial_reference_check(
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
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(4)
    def test_hw_split_no_halo(
        self, device: torch.device, spatial_mesh_2x2_hw: dict
    ) -> None:
        """2x2 height/width split, stride == kernel (no communication)."""
        _conv2d_transpose_multi_spatial_reference_check(
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
        """2x2 height/width split, overlapping raw outputs on both axes."""
        _conv2d_transpose_multi_spatial_reference_check(
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
        _conv2d_transpose_multi_spatial_reference_check(
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
        """2x2 split, dilated 'same'-style transpose conv (stride 1) on both axes."""
        _conv2d_transpose_multi_spatial_reference_check(
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

    @pytest.mark.dist(2)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), MULTI_CASES_2D, ids=MULTI_IDS_2D
    )
    def test_one_axis_split_no_channel(
        self,
        device: torch.device,
        spatial_mesh_2x1_hw: dict,
        kernel: Pair,
        stride: Pair,
        padding: Pair,
        dilation: Pair,
    ) -> None:
        """Only the height axis split (width group size 1), no channel split.

        Exercises the degenerate ``spatial_groups`` grid ``(group_h, size-1)``
        across every kernel/stride/padding/dilation combo, including the
        ``channel_group.size() == 1`` ``_forward_local`` fast path.
        """
        _conv2d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x1_hw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            height=24,
            width=12,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(4)
    def test_one_axis_split_plus_channel(
        self, device: torch.device, spatial_mesh_2x1_hw_x2c: dict
    ) -> None:
        """Height split only (width group size 1) combined with channel split."""
        _conv2d_transpose_multi_spatial_reference_check(
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
    @pytest.mark.parametrize("channel_comm", ["all_gather", "p2p"])
    def test_hw_split_plus_channel_no_halo(
        self,
        device: torch.device,
        spatial_mesh_2x2_hw_x2c: dict,
        channel_comm: str,
    ) -> None:
        """2x2 height/width split combined with 2-way channel parallelism."""
        _conv2d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("channel_comm", ["all_gather", "p2p"])
    def test_hw_split_plus_channel_halo(
        self,
        device: torch.device,
        spatial_mesh_2x2_hw_x2c: dict,
        channel_comm: str,
    ) -> None:
        """2x2 height/width split, overlapping raw outputs, plus channel split."""
        _conv2d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
            channel_comm=channel_comm,
        )
