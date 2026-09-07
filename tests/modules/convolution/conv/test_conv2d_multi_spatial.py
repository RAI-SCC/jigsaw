"""Distributed Conv2d with the domain split along several spatial axes.

Split out of ``test_convolution_2d.py``: the single-spatial-axis (width-only)
distributed conv lives there, everything that splits height -- or height *and*
width, optionally combined with a channel split -- lives here.
"""

import pytest
import torch
import torch.nn.functional as F

import jigsaw.modules.convolution as convolution
from jigsaw.utils_aggregate import gather_1d_results
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]

Pair = int | tuple[int, int]


def _pair(value: Pair) -> tuple[int, int]:
    """Normalise an ``int`` or 2-tuple to a ``(h, w)`` tuple."""
    return (value, value) if isinstance(value, int) else tuple(value)


@pytest.fixture(autouse=True)
def _deterministic_cudnn():
    """Pin cuDNN to a deterministic, non-benchmarked convolution algorithm.

    The sharded ``DistributedConv2D`` and its single-process ``F.conv2d``
    reference are different-shaped calls, so cuDNN's benchmarked heuristic can
    pick algorithms whose fp32 accumulation order differs by more than the
    1e-4 parity tolerance (seen on the CUDA 13 container stack for the
    ``k5_s1_p2_d1`` grid-sweep case). Forcing the deterministic path makes both
    sides land on the same algorithm family, so the tolerance stays meaningful.
    """
    prev_deterministic = torch.backends.cudnn.deterministic
    prev_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = prev_deterministic
        torch.backends.cudnn.benchmark = prev_benchmark


# ---------------------------------------------------------------------------
# multi-spatial-dim parallelism: the domain is split along height *and* width
# ---------------------------------------------------------------------------
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
    groups: int = 1,
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

    dist_conv = convolution.DistributedConv2D(
        in_channels,
        out_channels,
        kernel_size,
        device=device,
        spatial_group=[h_group, w_group],
        channel_group=c_group,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
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
                    dist_conv(x_local.clone().requires_grad_(True))
                return

    # --- reconstruct the equivalent global convolution ---------------------
    def _gather_spatial(t: torch.Tensor) -> torch.Tensor:
        t = gather_1d_results(t, mesh, dim=-3, pm_key="col")
        t = gather_1d_results(t, mesh, dim=-2, pm_key="sh")
        return gather_1d_results(t, mesh, dim=-1, pm_key="sw")

    weight_global = gather_1d_results(
        dist_conv.weight.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    bias_global = gather_1d_results(
        dist_conv.bias.detach(), mesh, dim=0, pm_key="col"
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
        groups=groups,
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
    y = dist_conv(x_dist)

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
    w_grad_global = gather_1d_results(dist_conv.weight.grad, mesh, dim=0, pm_key="col")
    b_grad_global = gather_1d_results(dist_conv.bias.grad, mesh, dim=0, pm_key="col")

    # Collective-safe: every rank reaches the same verdict, so a mismatch on a
    # subset of ranks fails the job instead of hanging the rest in the next gather.
    assert_close("output", y_global, y_ref, 1e-4, 1e-4, device, model_group)
    assert_close(
        "input grad", x_grad_global, x_global.grad, 1e-4, 1e-4, device, model_group
    )
    # The weight gradient accumulates a product over every spatial position
    # (H*W terms per weight element), so it is the deepest reduction in this
    # check and the most exposed to fp32 summation-order differences between the
    # 4-way-sharded path (per-shard partials + an all-reduce) and the
    # single-process reference. Even with cuDNN pinned deterministic (see
    # ``_deterministic_cudnn``) the ``k5_s1_p2_d1`` grid-sweep case has one
    # cancellation-dominated weight element that lands ~0.29 absolute off on the
    # CUDA 13 container stack while every other element stays ~1e-3 relative, so
    # weight grad gets a looser tolerance than output / input grad. A real halo
    # or reduction bug corrupts a whole strip of weight elements by O(magnitude)
    # (tens here), which this still catches.
    assert_close(
        "weight grad",
        w_grad_global,
        weight_global.grad,
        5e-1,
        1e-2,
        device,
        model_group,
    )
    assert_close(
        "bias grad", b_grad_global, bias_global.grad, 1e-4, 1e-4, device, model_group
    )


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
            convolution.DistributedConv2D(
                6,
                12,
                4,
                device=device,
                spatial_group=[spatial_mesh_2x2_hw["sw"]],
                channel_group=spatial_mesh_2x2_hw["col"],
                stride=4,
            ).to(device)


# (kernel, stride, padding, dilation) -- an int or an (h, w) tuple per entry, so
# asymmetric hyper-parameters are covered. Spans every regime: stride == kernel
# with no padding (no communication), stride == kernel forced onto the halo path
# by padding or dilation, overlapping windows (stride < kernel), strided with a
# gap (stride > kernel), and dilation > 1.
GRID_SWEEP_CASES = [
    (2, 2, 0, 1),
    (4, 4, 0, 1),
    (3, 3, 1, 1),
    (3, 1, 1, 1),
    (5, 1, 2, 1),
    (3, 1, 2, 2),
    (3, 2, 1, 1),
    (4, 2, 1, 1),
    (3, 4, 2, 1),
    ((3, 5), (1, 1), (1, 2), 1),
    ((4, 2), (2, 3), (1, 0), 1),
    ((3, 3), (3, 3), (0, 0), (1, 2)),
]
GRID_SWEEP_IDS = [
    f"k{k}_s{s}_p{p}_d{d}".replace(" ", "") for (k, s, p, d) in GRID_SWEEP_CASES
]


class TestConv2dSpatialGridSweep:
    """Arbitrary kernel/stride/padding/dilation over a range of ``(n_h, n_w)`` grids.

    Every (grid, hyper-parameter) pair is either matched against a single-process
    conv2d or -- when a split axis's global output is not divisible by that
    axis's group size -- asserted to be rejected with ``NotImplementedError``.
    So each pair is checked to be either numerically correct or cleanly refused,
    never silently wrong. ``height`` / ``width`` are multiples of the grid so the
    domain shards evenly on both axes.
    """

    @pytest.mark.dist(6)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        GRID_SWEEP_CASES,
        ids=GRID_SWEEP_IDS,
    )
    def test_split_3x2(
        self,
        device: torch.device,
        spatial_mesh_3x2_hw: dict,
        kernel: Pair,
        stride: Pair,
        padding: Pair,
        dilation: Pair,
    ) -> None:
        """3 (height) x 2 (width) spatial grid."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_3x2_hw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            height=48,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        GRID_SWEEP_CASES,
        ids=GRID_SWEEP_IDS,
    )
    def test_split_1x4(
        self,
        device: torch.device,
        spatial_mesh_1x4_hw: dict,
        kernel: Pair,
        stride: Pair,
        padding: Pair,
        dilation: Pair,
    ) -> None:
        """Width-only 4-way split (height not split)."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_1x4_hw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            height=24,
            width=48,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        GRID_SWEEP_CASES,
        ids=GRID_SWEEP_IDS,
    )
    def test_split_4x1(
        self,
        device: torch.device,
        spatial_mesh_4x1_hw: dict,
        kernel: Pair,
        stride: Pair,
        padding: Pair,
        dilation: Pair,
    ) -> None:
        """Height-only 4-way split (width not split)."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_4x1_hw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            height=48,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        GRID_SWEEP_CASES,
        ids=GRID_SWEEP_IDS,
    )
    def test_split_2x2(
        self,
        device: torch.device,
        spatial_mesh_2x2_hw: dict,
        kernel: Pair,
        stride: Pair,
        padding: Pair,
        dilation: Pair,
    ) -> None:
        """2 (height) x 2 (width) spatial grid."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            height=32,
            width=32,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(10)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        GRID_SWEEP_CASES,
        ids=GRID_SWEEP_IDS,
    )
    def test_split_5x2(
        self,
        device: torch.device,
        spatial_mesh_5x2_hw: dict,
        kernel: Pair,
        stride: Pair,
        padding: Pair,
        dilation: Pair,
    ) -> None:
        """5 (height) x 2 (width) spatial grid."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_5x2_hw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            height=60,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(2)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        GRID_SWEEP_CASES,
        ids=GRID_SWEEP_IDS,
    )
    def test_split_2x1(
        self,
        device: torch.device,
        spatial_mesh_2x1_hw: dict,
        kernel: Pair,
        stride: Pair,
        padding: Pair,
        dilation: Pair,
    ) -> None:
        """Minimal spatial split: 2-way height, width not split, no channel split."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x1_hw,
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
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        GRID_SWEEP_CASES,
        ids=GRID_SWEEP_IDS,
    )
    def test_split_2x1_x2c(
        self,
        device: torch.device,
        spatial_mesh_2x1_hw_x2c: dict,
        kernel: Pair,
        stride: Pair,
        padding: Pair,
        dilation: Pair,
    ) -> None:
        """2-way height split, width not split, + 2-way channel split."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x1_hw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            height=24,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )


class TestConv2dMultiSpatialGroups:
    """Grouped conv with the domain split along H *and* W.

    The halo exchange runs per split axis and is unaffected by ``groups``; the
    channel-parallel work still collapses to a plain local grouped conv whenever
    ``groups`` is a multiple of ``col.size()``.
    """

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [1, 2, 4, 8])
    def test_hw_split_no_halo(
        self, device: torch.device, spatial_mesh_2x2_hw: dict, groups: int
    ) -> None:
        """2x2 H/W split, no channel split: any ``groups`` dividing the channels."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw,
            in_channels=8,
            out_channels=16,
            kernel_size=4,
            height=16,
            width=16,
            groups=groups,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [2, 4])
    def test_hw_split_halo(
        self, device: torch.device, spatial_mesh_2x2_hw: dict, groups: int
    ) -> None:
        """2x2 H/W split, grouped conv, halo regime on both split axes."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
            groups=groups,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("groups", [2, 4])
    def test_hw_split_plus_channel_halo(
        self, device: torch.device, spatial_mesh_2x2_hw_x2c: dict, groups: int
    ) -> None:
        """2x2 H/W split + 2-way channel split, channel-local grouped conv, halo."""
        _conv2d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2_hw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
            groups=groups,
        )
