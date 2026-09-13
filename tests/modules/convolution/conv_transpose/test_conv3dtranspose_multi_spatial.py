"""Distributed Conv3dTranspose with the domain split along several spatial axes.

Split out of ``test_conv3dtranspose.py``: the single-spatial-axis (width-only)
distributed transpose conv lives there, everything that splits depth / height --
or any combination, optionally with a channel split -- lives here.
"""

import pytest
import torch
import torch.nn.functional as F

import jigsaw.modules.convolution as convolution
from jigsaw.utils_aggregate import gather_1d_results
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]

Triple = int | tuple[int, int, int]


def _triple(value: Triple) -> tuple[int, int, int]:
    """Normalise an ``int`` or 3-tuple to a ``(d, h, w)`` tuple."""
    return (value, value, value) if isinstance(value, int) else tuple(value)


# (kernel, stride, padding, dilation) -- int or (d, h, w) tuple. Sweeps the
# no-halo regime, stride < / > kernel, unit-stride, dilated and stride == kernel
# forced onto the halo path by padding/dilation.
MULTI_CASES_3D = [
    (2, 2, 0, 1),
    (4, 2, 1, 1),
    (2, 4, 1, 1),
    (3, 1, 2, 2),
    ((2, 3, 4), (2, 2, 2), (0, 1, 1), 1),
    (2, 2, 1, 1),
    ((3, 3, 3), (3, 3, 3), (0, 0, 0), (1, 1, 2)),
]
MULTI_IDS_3D = [
    f"k{k}_s{s}_p{p}_d{d}".replace(" ", "") for (k, s, p, d) in MULTI_CASES_3D
]


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
    groups: int = 1,
    channel_comm: str = "all_gather",
) -> None:
    """Compare a ``DistributedConv3DTranspose`` split along D/H/W to a plain op.

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

    conv = convolution.DistributedConv3DTranspose(
        in_channels,
        out_channels,
        kernel_size,
        device=device,
        spatial_group=[d_group, h_group, w_group],
        channel_group=c_group,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
        bayesian=False,
        dtype=torch.float32,
        channel_comm=channel_comm,
    ).to(device)

    # weight is (C_in, C_out // groups, kD, kH, kW). The gather regime
    # (groups == 1) shards the output-channel axis (dim 1); the channel-local
    # regime (groups a multiple of col.size()) shards the input-channel axis
    # (dim 0) instead.
    weight_dim = 0 if conv._channel_local else 1

    def _gather_spatial(t: torch.Tensor, channel_dim: int) -> torch.Tensor:
        t = gather_1d_results(t, mesh, dim=channel_dim, pm_key="col")
        t = gather_1d_results(t, mesh, dim=-3, pm_key="sd")
        t = gather_1d_results(t, mesh, dim=-2, pm_key="sh")
        return gather_1d_results(t, mesh, dim=-1, pm_key="sw")

    weight_global = gather_1d_results(
        conv.weight.detach(), mesh, dim=weight_dim, pm_key="col"
    ).clone()
    bias_global = gather_1d_results(
        conv.bias.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    x_global = _gather_spatial(x_local, -4).clone()

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
        groups=groups,
    )

    # When a halo exchange is needed, every split axis's global output must be
    # divisible by that axis's group size; otherwise the module rejects the call
    # cleanly rather than returning a wrong result.
    split_idxs = [i for i, n in enumerate((nd, nh, nw)) if n > 1]
    groups_per_axis = (nd, nh, nw)
    halo = not all(k == s for k, s in zip(kernel_size, stride)) or any(
        padding[i] != 0 or dilation[i] != 1 for i in split_idxs
    )
    if halo and any(y_ref.shape[-3:][i] % groups_per_axis[i] != 0 for i in split_idxs):
        with pytest.raises(NotImplementedError):
            conv(x_local.clone().requires_grad_(True))
        return

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

    y_global = _gather_spatial(y.detach(), -4)
    x_grad_global = _gather_spatial(x_dist.grad, -4)
    w_grad_global = gather_1d_results(
        conv.weight.grad, mesh, dim=weight_dim, pm_key="col"
    )
    b_grad_global = gather_1d_results(conv.bias.grad, mesh, dim=0, pm_key="col")

    # Collective-safe: every rank reaches the same verdict, so a mismatch on a
    # subset of ranks fails the job instead of hanging the rest in the next gather.
    assert_close("output", y_global, y_ref, 1e-4, 1e-4, device, model_group)
    assert_close(
        "input grad", x_grad_global, x_global.grad, 1e-4, 1e-4, device, model_group
    )
    # The parameter grads reduce over every output position; across a big volume
    # the distributed partial sums land in a different float32 order than the
    # single-process reference, so allow ~1e-3 there.
    assert_close(
        "weight grad",
        w_grad_global,
        weight_global.grad,
        1e-3,
        1e-3,
        device,
        model_group,
    )
    assert_close(
        "bias grad", b_grad_global, bias_global.grad, 1e-3, 1e-3, device, model_group
    )


class TestConv3dTransposeMultiSpatial:
    """Distributed Conv3dTranspose with the domain split along several spatial axes."""

    @pytest.mark.dist(2)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), MULTI_CASES_3D, ids=MULTI_IDS_3D
    )
    def test_d_split_only_arbitrary_params(
        self,
        device: torch.device,
        spatial_mesh_2x1x1_dhw: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """Only one of the three axes split: 2 (depth), height/width not split."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x1x1_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            depth=16,
            height=8,
            width=8,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), MULTI_CASES_3D, ids=MULTI_IDS_3D
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
        """2x2 depth/height split (width not split): every param combo."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            depth=16,
            height=16,
            width=8,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("channel_comm", ["all_gather", "p2p"])
    def test_dw_split_halo(
        self,
        device: torch.device,
        spatial_mesh_2x1x2_dhw: dict,
        channel_comm: str,
    ) -> None:
        """2x2 depth/width split (height not split), overlapping raw outputs."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x1x2_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=4,
            depth=16,
            height=8,
            width=16,
            stride=2,
            padding=1,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(8)
    def test_dhw_split_no_halo(
        self, device: torch.device, spatial_mesh_2x2x2_dhw: dict
    ) -> None:
        """2x2x2 depth/height/width split, stride == kernel (no communication)."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x2_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=2,
            depth=16,
            height=16,
            width=16,
        )

    @pytest.mark.dist(8)
    def test_dhw_split_halo(
        self, device: torch.device, spatial_mesh_2x2x2_dhw: dict
    ) -> None:
        """2x2x2 depth/height/width split, overlapping raw outputs on every axis."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x2_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=4,
            depth=16,
            height=16,
            width=16,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("channel_comm", ["all_gather", "p2p"])
    def test_dh_split_plus_channel_halo(
        self,
        device: torch.device,
        spatial_mesh_2x2x1_dhw_x2c: dict,
        channel_comm: str,
    ) -> None:
        """2x2 depth/height split plus 2-way channel parallelism, halo regime."""
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
            channel_comm=channel_comm,
        )


class TestConv3dTransposeMultiSpatialGroups:
    """Grouped transpose conv3d with the domain split along several spatial axes.

    The per-axis halo exchange is unaffected by ``groups``; the channel-parallel
    work still collapses to a plain local grouped transpose conv3d whenever
    ``groups`` is a multiple of ``col.size()`` (weight then sharded on the
    input-channel axis 0 instead of the output axis 1).
    """

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [1, 2, 4, 8])
    def test_dh_split_no_halo(
        self, device: torch.device, spatial_mesh_2x2x1_dhw: dict, groups: int
    ) -> None:
        """2x2 D/H split, no channel split: any ``groups`` dividing the channels."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw,
            in_channels=8,
            out_channels=16,
            kernel_size=2,
            depth=16,
            height=16,
            width=8,
            groups=groups,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [2, 4])
    def test_dh_split_halo(
        self, device: torch.device, spatial_mesh_2x2x1_dhw: dict, groups: int
    ) -> None:
        """2x2 D/H split, grouped transpose conv3d, halo regime on both split axes."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            depth=16,
            height=16,
            width=8,
            stride=2,
            padding=1,
            groups=groups,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("groups", [2, 4])
    @pytest.mark.parametrize("channel_comm", ["all_gather", "p2p"])
    def test_dh_split_plus_channel(
        self,
        device: torch.device,
        spatial_mesh_2x2x1_dhw_x2c: dict,
        groups: int,
        channel_comm: str,
    ) -> None:
        """2x2 D/H split + 2-way channel split, channel-local grouped transpose conv3d."""
        _conv3d_transpose_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=2,
            depth=16,
            height=16,
            width=8,
            groups=groups,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("groups", [2, 4])
    def test_dh_split_plus_channel_halo(
        self, device: torch.device, spatial_mesh_2x2x1_dhw_x2c: dict, groups: int
    ) -> None:
        """2x2 D/H split + 2-way channel split, channel-local grouped transpose, halo."""
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
            groups=groups,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("groups", [5, 7])
    def test_indivisible_groups_rejected(
        self, device: torch.device, spatial_mesh_2x2x1_dhw_x2c: dict, groups: int
    ) -> None:
        """``groups`` not dividing the channel counts fails fast at construction.

        (``1 < groups`` not a multiple of ``col.size()`` -- e.g. ``groups=3``
        with a 2-way channel group -- is now the supported channel-grouped
        regime, so it is no longer rejected.)
        """
        mesh = spatial_mesh_2x2x1_dhw_x2c
        with pytest.raises(ValueError):
            convolution.DistributedConv3DTranspose(
                12,
                12,
                2,
                device=device,
                spatial_group=[mesh["sd"], mesh["sh"], mesh["sw"]],
                channel_group=mesh["col"],
                stride=2,
                groups=groups,
            ).to(device)
