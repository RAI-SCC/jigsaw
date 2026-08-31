"""Distributed Conv3dTranspose with the domain split along several spatial axes.

Split out of ``test_conv3dtranspose.py``: the single-spatial-axis (width-only)
distributed transpose conv lives there, everything that splits depth / height --
or any combination, optionally with a channel split -- lives here.
"""

import pytest
import torch
import torch.nn.functional as F

import jigsaw.modules.conv3dtranspose as conv3dtranspose
from jigsaw.utils_aggregate import gather_1d_results
from tests.process_mesh import *  # noqa: F401,F403

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

    conv = conv3dtranspose.DistributedConv3DTranspose(
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
        channel_comm=channel_comm,
    ).to(device)

    def _gather_spatial(t: torch.Tensor, channel_dim: int) -> torch.Tensor:
        t = gather_1d_results(t, mesh, dim=channel_dim, pm_key="col")
        t = gather_1d_results(t, mesh, dim=-3, pm_key="sd")
        t = gather_1d_results(t, mesh, dim=-2, pm_key="sh")
        return gather_1d_results(t, mesh, dim=-1, pm_key="sw")

    weight_global = gather_1d_results(
        conv.weight.detach(), mesh, dim=1, pm_key="col"
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
    w_grad_global = gather_1d_results(conv.weight.grad, mesh, dim=1, pm_key="col")
    b_grad_global = gather_1d_results(conv.bias.grad, mesh, dim=0, pm_key="col")

    assert torch.allclose(y_global, y_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(x_grad_global, x_global.grad, atol=1e-4, rtol=1e-4)
    # The parameter grads reduce over every output position; across a big volume
    # the distributed partial sums land in a different float32 order than the
    # single-process reference, so allow ~1e-3 there.
    assert torch.allclose(w_grad_global, weight_global.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(b_grad_global, bias_global.grad, atol=1e-3, rtol=1e-3)


class TestConv3dTransposeMultiSpatial:
    """Distributed Conv3dTranspose with the domain split along several spatial axes."""

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
