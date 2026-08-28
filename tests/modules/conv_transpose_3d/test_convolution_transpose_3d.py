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
    the halo regime. Rebuilds the equivalent non-distributed transpose convolution
    from the sharded weight, bias and input, runs forward and backward on both,
    and checks the output plus every gradient after gathering the shards.
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

    torch.manual_seed(2024)
    grad_global = torch.randn(y_ref.shape, device=device)
    y_ref.backward(grad_global)

    # --- distributed forward / backward on this rank's shard --------------
    x_dist = x_local.clone().requires_grad_(True)
    y = conv(x_dist)

    out_local = out_channels // n_channel
    assert y_ref.shape[-1] % n_spatial == 0, (
        f"global output width {y_ref.shape[-1]} not divisible by spatial "
        f"size {n_spatial}"
    )
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

    def test_forward_backward(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """No-halo regime (stride == kernel) matches a single-process op."""
        _conv3d_transpose_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=2,
            depth=6,
            height=6,
            width=16,
        )

    def test_stride_lt_kernel_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Overlapping raw outputs (stride < kernel_size) match a plain op."""
        _conv3d_transpose_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            depth=6,
            height=6,
            width=16,
            stride=2,
            padding=1,
        )

    def test_asymmetric_kernel_stride_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Non-square kernel and stride, stride != kernel on the split axis."""
        _conv3d_transpose_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=(2, 3, 4),
            depth=6,
            height=6,
            width=16,
            stride=(2, 2, 2),
            padding=(0, 1, 1),
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
        x = torch.randn(2, 4, 6, 6, 8, device=device)
        with pytest.raises(NotImplementedError):
            conv(x)


class TestConv3dTransposeParallel:
    """DistributedConvTranspose3d across arbitrary spatial and channel sizes."""

    @pytest.mark.dist(4)
    def test_forward_backward_1x4(
        self, device: torch.device, process_mesh_dict_1x4: dict
    ) -> None:
        """Pure channel parallelism (4 channel ranks, no spatial split)."""
        _conv3d_transpose_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=8,
            out_channels=12,
            kernel_size=2,
            depth=6,
            height=6,
            width=16,
        )

    @pytest.mark.dist(4)
    def test_halo_4x1(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        """Halo regime with 4-way spatial split (real cross-rank exchange)."""
        _conv3d_transpose_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            depth=6,
            height=6,
            width=32,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(8)
    def test_halo_2x4(
        self, device: torch.device, process_mesh_dict_2x4_rowcol: dict
    ) -> None:
        """Halo regime combined with 4-way channel parallelism."""
        _conv3d_transpose_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            depth=6,
            height=6,
            width=16,
            stride=2,
            padding=1,
        )
