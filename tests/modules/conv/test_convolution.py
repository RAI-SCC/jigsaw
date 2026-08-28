import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import jigsaw.modules.conv as conv
import jigsaw.modules.conv_transpose as conv_transpose
from jigsaw.utils_aggregate import gather_1d_results, gather_results_as_list
from tests.process_mesh import *

pytestmark = [pytest.mark.dist_required]


def _conv2d_reference_check(
    device: torch.device,
    mesh: dict,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    height: int,
    width: int,
    batch: int = 2,
    stride: int | None = None,
    padding: int = 0,
    dilation: int = 1,
) -> None:
    """Compare a distributed ``DistributedConv2D`` against a single-process conv2d.

    Works for any ``row.size()`` (spatial) x ``col.size()`` (channel) mesh and
    for both the no-halo regime (``stride == kernel_size``, the default) and the
    halo regime (any other stride). Rebuilds the equivalent non-distributed
    convolution from the sharded weight, bias and input, runs both forward and
    backward, and checks the output plus every gradient (input, weight, bias)
    after gathering the distributed shards.
    """
    row_group, col_group = mesh["row"], mesh["col"]
    n_spatial, n_channel = row_group.size(), col_group.size()
    model_group = mesh["model"]
    stride = kernel_size if stride is None else stride

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
        padding=(padding, padding),
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
    def test_halo_1x4(
        self, device: torch.device, process_mesh_dict_1x4: dict
    ) -> None:
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
    def test_halo_4x1(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
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
