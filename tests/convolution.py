import pytest
import torch
import torch.distributed as dist

from jigsaw.utils_aggregate import gather_results_as_list
import jigsaw.convolutions as conv
from tests import conftest
from tests.process_mesh import process_mesh_2x2


def test_param_init_conv2d(device, process_mesh_2x2):
    """Verify initial weights for Conv2d are correctly synchronized."""
    local_rank = dist.get_rank(group=process_mesh_2x2["model"])
    torch.manual_seed(local_rank)  # each process gets a different initial seed
    row_comm_size, col_comm_size = (
        process_mesh_2x2["row"].size(),
        process_mesh_2x2["col"].size(),
    )
    in_dim, out_dim = 5, 10
    kernel_size = 4
    conv2d = conv.DistributedConv2D(
        in_dim,
        out_dim,
        kernel_size,
        stride=kernel_size,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        local_rank=local_rank,  # is this model-local or whole convolutional-local?
        spatial_group=process_mesh_2x2["row"],
        channel_group=process_mesh_2x2["col"],
        device=device,
        parallelism=4,
        bayesian=False,
    ).to(device)

    weights_gathered_as_list = gather_results_as_list(conv2d.weight, process_mesh_2x2)

    for i in range(process_mesh_2x2["model"].size() // row_comm_size):
        assert torch.allclose(
            weights_gathered_as_list[i],
            weights_gathered_as_list[i + col_comm_size],
            atol=1e-5,
        )


def test_param_init_conv2d_transpose(device, process_mesh_2x2):
    """Verify initial weights for Conv2d are correctly synchronized."""
    local_rank = dist.get_rank(group=process_mesh_2x2["model"])
    torch.manual_seed(local_rank)  # each process gets a different initial seed
    row_comm_size, col_comm_size = (
        process_mesh_2x2["row"].size(),
        process_mesh_2x2["col"].size(),
    )
    in_dim, out_dim = 6, 10
    kernel_size = 4
    conv2d = conv.DistributedConvTranspose2D(
        in_dim,
        out_dim,
        kernel_size,
        stride=kernel_size,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        local_rank=local_rank,  # is this model-local or whole convolutional-local?
        spatial_group=process_mesh_2x2["row"],
        channel_group=process_mesh_2x2["col"],
        device=device,
        parallelism=4,
        bayesian=False,
    ).to(device)

    weights_gathered_as_list = gather_results_as_list(conv2d.weight, process_mesh_2x2)

    for i in range(process_mesh_2x2["model"].size() // row_comm_size):
        assert torch.allclose(
            weights_gathered_as_list[i],
            weights_gathered_as_list[i + col_comm_size],
            atol=1e-5,
        )
