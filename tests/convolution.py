import pytest
import torch
import torch.distributed as dist

import jigsaw.convolutions as conv
from tests import conftest


@pytest.fixture()
def process_mesh():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 2], mesh_dim_names=["row", "col"]
    )


def test_param_init_conv2d(device, process_mesh):
    """Verify initial weights for Conv2d are correctly synchronized."""
    local_rank = dist.get_rank(group=process_mesh["model"])
    torch.manual_seed(local_rank)  # each process gets a different initial seed
    row_comm_size, col_comm_size = (
        process_mesh["row"].size(),
        process_mesh["col"].size(),
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
        spatial_group=process_mesh["row"],
        channel_group=process_mesh["col"],
        device=device,
        parallelism=4,
        bayesian=False,
    ).to(device)

    weights_gathered_as_list = conftest.gather_results_as_list(
        conv2d.weight, process_mesh
    )

    for i in range(process_mesh["model"].size() // row_comm_size):
        assert torch.allclose(
            weights_gathered_as_list[i],
            weights_gathered_as_list[i + col_comm_size],
            atol=1e-5,
        )


def test_param_init_conv2d_transpose(device, process_mesh):
    """Verify initial weights for Conv2d are correctly synchronized."""
    local_rank = dist.get_rank(group=process_mesh["model"])
    torch.manual_seed(local_rank)  # each process gets a different initial seed
    row_comm_size, col_comm_size = (
        process_mesh["row"].size(),
        process_mesh["col"].size(),
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
        spatial_group=process_mesh["row"],
        channel_group=process_mesh["col"],
        device=device,
        parallelism=4,
        bayesian=False,
    ).to(device)

    weights_gathered_as_list = conftest.gather_results_as_list(
        conv2d.weight, process_mesh
    )

    for i in range(process_mesh["model"].size() // row_comm_size):
        assert torch.allclose(
            weights_gathered_as_list[i],
            weights_gathered_as_list[i + col_comm_size],
            atol=1e-5,
        )
