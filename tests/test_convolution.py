import pytest
import torch
import torch.distributed as dist

import jigsaw.convolutions as conv
from jigsaw.utils_aggregate import gather_results_as_list

from .process_mesh import *

pytestmark = [pytest.mark.distributed_required]


def test_param_init_conv2d(device: torch.device, process_mesh_dict_2x2: dict) -> None:
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
        stride=kernel_size,
        padding=(0,),
        dilation=1,
        groups=1,
        bias=True,
        local_rank=local_rank,  # is this model-local or whole convolutional-local?
        spatial_group=process_mesh_dict_2x2["row"],
        channel_group=process_mesh_dict_2x2["col"],
        device=device,
        parallelism=4,
        bayesian=False,
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


def test_param_init_conv2d_transpose(
    device: torch.device, process_mesh_dict_2x2: dict
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
    conv2d = conv.DistributedConvTranspose2D(
        in_dim,
        out_dim,
        kernel_size=(kernel_size,),
        stride=kernel_size,
        padding=(0, 0),
        dilation=(1,),
        groups=1,
        bias=True,
        local_rank=local_rank,  # is this model-local or whole convolutional-local?
        spatial_group=process_mesh_dict_2x2["row"],
        channel_group=process_mesh_dict_2x2["col"],
        device=device,
        parallelism=4,
        bayesian=False,
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


def test_conv2d_forward(device: torch.device, process_mesh_dict_2x2: dict) -> None:
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
    conv2d = conv.DistributedConv2D(
        in_dim,
        out_dim,
        kernel_size,
        stride=kernel_size,
        padding=(0,),
        dilation=1,
        groups=1,
        bias=True,
        local_rank=local_rank,  # is this model-local or whole convolutional-local?
        spatial_group=process_mesh_dict_2x2["row"],
        channel_group=process_mesh_dict_2x2["col"],
        device=device,
        parallelism=4,
        bayesian=False
    )

    weights_gathered_as_list = gather_results_as_list(
        conv2d.weight, process_mesh_dict_2x2
    )

    for i in range(process_mesh_dict_2x2["model"].size() // row_comm_size):
        assert torch.allclose(
            weights_gathered_as_list[i],
            weights_gathered_as_list[i + col_comm_size],
            atol=1e-5,
        )
    
    x = torch.randn(4, in_dim // 2, h, w // 2)
    y = conv2d(x)
