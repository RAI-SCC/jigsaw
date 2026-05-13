import pytest
import torch
import torch.distributed as dist
import random

from tests import conftest
from jigsaw import linear


@pytest.fixture()
def process_mesh_2x2():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 2], mesh_dim_names=["row", "col"]
    )


@pytest.fixture()
def process_mesh_4x1():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[4, 1], mesh_dim_names=["row", "col"]
    )


@pytest.fixture()
def process_mesh_1x4():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[1, 4], mesh_dim_names=["row", "col"]
    )


@pytest.fixture()
def process_mesh_2x1x2():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 1, 2], mesh_dim_names=["dp", "row", "col"]
    )


@pytest.fixture()
def process_mesh_2x2x1():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 2, 1], mesh_dim_names=["dp", "row", "col"]
    )


@pytest.mark.parametrize(
    "from_pm,to_pm",
    [
        (from_pm, to_pm)
        for from_pm in [
            "process_mesh_2x2",
            "process_mesh_2x1x2",
            "process_mesh_1x4",
            "process_mesh_4x1",
            "process_mesh_2x2x1",
        ]
        for to_pm in [
            "process_mesh_2x2",
            "process_mesh_2x1x2",
            "process_mesh_1x4",
            "process_mesh_4x1",
            "process_mesh_2x2x1",
        ]
    ],
)
def test_weight_merging(request, from_pm, to_pm, device):
    # initialize a linear layer
    random.seed(0)
    local_in_dim, local_out_dim = random.randint(1, 100)*4, random.randint(1, 100)*4  # local
    from_pm = request.getfixturevalue(from_pm)
    to_pm = request.getfixturevalue(to_pm)
    torch.manual_seed(from_pm["model"].rank())

    linear_layer = linear.DistributedXWT(
        in_dim=local_in_dim,
        out_dim=local_out_dim,
        bias=True,
        local_rank=from_pm["model"].rank(),
        device=device,
        parallelism=from_pm["model"].size(),
        model_group=from_pm["model"],
        spatial_group=from_pm["col"],
        bayesian=False,
    )
    local_weights_before, local_bias_before = linear_layer.weight, linear_layer.bias
    gathered_weights_before = conftest.gather_results(local_weights_before, from_pm)
    gathered_bias_before = conftest.gather_1d_results(
        local_bias_before, from_pm, dim=0, pm_key="col"
    )
    linear_layer.reshard_parameters(from_pm, to_pm)
    local_weights_after = linear_layer.weight
    local_bias_after = linear_layer.bias
    gathered_weights_after = conftest.gather_results(local_weights_after, to_pm)
    gathered_bias_after = conftest.gather_1d_results(
        local_bias_after, to_pm, dim=0, pm_key="col"
    )
    assert torch.allclose(gathered_weights_before, gathered_weights_after)
    assert linear_layer.parallelism == to_pm["model"].size()
    assert torch.allclose(gathered_bias_before, gathered_bias_after)

# def test_weight_merging(process_mesh_2x2, process_mesh_2x2):
#     # initialize a linear layer
#     local_in_dim, local_out_dim = random.randint(1, 100), random.randint(1, 100)  # local
        
#     torch.manual_seed(0)

#     linear_layer = linear.DistributedXWT(
#         in_dim=local_in_dim,
#         out_dim=local_out_dim,
#         bias=True,
#         local_rank=from_pm["model"].rank(),
#         device=device,
#         parallelism=from_pm["model"].size(),
#         model_group=from_pm["model"],
#         spatial_group=from_pm["col"],
#         bayesian=False,
#     )
#     local_weights_before, local_bias_before = linear_layer.weight, linear_layer.bias

#     gathered_weights_before = conftest.gather_results(local_weights_before, from_pm)
#     gathered_bias_before = conftest.gather_1d_results(
#         local_bias_before, from_pm, dim=0, pm_key="col"
#     )
#     linear_layer.reshard_parameters(from_pm, to_pm)
#     local_weights_after = linear_layer.weight
#     local_bias_after = linear_layer.bias
#     gathered_weights_after = conftest.gather_results(local_weights_after, to_pm)
#     gathered_bias_after = conftest.gather_1d_results(
#         local_bias_after, to_pm, dim=0, pm_key="col"
#     )
#     assert torch.allclose(gathered_weights_before, gathered_weights_after)
#     assert linear_layer.parallelism == to_pm["model"].size()
#     assert torch.allclose(gathered_bias_before, gathered_bias_after)
