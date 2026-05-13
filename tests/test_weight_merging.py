import pytest
import torch
import torch.distributed as dist
import random

from tests import conftest
from jigsaw import linear, mlp


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
def process_mesh_4x1x1():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[4, 1, 1], mesh_dim_names=["dp", "row", "col"]
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
            "process_mesh_4x1x1",
        ]
    ],
)
def test_weight_merging(request, from_pm, to_pm, device):
    # initialize a linear layer
    random.seed(0)
    local_in_dim, local_out_dim = (
        random.randint(1, 100) * 4,
        random.randint(1, 100) * 4,
    )  # local
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


@pytest.mark.parametrize(
    "from_pm",
    [
        (from_pm)
        for from_pm in [
            "process_mesh_2x2",
            "process_mesh_2x1x2",
            "process_mesh_1x4",
            "process_mesh_4x1",
            "process_mesh_2x2x1",
        ]
    ],
)
def test_mlp_weight_merging(request, from_pm, process_mesh_4x1x1, device):
    from_pm = request.getfixturevalue(from_pm)

    mlp_layer = mlp.DistributedMLP(
        8,
        12,
        12,
        True,
        0.0,
        from_pm["model"].rank(),
        device,
        from_pm["model"].size(),
        from_pm["model"],
        from_pm["row"],
    )

    layer1_weights = mlp_layer.linear1.weight
    layer2_weights = mlp_layer.linear2.weight
    layer1_bias = mlp_layer.linear1.bias
    layer2_bias = mlp_layer.linear2.bias

    # gather linear 1 weights
    layer1_weights_global = conftest.gather_results(layer1_weights, from_pm)
    layer2_weights_global = conftest.gather_results(layer2_weights, from_pm)
    layer1_bias_global = conftest.gather_1d_results(
        layer1_bias, from_pm, dim=0, pm_key="col"
    )
    layer2_bias_global = conftest.gather_1d_results(
        layer2_bias, from_pm, dim=0, pm_key="col"
    )

    for module in mlp_layer.modules():
        if hasattr(module, "reshard_parameters"):
            module.reshard_parameters(from_pm, process_mesh_4x1x1)
        else:
            pass

    layer1_weights_gathered = mlp_layer.linear1.weight
    layer2_weights_gathered = mlp_layer.linear2.weight
    layer1_bias_gathered = mlp_layer.linear1.bias
    layer2_bias_gathered = mlp_layer.linear2.bias

    assert torch.allclose(
        layer1_weights_global, layer1_weights_gathered
    ), "Layer 1 gathered weights are not the same as the original"
    assert torch.allclose(
        layer2_weights_global, layer2_weights_gathered
    ), "Layer 2 gathered weights are not the same as the original"
    assert torch.allclose(
        layer1_bias_global, layer1_bias_gathered
    ), "Layer 1 gathered bias is not the same as the original"
    assert torch.allclose(
        layer2_bias_global, layer2_bias_gathered
    ), "Layer 2 gathered bias is not the same as the original"
