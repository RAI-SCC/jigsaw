import random

import pytest
import torch

from jigsaw import convolutions, linear, mlp
from jigsaw.utils_aggregate import gather_1d_results, gather_results
from tests.process_mesh import *


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
    """Test weight merging for linear layer."""
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
        spatial_group=from_pm["row"],
        bayesian=False,
    )
    local_weights_before, local_bias_before = linear_layer.weight, linear_layer.bias
    gathered_weights_before = gather_results(local_weights_before, from_pm)
    gathered_bias_before = gather_1d_results(
        local_bias_before, from_pm, dim=0, pm_key="col"
    )
    linear_layer.reshard_parameters(from_pm, to_pm)
    local_weights_after = linear_layer.weight
    local_bias_after = linear_layer.bias
    gathered_weights_after = gather_results(local_weights_after, to_pm)
    gathered_bias_after = gather_1d_results(
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
    """Test weight merging for MLP."""
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
    layer1_weights_global = gather_results(layer1_weights, from_pm)
    layer2_weights_global = gather_results(layer2_weights, from_pm)
    layer1_bias_global = gather_1d_results(layer1_bias, from_pm, dim=0, pm_key="col")
    layer2_bias_global = gather_1d_results(layer2_bias, from_pm, dim=0, pm_key="col")

    for module in mlp_layer.modules():
        if hasattr(module, "reshard_parameters"):
            module.reshard_parameters(from_pm, process_mesh_4x1x1)
        else:
            pass

    layer1_weights_gathered = mlp_layer.linear1.weight
    layer2_weights_gathered = mlp_layer.linear2.weight
    layer1_bias_gathered = mlp_layer.linear1.bias
    layer2_bias_gathered = mlp_layer.linear2.bias

    assert torch.allclose(layer1_weights_global, layer1_weights_gathered), (
        "Layer 1 gathered weights are not the same as the original"
    )
    assert torch.allclose(layer2_weights_global, layer2_weights_gathered), (
        "Layer 2 gathered weights are not the same as the original"
    )
    assert torch.allclose(layer1_bias_global, layer1_bias_gathered), (
        "Layer 1 gathered bias is not the same as the original"
    )
    assert torch.allclose(layer2_bias_global, layer2_bias_gathered), (
        "Layer 2 gathered bias is not the same as the original"
    )


@pytest.mark.parametrize(
    "from_pm,to_pm",
    [
        (from_pm, to_pm)
        for from_pm in [
            "process_mesh_2x2",
            "process_mesh_4x1",
            "process_mesh_4x1x1",
            "process_mesh_2x1x2",
            "process_mesh_2x2x1",
        ]
        for to_pm in [
            "process_mesh_2x2",
            "process_mesh_4x1",
            "process_mesh_4x1x1",
            "process_mesh_2x1x2",
            "process_mesh_2x2x1",
        ]
    ],
)
def test_conv_weight_merging(request, from_pm, to_pm, device):
    """Test weight merging of Conv2d."""
    random.seed(0)
    in_channels, out_channels, kernel_size = (
        random.randint(1, 100) * 4,
        random.randint(1, 100) * 4,
        random.randint(1, 5),
    )  # local
    from_pm = request.getfixturevalue(from_pm)
    to_pm = request.getfixturevalue(to_pm)
    torch.manual_seed(from_pm["col"].rank())

    conv_layer = convolutions.DistributedConv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        local_rank=from_pm["model"].rank(),
        spatial_group=from_pm["row"],
        channel_group=from_pm["col"],
        device=device,
        parallelism=from_pm["col"].size(),
    )
    local_weights_before, local_bias_before = conv_layer.weight, conv_layer.bias
    gathered_weights_before = gather_1d_results(
        local_weights_before, from_pm, dim=0, pm_key="col"
    )
    gathered_bias_before = gather_1d_results(
        local_bias_before, from_pm, dim=0, pm_key="col"
    )
    conv_layer.reshard_parameters(from_pm, to_pm)
    local_weights_after = conv_layer.weight
    local_bias_after = conv_layer.bias
    gathered_weights_after = gather_1d_results(
        local_weights_after, to_pm, dim=0, pm_key="col"
    )
    gathered_bias_after = gather_1d_results(
        local_bias_after, to_pm, dim=0, pm_key="col"
    )
    assert torch.allclose(gathered_weights_before, gathered_weights_after)
    assert conv_layer.parallelism == to_pm["col"].size()
    assert torch.allclose(gathered_bias_before, gathered_bias_after)


@pytest.mark.parametrize(
    "from_pm,to_pm",
    [
        (from_pm, to_pm)
        for from_pm in [
            "process_mesh_2x2",
            "process_mesh_4x1",
            "process_mesh_4x1x1",
            "process_mesh_2x1x2",
            "process_mesh_2x2x1",
        ]
        for to_pm in [
            "process_mesh_2x2",
            "process_mesh_4x1",
            "process_mesh_4x1x1",
            "process_mesh_2x1x2",
            "process_mesh_2x2x1",
        ]
    ],
)
def test_conv_transpose_weight_merging(request, from_pm, to_pm, device):
    """Test weight merging for ConvTranspose2d."""
    random.seed(0)
    in_channels, out_channels, kernel_size = (
        random.randint(1, 100) * 4,
        random.randint(1, 100) * 4,
        random.randint(1, 5),
    )  # local
    from_pm = request.getfixturevalue(from_pm)
    to_pm = request.getfixturevalue(to_pm)
    torch.manual_seed(from_pm["col"].rank())

    conv_layer = convolutions.DistributedConvTranspose2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        local_rank=from_pm["model"].rank(),
        spatial_group=from_pm["row"],
        channel_group=from_pm["col"],
        device=device,
        parallelism=from_pm["col"].size(),
    )
    local_weights_before, local_bias_before = conv_layer.weight, conv_layer.bias
    gathered_weights_before = gather_1d_results(
        local_weights_before, from_pm, dim=1, pm_key="col"
    )
    gathered_bias_before = gather_1d_results(
        local_bias_before, from_pm, dim=0, pm_key="col"
    )
    conv_layer.reshard_parameters(from_pm, to_pm)

    local_weights_after = conv_layer.weight
    local_bias_after = conv_layer.bias
    gathered_weights_after = gather_1d_results(
        local_weights_after, to_pm, dim=1, pm_key="col"
    )
    gathered_bias_after = gather_1d_results(
        local_bias_after, to_pm, dim=0, pm_key="col"
    )

    assert torch.allclose(gathered_weights_before, gathered_weights_after)
    assert conv_layer.parallelism == to_pm["col"].size()
    assert torch.allclose(gathered_bias_before, gathered_bias_after)
