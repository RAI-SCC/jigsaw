import random

import pytest
import torch

from jigsaw import convolutions, linear
from jigsaw.utils_aggregate import gather_1d_results, gather_results


pytestmark = [pytest.mark.gpu_required, pytest.mark.distributed_required]


@pytest.mark.parametrize(
    "from_pm_dict,to_pm_dict",
    [
        (from_pm_dict, to_pm_dict)
        for from_pm_dict in [
            "process_mesh_dict_2x2",
            "process_mesh_dict_2x1x2",
            "process_mesh_dict_1x4",
            "process_mesh_dict_4x1",
            "process_mesh_dict_2x2x1",
        ]
        for to_pm_dict in [
            "process_mesh_dict_2x2",
            "process_mesh_dict_2x1x2",
            "process_mesh_dict_1x4",
            "process_mesh_dict_4x1",
            "process_mesh_dict_2x2x1",
            "process_mesh_dict_4x1x1",
        ]
    ],
)
def test_weight_merging(
    request: pytest.FixtureRequest,
    from_pm_dict: dict,
    to_pm_dict: dict,
    device: torch.device,
) -> None:
    """Test weight merging for linear layer."""
    random.seed(0)
    local_in_dim, local_out_dim = (
        random.randint(1, 100) * 4,
        random.randint(1, 100) * 4,
    )  # local
    from_pm_dict = request.getfixturevalue(from_pm_dict)
    to_pm_dict = request.getfixturevalue(to_pm_dict)
    torch.manual_seed(from_pm_dict["model"].rank())

    linear_layer = linear.DistributedXWT(
        in_dim=local_in_dim,
        out_dim=local_out_dim,
        bias=True,
        local_rank=from_pm_dict["model"].rank(),
        device=device,
        model_group=from_pm_dict["model"],
        spatial_group=from_pm_dict["row"],
        channel_group=from_pm_dict["col"],
        bayesian=False,
    )
    assert isinstance(linear_layer.weight, torch.Tensor)
    assert isinstance(linear_layer.bias, torch.Tensor)
    local_weights_before, local_bias_before = (
        linear_layer.weight.data,
        linear_layer.bias.data,
    )
    gathered_weights_before = gather_results(local_weights_before, from_pm_dict)
    gathered_bias_before = gather_1d_results(
        local_bias_before, from_pm_dict, dim=0, pm_key="col"
    )
    linear_layer.reshard_parameters(from_pm_dict, to_pm_dict)
    local_weights_after = linear_layer.weight
    local_bias_after = linear_layer.bias
    gathered_weights_after = gather_results(local_weights_after, to_pm_dict)
    gathered_bias_after = gather_1d_results(
        local_bias_after, to_pm_dict, dim=0, pm_key="col"
    )
    assert torch.allclose(gathered_weights_before, gathered_weights_after)
    assert linear_layer.parallelism == to_pm_dict["model"].size()
    assert torch.allclose(gathered_bias_before, gathered_bias_after)


@pytest.mark.parametrize(
    "from_pm_dict",
    [
        (from_pm_dict)
        for (from_pm_dict) in [
            "process_mesh_dict_2x2",
            "process_mesh_dict_2x1x2",
            "process_mesh_dict_1x4",
            "process_mesh_dict_4x1",
            "process_mesh_dict_2x2x1",
        ]
    ],
)
def test_mlp_weight_merging(
    request: pytest.FixtureRequest,
    from_pm_dict: dict,
    process_mesh_dict_4x1x1: dict,
    device: torch.device,
) -> None:
    """Test weight merging for MLP."""
    from_pm_dict = request.getfixturevalue(from_pm_dict)

    in_dim, hidden_dim, out_dim = 12, 12, 12

    class MLP(torch.nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
            super().__init__()
            self.linear1 = linear.DistributedXWT(
                in_dim,
                hidden_dim,
                bias=True,
                local_rank=from_pm_dict["model"].rank(),
                device=device,
                model_group=from_pm_dict["model"],
                spatial_group=from_pm_dict["row"],
                channel_group=from_pm_dict["col"],
            )

            self.linear2 = linear.DistributedXWT(
                hidden_dim,
                out_dim,
                bias=True,
                local_rank=from_pm_dict["model"].rank(),
                device=device,
                model_group=from_pm_dict["model"],
                spatial_group=from_pm_dict["row"],
                channel_group=from_pm_dict["col"],
            )
            self.activation = torch.nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x

    mlp = MLP(in_dim, hidden_dim, out_dim)

    assert isinstance(mlp.linear1.weight, torch.Tensor)
    assert isinstance(mlp.linear2.weight, torch.Tensor)
    assert isinstance(mlp.linear1.bias, torch.Tensor)
    assert isinstance(mlp.linear2.bias, torch.Tensor)

    layer1_weights = mlp.linear1.weight.data
    layer2_weights = mlp.linear2.weight.data
    layer1_bias = mlp.linear1.bias.data
    layer2_bias = mlp.linear2.bias.data

    # gather linear 1 weights
    layer1_weights_global = gather_results(layer1_weights, from_pm_dict)
    layer2_weights_global = gather_results(layer2_weights, from_pm_dict)
    layer1_bias_global = gather_1d_results(
        layer1_bias, from_pm_dict, dim=0, pm_key="col"
    )
    layer2_bias_global = gather_1d_results(
        layer2_bias, from_pm_dict, dim=0, pm_key="col"
    )

    for module in mlp.modules():
        if hasattr(module, "reshard_parameters"):
            module.reshard_parameters(from_pm_dict, process_mesh_dict_4x1x1)
        else:
            pass

    layer1_weights_gathered = mlp.linear1.weight
    layer2_weights_gathered = mlp.linear2.weight
    layer1_bias_gathered = mlp.linear1.bias
    layer2_bias_gathered = mlp.linear2.bias

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
    "from_pm_dict,to_pm_dict",
    [
        (from_pm_dict, to_pm_dict)
        for from_pm_dict in [
            "process_mesh_dict_2x2",
            "process_mesh_dict_4x1",
            "process_mesh_dict_4x1x1",
            "process_mesh_dict_2x1x2",
            "process_mesh_dict_2x2x1",
        ]
        for to_pm_dict in [
            "process_mesh_dict_2x2",
            "process_mesh_dict_4x1",
            "process_mesh_dict_4x1x1",
            "process_mesh_dict_2x1x2",
            "process_mesh_dict_2x2x1",
        ]
    ],
)
def test_conv_weight_merging(
    request: pytest.FixtureRequest,
    from_pm_dict: dict,
    to_pm_dict: dict,
    device: torch.device,
) -> None:
    """Test weight merging of Conv2d."""
    random.seed(0)
    in_channels, out_channels, kernel_size = (
        random.randint(1, 100) * 4,
        random.randint(1, 100) * 4,
        random.randint(1, 5),
    )  # local
    from_pm_dict = request.getfixturevalue(from_pm_dict)
    to_pm_dict = request.getfixturevalue(to_pm_dict)
    torch.manual_seed(from_pm_dict["col"].rank())

    conv_layer = convolutions.DistributedConv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        local_rank=from_pm_dict["model"].rank(),
        spatial_group=from_pm_dict["row"],
        channel_group=from_pm_dict["col"],
        device=device,
        parallelism=from_pm_dict["col"].size(),
    )
    assert isinstance(conv_layer.weight, torch.Tensor)
    assert isinstance(conv_layer.bias, torch.Tensor)
    local_weights_before, local_bias_before = (
        conv_layer.weight.data,
        conv_layer.bias.data,
    )
    gathered_weights_before = gather_1d_results(
        local_weights_before, from_pm_dict, dim=0, pm_key="col"
    )
    gathered_bias_before = gather_1d_results(
        local_bias_before, from_pm_dict, dim=0, pm_key="col"
    )
    conv_layer.reshard_parameters(from_pm_dict, to_pm_dict)
    local_weights_after = conv_layer.weight
    local_bias_after = conv_layer.bias
    gathered_weights_after = gather_1d_results(
        local_weights_after, to_pm_dict, dim=0, pm_key="col"
    )
    gathered_bias_after = gather_1d_results(
        local_bias_after, to_pm_dict, dim=0, pm_key="col"
    )
    assert torch.allclose(gathered_weights_before, gathered_weights_after)
    assert conv_layer.parallelism == to_pm_dict["col"].size()
    assert torch.allclose(gathered_bias_before, gathered_bias_after)


@pytest.mark.parametrize(
    "from_pm_dict,to_pm_dict",
    [
        (from_pm_dict, to_pm_dict)
        for from_pm_dict in [
            "process_mesh_dict_2x2",
            "process_mesh_dict_4x1",
            "process_mesh_dict_4x1x1",
            "process_mesh_dict_2x1x2",
            "process_mesh_dict_2x2x1",
        ]
        for to_pm_dict in [
            "process_mesh_dict_2x2",
            "process_mesh_dict_4x1",
            "process_mesh_dict_4x1x1",
            "process_mesh_dict_2x1x2",
            "process_mesh_dict_2x2x1",
        ]
    ],
)
def test_conv_transpose_weight_merging(
    request: pytest.FixtureRequest,
    from_pm_dict: dict,
    to_pm_dict: dict,
    device: torch.device,
) -> None:
    """Test weight merging for ConvTranspose2d."""
    random.seed(0)
    in_channels, out_channels, kernel_size = (
        random.randint(1, 100) * 4,
        random.randint(1, 100) * 4,
        random.randint(1, 5),
    )  # local
    from_pm_dict = request.getfixturevalue(from_pm_dict)
    to_pm_dict = request.getfixturevalue(to_pm_dict)
    torch.manual_seed(from_pm_dict["col"].rank())

    conv_layer = convolutions.DistributedConvTranspose2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        local_rank=from_pm_dict["model"].rank(),
        spatial_group=from_pm_dict["row"],
        channel_group=from_pm_dict["col"],
        device=device,
        parallelism=from_pm_dict["col"].size(),
    )
    assert isinstance(conv_layer.weight, torch.Tensor)
    assert isinstance(conv_layer.bias, torch.Tensor)
    local_weights_before, local_bias_before = (
        conv_layer.weight.data,
        conv_layer.bias.data,
    )

    gathered_weights_before = gather_1d_results(
        local_weights_before, from_pm_dict, dim=1, pm_key="col"
    )
    gathered_bias_before = gather_1d_results(
        local_bias_before, from_pm_dict, dim=0, pm_key="col"
    )
    conv_layer.reshard_parameters(from_pm_dict, to_pm_dict)

    local_weights_after = conv_layer.weight
    local_bias_after = conv_layer.bias
    assert isinstance(local_weights_after, torch.Tensor)
    assert isinstance(local_bias_after, torch.Tensor)
    gathered_weights_after = gather_1d_results(
        local_weights_after, to_pm_dict, dim=1, pm_key="col"
    )
    gathered_bias_after = gather_1d_results(
        local_bias_after, to_pm_dict, dim=0, pm_key="col"
    )

    assert torch.allclose(gathered_weights_before, gathered_weights_after)
    assert conv_layer.parallelism == to_pm_dict["col"].size()
    assert torch.allclose(gathered_bias_before, gathered_bias_after)
