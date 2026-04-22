import pytest
import torch
import torch.distributed as dist

import jigsaw.linear as linear
from tests import conftest
from tests.conftest import make_model_group

MESHES = {"2x2": (2, 2), "4": (4,)}


@pytest.fixture()
def model_group(request):
    """Return model group in process mesh."""
    return make_model_group(MESHES["2x2"])


@pytest.fixture()
def sync_group(request):
    """Return model group in process mesh."""
    return make_model_group(MESHES["4"])


def test_sync_linear(device, model_group):
    process_mesh = conftest.make_process_mesh()

    rank = dist.get_rank(group=None)
    torch.manual_seed(42 + (rank % 2))  # for now, force the two to have the same
    in_dim, out_dim = 5, 10
    print(f"model group size {model_group.size()}")
    linlayer = linear.DistributedXWTSync(
        in_dim,
        out_dim,
        True,
        rank,
        device,
        2,
        process_mesh.get_group("col"),
        process_mesh.get_group("row"),
        False,
    ).to(device)

    for _ in range(10):
        x_in = torch.randn(1, 20, 5, dtype=linlayer.weight.dtype, device=device)
        x_in.requires_grad_(True)
        x_in = x_in.contiguous()
        y = linlayer(x_in)
        target = (
            torch.ones(y.shape, dtype=x_in.dtype, device=device) * rank
        )  # different target per rank
        optimizer = torch.optim.Adam(linlayer.parameters())

        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()

        weight_grad = linlayer.weight.grad
        optimizer.step()

    weights_gathered_as_list = conftest.gather_results_fourway_list(
        linlayer.weight, group=None
    )
    weight_grad_gathered_as_list = conftest.gather_results_fourway_list(
        weight_grad, group=None
    )
    assert torch.allclose(weights_gathered_as_list[0], weights_gathered_as_list[2])
    assert torch.allclose(weights_gathered_as_list[1], weights_gathered_as_list[3])
    assert torch.allclose(
        weight_grad_gathered_as_list[0], weight_grad_gathered_as_list[2]
    )
    assert torch.allclose(
        weight_grad_gathered_as_list[1], weight_grad_gathered_as_list[3]
    )
