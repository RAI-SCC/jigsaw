import pytest
import torch
import torch.distributed as dist

import jigsaw.linear as linear
from tests import conftest

MESHES = {"2x2": (2, 2), "4": (4,)}


@pytest.fixture()
def process_mesh():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 2], mesh_dim_names=["row", "col"]
    )


def test_sync_linear(device, process_mesh):
    """Verify weights for XWT_Sync are synchronized after a few iterations."""
    rank = dist.get_rank(group=None)
    torch.manual_seed(
        42 + (rank % 2)
    )  # for now, force the two "pairs" to have the same
    in_dim, out_dim = 5, 10
    linlayer = linear.DistributedXWTSync(
        in_dim,
        out_dim,
        True,
        rank,
        device,
        2,
        process_mesh["col"],
        process_mesh["row"],
        False,
    ).to(device)

    for _ in range(10):
        x_in = torch.randn(1, 20, 5, dtype=linlayer.weight.dtype, device=device)
        y = linlayer(x_in)
        target = (
            torch.ones(y.shape, dtype=x_in.dtype, device=device) * rank
        )  # different target per rank
        optimizer = torch.optim.Adam(linlayer.parameters())

        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()

        weight_grad = linlayer.weight.grad
        optimizer.step()

    weights_gathered_as_list = conftest.gather_results_as_list(
        linlayer.weight, process_mesh
    )
    weight_grad_gathered_as_list = conftest.gather_results_as_list(
        weight_grad, process_mesh
    )
    assert torch.allclose(weights_gathered_as_list[0], weights_gathered_as_list[2])
    assert torch.allclose(weights_gathered_as_list[1], weights_gathered_as_list[3])
    assert torch.allclose(
        weight_grad_gathered_as_list[0], weight_grad_gathered_as_list[2]
    )
    assert torch.allclose(
        weight_grad_gathered_as_list[1], weight_grad_gathered_as_list[3]
    )
