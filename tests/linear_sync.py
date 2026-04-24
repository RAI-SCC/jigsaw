import pytest
import torch
import torch.distributed as dist

import jigsaw.linear as linear
from tests import conftest


@pytest.fixture()
def process_mesh():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 2], mesh_dim_names=["row", "col"]
    )


def test_sync_xwt(device, process_mesh):
    """Verify weights for XWT_Sync are synchronized after a few iterations."""
    local_rank = dist.get_rank(group=None)
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same
    in_dim, out_dim = 5, 10
    linlayer = linear.DistributedXWTSync(
        in_dim,
        out_dim,
        True,
        local_rank,
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
            torch.ones(y.shape, dtype=x_in.dtype, device=device) * local_rank
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


def test_param_init_xwt(device, process_mesh):
    """Verify initial weights for XWT_Sync are correctly synchronized."""
    rank = dist.get_rank(group=None)
    torch.manual_seed(rank)  # each process gets a different initial seed
    row_comm_size, col_comm_size = (
        process_mesh["row"].size(),
        process_mesh["col"].size(),
    )
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

    weights_gathered_as_list = conftest.gather_results_as_list(
        linlayer.weight, process_mesh
    )

    for i in range(process_mesh["model"].size() // row_comm_size):
        assert torch.allclose(
            weights_gathered_as_list[i], weights_gathered_as_list[i + col_comm_size]
        )

    # same, with no bias
    in_dim, out_dim = 5, 10
    linlayer = linear.DistributedXWTSync(
        in_dim,
        out_dim,
        False,
        rank,
        device,
        2,
        process_mesh["col"],
        process_mesh["row"],
        False,
    ).to(device)

    weights_gathered_as_list = conftest.gather_results_as_list(
        linlayer.weight, process_mesh
    )

    for i in range(process_mesh["model"].size() // row_comm_size):
        assert torch.allclose(
            weights_gathered_as_list[i], weights_gathered_as_list[i + col_comm_size]
        )


# parameter counts accurate

# forward pass of linear layer accurate
# backward pass of linear layer accurate?
