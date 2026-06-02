import torch
import torch.distributed as dist

import jigsaw.linear as linear
from jigsaw.utils_aggregate import gather_results_as_list


def test_sync_xwt(device, process_mesh_2x2):
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
        process_mesh_2x2["col"],
        process_mesh_2x2["row"],
        False,
    ).to(device)
    optimizer = torch.optim.Adam(linlayer.parameters())

    for _ in range(10):
        x_in = torch.randn(1, 20, 5, dtype=linlayer.weight.dtype, device=device)
        y = linlayer(x_in)
        target = (
            torch.ones(y.shape, dtype=x_in.dtype, device=device) * local_rank
        )  # different target per rank

        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()

        optimizer.step()

    weights_gathered_as_list = gather_results_as_list(linlayer.weight, process_mesh_2x2)
    weight_grad_gathered_as_list = gather_results_as_list(
        linlayer.weight.grad, process_mesh_2x2
    )
    assert torch.allclose(weights_gathered_as_list[0], weights_gathered_as_list[2])
    assert torch.allclose(weights_gathered_as_list[1], weights_gathered_as_list[3])
    assert torch.allclose(
        weight_grad_gathered_as_list[0], weight_grad_gathered_as_list[2]
    )
    assert torch.allclose(
        weight_grad_gathered_as_list[1], weight_grad_gathered_as_list[3]
    )


def test_param_init_xwt(device, process_mesh_2x2):
    """Verify initial weights for XWT_Sync are correctly synchronized."""
    rank = dist.get_rank(group=None)
    torch.manual_seed(rank)  # each process gets a different initial seed
    row_comm_size, col_comm_size = (
        process_mesh_2x2["row"].size(),
        process_mesh_2x2["col"].size(),
    )
    in_dim, out_dim = 5, 10
    linlayer = linear.DistributedXWTSync(
        in_dim,
        out_dim,
        True,
        rank,
        device,
        2,
        process_mesh_2x2["col"],
        process_mesh_2x2["row"],
        False,
    ).to(device)

    weights_gathered_as_list = gather_results_as_list(linlayer.weight, process_mesh_2x2)

    for i in range(process_mesh_2x2["model"].size() // row_comm_size):
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
        process_mesh_2x2["col"],
        process_mesh_2x2["row"],
        False,
    ).to(device)

    weights_gathered_as_list = gather_results_as_list(linlayer.weight, process_mesh_2x2)

    for i in range(process_mesh_2x2["model"].size() // row_comm_size):
        assert torch.allclose(
            weights_gathered_as_list[i], weights_gathered_as_list[i + col_comm_size]
        )


# parameter counts accurate
