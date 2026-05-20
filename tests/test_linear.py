import torch
import torch.distributed as dist
from process_mesh import *

import jigsaw.linear as linear
from jigsaw.utils_aggregate import (
    gather_1d_results,
    gather_1d_results_list,
    gather_results,
)


def test_xwt_forward(device, process_mesh_2x2):
    """Test single forward pass of a linear layer."""
    local_rank = dist.get_rank(group=process_mesh_2x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    # local
    in_dim, out_dim = 2, 1
    seq_local = 20

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        False,
        local_rank,
        device,
        4,
        process_mesh_2x2["model"],
        process_mesh_2x2["row"],
        False,
    ).to(device)
    x_in = torch.randn(1, seq_local, in_dim, dtype=linlayer.weight.dtype, device=device)

    x_gathered = gather_results(x_in, process_mesh=process_mesh_2x2)
    w_gathered = gather_results(linlayer.weight.data, process_mesh=process_mesh_2x2)

    ref_linlayer = torch.nn.Linear(in_dim * 2, out_dim * 2, bias=False).to(device)
    ref_linlayer.weight.data = w_gathered
    print(f"w global {ref_linlayer.weight.shape}, w local {linlayer.weight.shape}")

    y = linlayer(x_in)
    y_ref = ref_linlayer(x_gathered)

    y_gathered = gather_results(y, process_mesh=process_mesh_2x2)
    assert torch.allclose(y_gathered, y_ref, atol=1e-5)


def test_xwt_forward_2(device, process_mesh_2x1x2):
    """Test single forward pass of a linear layer."""
    local_rank = dist.get_rank(group=process_mesh_2x1x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    # local
    in_dim, out_dim = 2, 2
    seq_local = 5

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        False,
        local_rank,
        device,
        process_mesh_2x1x2["model"].size(),
        process_mesh_2x1x2["model"],
        process_mesh_2x1x2["row"],
        False,
    ).to(device)

    x_in = torch.randn(1, seq_local, in_dim, dtype=linlayer.weight.dtype, device=device)

    x_gathered = gather_results(x_in, process_mesh=process_mesh_2x1x2)
    w_gathered = gather_results(linlayer.weight.data, process_mesh=process_mesh_2x1x2)

    ref_linlayer = torch.nn.Linear(in_dim * 2, out_dim, bias=False).to(device)
    ref_linlayer.weight.data = w_gathered

    y = linlayer(x_in)
    y_ref = ref_linlayer(x_gathered)

    y_gathered = gather_results(y, process_mesh=process_mesh_2x1x2)
    assert torch.allclose(y_gathered, y_ref, atol=1e-5)


def test_xwt_backward(device, process_mesh_2x2):
    """Test single forward and backward pass of a linear layer."""
    local_rank = dist.get_rank(group=process_mesh_2x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    in_dim, out_dim = 3, 2
    seq_local = 4

    loss_fn = torch.nn.MSELoss(reduction="mean")

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        False,
        local_rank,
        device,
        4,
        process_mesh_2x2["model"],
        process_mesh_2x2["row"],
        False,
    ).to(device)
    w_gathered = gather_results(linlayer.weight.data, process_mesh=process_mesh_2x2)

    ref_linlayer = torch.nn.Linear(in_dim * 2, out_dim * 2, bias=False).to(device)
    ref_linlayer.weight.data = w_gathered
    optimizer = torch.optim.Adam(linlayer.parameters())
    ref_optimizer = torch.optim.Adam(ref_linlayer.parameters())

    for _ in range(20):
        optimizer.zero_grad()
        ref_optimizer.zero_grad()
        x_in = torch.randn(
            1, seq_local, in_dim, dtype=linlayer.weight.dtype, device=device
        )
        x_in.requires_grad_(True)
        y = linlayer(x_in)
        target = (
            torch.ones(y.shape, dtype=x_in.dtype, device=device) * local_rank
        )  # different target per rank

        # gather x and target
        x_gathered = gather_results(x_in.detach(), process_mesh=process_mesh_2x2)
        x_gathered.requires_grad_(True)
        ref_target = gather_results(target, process_mesh=process_mesh_2x2)
        ref_y = ref_linlayer(x_gathered)
        y_gathered = gather_results(y, process_mesh=process_mesh_2x2)
        loss = loss_fn(y, target) / process_mesh_2x2["model"].size()
        loss_ref = loss_fn(ref_y, ref_target)
        loss.backward()
        loss_ref.backward()

        optimizer.step()
        ref_optimizer.step()

    w_gathered = gather_results(linlayer.weight, process_mesh=process_mesh_2x2)
    w_grad_gathered = gather_results(
        linlayer.weight.grad, process_mesh=process_mesh_2x2
    )
    x_grad_gathered = gather_results(x_in.grad, process_mesh=process_mesh_2x2)

    assert torch.allclose(y_gathered, ref_y, atol=1e-5)
    assert torch.allclose(w_gathered, ref_linlayer.weight, atol=1e-5)
    assert torch.allclose(w_grad_gathered, ref_linlayer.weight.grad, atol=1e-5)
    assert torch.allclose(x_grad_gathered, x_gathered.grad, atol=1e-5)


def test_xwt_backward_sum(device, process_mesh_2x2):
    """Test forward and backward pass of a few iterations."""
    local_rank = dist.get_rank(group=process_mesh_2x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    in_dim, out_dim = 3, 2
    seq_local = 4

    loss_fn = torch.nn.MSELoss(reduction="sum")

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        False,
        local_rank,
        device,
        4,
        process_mesh_2x2["model"],
        process_mesh_2x2["row"],
        False,
    ).to(device)
    w_gathered = gather_results(linlayer.weight.data, process_mesh=process_mesh_2x2)

    ref_linlayer = torch.nn.Linear(in_dim * 2, out_dim * 2, bias=False).to(device)
    ref_linlayer.weight.data = w_gathered
    optimizer = torch.optim.Adam(linlayer.parameters())
    ref_optimizer = torch.optim.Adam(ref_linlayer.parameters())

    for _ in range(20):
        x_in = torch.randn(
            1, seq_local, in_dim, dtype=linlayer.weight.dtype, device=device
        )
        x_in.requires_grad_(True)
        y = linlayer(x_in)
        target = (
            torch.ones(y.shape, dtype=x_in.dtype, device=device) * local_rank
        )  # different target per rank

        # gather x and target
        x_gathered = gather_results(x_in.detach(), process_mesh=process_mesh_2x2)
        x_gathered.requires_grad_(True)
        ref_target = gather_results(target, process_mesh=process_mesh_2x2)
        ref_y = ref_linlayer(x_gathered)
        y_gathered = gather_results(y, process_mesh=process_mesh_2x2)
        loss = loss_fn(y, target)
        loss_ref = loss_fn(ref_y, ref_target)
        loss.backward()
        loss_ref.backward()

        optimizer.step()
        ref_optimizer.step()

    w_grad_gathered = gather_results(
        linlayer.weight.grad, process_mesh=process_mesh_2x2
    )
    x_grad_gathered = gather_results(x_in.grad, process_mesh=process_mesh_2x2)

    assert torch.allclose(y_gathered, ref_y, atol=1e-5)
    assert torch.allclose(w_grad_gathered, ref_linlayer.weight.grad, atol=1e-5)
    assert torch.allclose(x_grad_gathered, x_gathered.grad, atol=1e-5)


def test_xwt_bias_init(device, process_mesh_2x2):
    """Ensure bias terms are synchronized across row groups."""
    local_rank = dist.get_rank(group=process_mesh_2x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    in_dim, out_dim = 5, 10

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        True,
        local_rank,
        device,
        4,
        process_mesh_2x2["model"],
        process_mesh_2x2["row"],
        False,
    ).to(device)

    b_gathered = gather_1d_results_list(
        linlayer.bias.data, process_mesh=process_mesh_2x2
    )
    assert torch.allclose(b_gathered[0], b_gathered[2], atol=1e-5)
    assert torch.allclose(b_gathered[1], b_gathered[3], atol=1e-5)
    assert not torch.allclose(
        b_gathered[0], b_gathered[1], atol=1e-5
    )  # ranks that are part of the same model group should have different biases.


def test_xwt_forward_bias(device, process_mesh_2x2):
    """Test forward pass including a bias."""
    local_rank = dist.get_rank(group=process_mesh_2x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    in_dim, out_dim = 5, 10
    seq_local = 20

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        True,
        local_rank,
        device,
        4,
        process_mesh_2x2["model"],
        process_mesh_2x2["row"],
        False,
    ).to(device)
    x_in = torch.randn(1, seq_local, in_dim, dtype=linlayer.weight.dtype, device=device)
    x_gathered = gather_results(x_in, process_mesh=process_mesh_2x2)
    w_gathered = gather_results(linlayer.weight.data, process_mesh=process_mesh_2x2)
    b_gathered = gather_1d_results(
        linlayer.bias.data, process_mesh=process_mesh_2x2, pm_key="col"
    )

    ref_linlayer = torch.nn.Linear(in_dim * 2, out_dim * 2, bias=True).to(device)
    ref_linlayer.weight.data = w_gathered
    ref_linlayer.bias.data = b_gathered

    y = linlayer(x_in)
    y_ref = ref_linlayer(x_gathered)

    y_gathered = gather_results(y, process_mesh=process_mesh_2x2)
    assert torch.allclose(y_gathered, y_ref, atol=1e-5)


def test_xwt_backward_bias(device, process_mesh_2x2):
    """Test single forward and backward pass of a linear layer including bias."""
    local_rank = dist.get_rank(group=process_mesh_2x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    in_dim, out_dim = 3, 2
    seq_local = 4

    loss_fn = torch.nn.MSELoss(reduction="mean")

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        True,
        local_rank,
        device,
        4,
        process_mesh_2x2["model"],
        process_mesh_2x2["row"],
        False,
    ).to(device)
    w_gathered = gather_results(linlayer.weight.data, process_mesh=process_mesh_2x2)
    b_gathered = gather_1d_results(
        linlayer.bias.data, process_mesh=process_mesh_2x2, pm_key="col"
    )

    ref_linlayer = torch.nn.Linear(in_dim * 2, out_dim * 2, bias=True).to(device)
    ref_linlayer.weight.data = w_gathered
    ref_linlayer.bias.data = b_gathered
    optimizer = torch.optim.Adam(linlayer.parameters())
    ref_optimizer = torch.optim.Adam(ref_linlayer.parameters())

    for _ in range(20):
        optimizer.zero_grad()
        ref_optimizer.zero_grad()
        x_in = torch.randn(
            1, seq_local, in_dim, dtype=linlayer.weight.dtype, device=device
        )
        x_in.requires_grad_(True)
        y = linlayer(x_in)
        target = (
            torch.ones(y.shape, dtype=x_in.dtype, device=device) * local_rank
        )  # different target per rank

        # gather x and target
        x_gathered = gather_results(x_in.detach(), process_mesh=process_mesh_2x2)
        x_gathered.requires_grad_(True)
        ref_target = gather_results(target, process_mesh=process_mesh_2x2)
        ref_y = ref_linlayer(x_gathered)
        y_gathered = gather_results(y, process_mesh=process_mesh_2x2)
        loss = loss_fn(y, target) / process_mesh_2x2["model"].size()
        loss_ref = loss_fn(ref_y, ref_target)
        loss.backward()
        loss_ref.backward()

        optimizer.step()
        ref_optimizer.step()

    w_gathered = gather_results(linlayer.weight, process_mesh=process_mesh_2x2)
    w_grad_gathered = gather_results(
        linlayer.weight.grad, process_mesh=process_mesh_2x2
    )
    b_gathered = gather_1d_results(
        linlayer.bias, process_mesh=process_mesh_2x2, pm_key="col"
    )
    b_grad_gathered = gather_1d_results(
        linlayer.bias.grad, process_mesh=process_mesh_2x2, pm_key="col"
    )
    x_grad_gathered = gather_results(x_in.grad, process_mesh=process_mesh_2x2)

    assert torch.allclose(y_gathered, ref_y, atol=1e-5)
    assert torch.allclose(w_gathered, ref_linlayer.weight, atol=1e-5)
    assert torch.allclose(b_gathered, ref_linlayer.bias, atol=1e-5)
    assert torch.allclose(w_grad_gathered, ref_linlayer.weight.grad, atol=1e-5)
    assert torch.allclose(b_grad_gathered, ref_linlayer.bias.grad, atol=1e-5)
    assert torch.allclose(x_grad_gathered, x_gathered.grad, atol=1e-5)
