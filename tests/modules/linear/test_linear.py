import pytest
import torch
import torch.distributed as dist

import jigsaw.hooks as hooks
import jigsaw.modules.linear as linear
from jigsaw.utils_aggregate import (
    gather_1d_results,
    gather_1d_results_list,
    gather_results,
)

pytestmark = [pytest.mark.distributed_required]


@pytest.mark.dist(4)
def test_xwt_forward(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Test single forward pass of a linear layer."""
    local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    # global dims
    in_dim, out_dim = 4, 4
    seq_global = 20
    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        False,
        local_rank,
        device,
        process_mesh_dict_2x2["model"],
        process_mesh_dict_2x2["col"],
        process_mesh_dict_2x2["row"],
        bayesian=False,
    ).to(device)

    x_in = torch.randn(
        1, seq_global // 2, in_dim // 2, dtype=linlayer.weight.dtype, device=device
    )

    x_gathered = gather_results(x_in, process_mesh_dict=process_mesh_dict_2x2)
    w_gathered = gather_results(
        linlayer.weight.data, process_mesh_dict=process_mesh_dict_2x2
    )

    ref_linlayer = torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
    ref_linlayer.weight.data = w_gathered

    y = linlayer(x_in)
    y_ref = ref_linlayer(x_gathered)

    y_gathered = gather_results(
        y,
        process_mesh_dict=process_mesh_dict_2x2,
        pm_key_row="row",
        pm_key_col="col",
        pm_key_model="model",
    )
    assert torch.allclose(y_gathered, y_ref, atol=1e-5)


@pytest.mark.dist(4)
def test_xwt_forward_2(device: torch.device, process_mesh_dict_2x1x2: dict) -> None:
    """Test single forward pass of a linear layer."""
    local_rank = dist.get_rank(group=process_mesh_dict_2x1x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    # global
    in_dim, out_dim = 4, 4
    seq_global = 10

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        bias=False,
        local_rank=local_rank,
        device=device,
        model_group=process_mesh_dict_2x1x2["model"],
        column_group=process_mesh_dict_2x1x2["col"],
        row_group=process_mesh_dict_2x1x2["row"],
        bayesian=False,
    ).to(device)

    x_in = torch.randn(
        1, seq_global, in_dim // 2, dtype=linlayer.weight.dtype, device=device
    )

    x_gathered = gather_results(x_in, process_mesh_dict=process_mesh_dict_2x1x2)
    w_gathered = gather_results(
        linlayer.weight.data, process_mesh_dict=process_mesh_dict_2x1x2
    )

    ref_linlayer = torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
    ref_linlayer.weight.data = w_gathered

    y = linlayer(x_in)
    y_ref = ref_linlayer(x_gathered)

    y_gathered = gather_results(y, process_mesh_dict=process_mesh_dict_2x1x2)
    assert torch.allclose(y_gathered, y_ref, atol=1e-5)


@pytest.mark.dist(4)
def test_xwt_backward(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Test single forward and backward pass of a linear layer."""
    local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    in_dim, out_dim = 6, 4
    seq_local = 4

    loss_fn = torch.nn.MSELoss(reduction="mean")

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        False,
        local_rank,
        device,
        process_mesh_dict_2x2["model"],
        process_mesh_dict_2x2["col"],
        process_mesh_dict_2x2["row"],
        False,
    ).to(device)
    w_gathered = gather_results(
        linlayer.weight.data, process_mesh_dict=process_mesh_dict_2x2
    )

    ref_linlayer = torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
    ref_linlayer.weight.data = w_gathered
    optimizer = torch.optim.Adam(linlayer.parameters())
    ref_optimizer = torch.optim.Adam(ref_linlayer.parameters())

    for _ in range(20):
        optimizer.zero_grad()
        ref_optimizer.zero_grad()
        x_in = torch.randn(
            1, seq_local // 2, in_dim // 2, dtype=linlayer.weight.dtype, device=device
        )
        x_in.requires_grad_(True)
        y = linlayer(x_in)
        target = (
            torch.ones(y.shape, dtype=x_in.dtype, device=device) * local_rank
        )  # different target per rank

        # gather x and target
        x_gathered = gather_results(
            x_in.detach(), process_mesh_dict=process_mesh_dict_2x2
        )
        x_gathered.requires_grad_(True)
        ref_target = gather_results(target, process_mesh_dict=process_mesh_dict_2x2)
        ref_y = ref_linlayer(x_gathered)
        y_gathered = gather_results(y, process_mesh_dict=process_mesh_dict_2x2)
        loss = loss_fn(y, target) / process_mesh_dict_2x2["model"].size()
        loss_ref = loss_fn(ref_y, ref_target)
        loss.backward()
        loss_ref.backward()

        optimizer.step()
        ref_optimizer.step()

    assert isinstance(linlayer.weight, torch.Tensor)
    assert isinstance(linlayer.weight.grad, torch.Tensor)
    assert isinstance(x_in.grad, torch.Tensor)
    w_gathered = gather_results(
        linlayer.weight.data, process_mesh_dict=process_mesh_dict_2x2
    )
    w_grad_gathered = gather_results(
        linlayer.weight.grad, process_mesh_dict=process_mesh_dict_2x2
    )
    x_grad_gathered = gather_results(x_in.grad, process_mesh_dict=process_mesh_dict_2x2)
    assert isinstance(ref_y, torch.Tensor)
    assert isinstance(ref_linlayer.weight, torch.Tensor)
    assert isinstance(ref_linlayer.weight.grad, torch.Tensor)
    assert isinstance(x_gathered.grad, torch.Tensor)

    assert torch.allclose(y_gathered, ref_y, atol=1e-5)
    assert torch.allclose(w_gathered, ref_linlayer.weight, atol=1e-5)
    assert torch.allclose(w_grad_gathered, ref_linlayer.weight.grad, atol=1e-5)
    assert torch.allclose(x_grad_gathered, x_gathered.grad, atol=1e-5)


@pytest.mark.dist(4)
def test_xwt_backward_sum(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Test forward and backward pass of a few iterations."""
    local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    in_dim, out_dim = 4, 2
    seq_local = 4

    loss_fn = torch.nn.MSELoss(reduction="sum")

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        False,
        local_rank,
        device,
        process_mesh_dict_2x2["model"],
        process_mesh_dict_2x2["col"],
        process_mesh_dict_2x2["row"],
        False,
    ).to(device)
    w_gathered = gather_results(
        linlayer.weight.data, process_mesh_dict=process_mesh_dict_2x2
    )

    ref_linlayer = torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
    ref_linlayer.weight.data = w_gathered
    optimizer = torch.optim.Adam(linlayer.parameters())
    ref_optimizer = torch.optim.Adam(ref_linlayer.parameters())

    for _ in range(20):
        x_in = torch.randn(
            1, seq_local // 2, in_dim // 2, dtype=linlayer.weight.dtype, device=device
        )
        x_in.requires_grad_(True)
        y = linlayer(x_in)
        target = (
            torch.ones(y.shape, dtype=x_in.dtype, device=device) * local_rank
        )  # different target per rank

        # gather x and target
        x_gathered = gather_results(
            x_in.detach(), process_mesh_dict=process_mesh_dict_2x2
        )
        x_gathered.requires_grad_(True)
        ref_target = gather_results(target, process_mesh_dict=process_mesh_dict_2x2)
        ref_y = ref_linlayer(x_gathered)
        y_gathered = gather_results(y, process_mesh_dict=process_mesh_dict_2x2)
        loss = loss_fn(y, target)
        loss_ref = loss_fn(ref_y, ref_target)
        loss.backward()
        loss_ref.backward()

        optimizer.step()
        ref_optimizer.step()

    assert isinstance(linlayer.weight.grad, torch.Tensor)
    assert isinstance(x_in.grad, torch.Tensor)

    w_grad_gathered = gather_results(
        linlayer.weight.grad, process_mesh_dict=process_mesh_dict_2x2
    )
    x_grad_gathered = gather_results(x_in.grad, process_mesh_dict=process_mesh_dict_2x2)

    assert isinstance(ref_y, torch.Tensor)
    assert isinstance(ref_linlayer.weight.grad, torch.Tensor)
    assert isinstance(x_gathered.grad, torch.Tensor)
    assert torch.allclose(y_gathered, ref_y, atol=1e-5)
    assert torch.allclose(w_grad_gathered, ref_linlayer.weight.grad, atol=1e-5)
    assert torch.allclose(x_grad_gathered, x_gathered.grad, atol=1e-5)


@pytest.mark.dist(4)
def test_xwt_bias_init(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Ensure bias terms are synchronized across row groups."""
    local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    in_dim, out_dim = 6, 10

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        True,
        local_rank,
        device,
        process_mesh_dict_2x2["model"],
        process_mesh_dict_2x2["col"],
        process_mesh_dict_2x2["row"],
        False,
    ).to(device)

    assert isinstance(linlayer.bias, torch.Tensor)

    b_gathered = gather_1d_results_list(
        linlayer.bias.data, process_mesh_dict=process_mesh_dict_2x2
    )
    assert torch.allclose(b_gathered[0], b_gathered[2], atol=1e-5)
    assert torch.allclose(b_gathered[1], b_gathered[3], atol=1e-5)
    assert not torch.allclose(
        b_gathered[0], b_gathered[1], atol=1e-5
    )  # ranks that are part of the same model group should have different biases.


@pytest.mark.dist(4)
def test_xwt_forward_bias(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Test forward pass including a bias."""
    local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    in_dim, out_dim = 6, 10
    seq_global = 20

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        True,
        local_rank,
        device,
        process_mesh_dict_2x2["model"],
        process_mesh_dict_2x2["col"],
        process_mesh_dict_2x2["row"],
        False,
    ).to(device)
    x_in = torch.randn(
        1, seq_global // 2, in_dim // 2, dtype=linlayer.weight.dtype, device=device
    )

    assert isinstance(linlayer.weight, torch.Tensor)
    assert isinstance(linlayer.bias, torch.Tensor)
    x_gathered = gather_results(x_in, process_mesh_dict=process_mesh_dict_2x2)
    w_gathered = gather_results(
        linlayer.weight.data, process_mesh_dict=process_mesh_dict_2x2
    )
    b_gathered = gather_1d_results(
        linlayer.bias.data, process_mesh_dict=process_mesh_dict_2x2, pm_key="col"
    )

    ref_linlayer = torch.nn.Linear(in_dim, out_dim, bias=True).to(device)
    ref_linlayer.weight.data = w_gathered
    ref_linlayer.bias.data = b_gathered

    y = linlayer(x_in)
    y_ref = ref_linlayer(x_gathered)

    assert isinstance(y_ref, torch.Tensor)

    y_gathered = gather_results(y, process_mesh_dict=process_mesh_dict_2x2)
    assert torch.allclose(y_gathered, y_ref, atol=1e-5)


@pytest.mark.dist(4)
def test_xwt_backward_bias(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Test single forward and backward pass of a linear layer including bias."""
    local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    in_dim, out_dim = 6, 2
    seq_global = 4

    loss_fn = torch.nn.MSELoss(reduction="mean")

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        True,
        local_rank,
        device,
        process_mesh_dict_2x2["model"],
        process_mesh_dict_2x2["col"],
        process_mesh_dict_2x2["row"],
        False,
    ).to(device)

    assert isinstance(linlayer.weight, torch.Tensor)
    assert isinstance(linlayer.bias, torch.Tensor)
    w_gathered = gather_results(
        linlayer.weight.data, process_mesh_dict=process_mesh_dict_2x2
    )
    b_gathered = gather_1d_results(
        linlayer.bias.data, process_mesh_dict=process_mesh_dict_2x2, pm_key="col"
    )

    ref_linlayer = torch.nn.Linear(in_dim, out_dim, bias=True).to(device)
    ref_linlayer.weight.data = w_gathered
    ref_linlayer.bias.data = b_gathered
    optimizer = torch.optim.Adam(linlayer.parameters())
    ref_optimizer = torch.optim.Adam(ref_linlayer.parameters())

    for _ in range(20):
        optimizer.zero_grad()
        ref_optimizer.zero_grad()
        x_in = torch.randn(
            1, seq_global // 2, in_dim // 2, dtype=linlayer.weight.dtype, device=device
        )
        x_in.requires_grad_(True)
        y = linlayer(x_in)
        target = (
            torch.ones(y.shape, dtype=x_in.dtype, device=device) * local_rank
        )  # different target per rank

        # gather x and target
        x_gathered = gather_results(
            x_in.detach(), process_mesh_dict=process_mesh_dict_2x2
        )
        x_gathered.requires_grad_(True)
        ref_target = gather_results(target, process_mesh_dict=process_mesh_dict_2x2)
        ref_y = ref_linlayer(x_gathered)
        y_gathered = gather_results(y, process_mesh_dict=process_mesh_dict_2x2)
        loss = loss_fn(y, target) / process_mesh_dict_2x2["model"].size()
        loss_ref = loss_fn(ref_y, ref_target)
        loss.backward()
        loss_ref.backward()

        optimizer.step()
        ref_optimizer.step()

    assert isinstance(linlayer.weight, torch.Tensor)
    assert isinstance(linlayer.weight.grad, torch.Tensor)
    assert isinstance(linlayer.bias, torch.Tensor)
    assert isinstance(linlayer.bias.grad, torch.Tensor)
    assert isinstance(x_in.grad, torch.Tensor)

    w_gathered = gather_results(
        linlayer.weight.data, process_mesh_dict=process_mesh_dict_2x2
    )
    w_grad_gathered = gather_results(
        linlayer.weight.grad, process_mesh_dict=process_mesh_dict_2x2
    )
    b_gathered = gather_1d_results(
        linlayer.bias.data, process_mesh_dict=process_mesh_dict_2x2, pm_key="col"
    )
    b_grad_gathered = gather_1d_results(
        linlayer.bias.grad, process_mesh_dict=process_mesh_dict_2x2, pm_key="col"
    )
    x_grad_gathered = gather_results(x_in.grad, process_mesh_dict=process_mesh_dict_2x2)

    assert isinstance(ref_y, torch.Tensor)
    assert isinstance(ref_linlayer.weight, torch.Tensor)
    assert isinstance(ref_linlayer.bias, torch.Tensor)
    assert isinstance(ref_linlayer.weight.grad, torch.Tensor)
    assert isinstance(ref_linlayer.bias.grad, torch.Tensor)
    assert isinstance(x_gathered.grad, torch.Tensor)

    assert torch.allclose(y_gathered, ref_y, atol=1e-5)
    assert torch.allclose(w_gathered, ref_linlayer.weight, atol=1e-5)
    assert torch.allclose(b_gathered, ref_linlayer.bias, atol=1e-5)
    assert torch.allclose(w_grad_gathered, ref_linlayer.weight.grad, atol=1e-5)
    assert torch.allclose(b_grad_gathered, ref_linlayer.bias.grad, atol=1e-5)
    assert torch.allclose(x_grad_gathered, x_gathered.grad, atol=1e-5)


@pytest.mark.dist(4)
def test_xwt_backward_2(device: torch.device, process_mesh_dict_2x1x2: dict) -> None:
    """Test forward--backward pass of a 2-way sharded model."""
    local_rank = dist.get_rank(group=process_mesh_dict_2x1x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    in_dim, out_dim = 4, 4
    seq_global = 10

    loss_fn = torch.nn.MSELoss(reduction="mean")

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        True,
        local_rank,
        device,
        process_mesh_dict_2x1x2["model"],
        process_mesh_dict_2x1x2["col"],
        process_mesh_dict_2x1x2["row"],
        False,
    ).to(device)

    w_gathered = gather_results(
        linlayer.weight.data, process_mesh_dict=process_mesh_dict_2x1x2
    )
    if linlayer.bias is not None:
        b_gathered = gather_1d_results(
            linlayer.bias.data, process_mesh_dict=process_mesh_dict_2x1x2, pm_key="col"
        )
    ref_linlayer = torch.nn.Linear(in_dim, out_dim, bias=True).to(device)
    ref_linlayer.weight.data = w_gathered
    if linlayer.bias is not None:
        ref_linlayer.bias.data = b_gathered
    optimizer = torch.optim.Adam(linlayer.parameters())
    ref_optimizer = torch.optim.Adam(ref_linlayer.parameters())

    for _ in range(20):
        optimizer.zero_grad()
        ref_optimizer.zero_grad()
        x_in = torch.randn(
            1, seq_global, in_dim // 2, dtype=linlayer.weight.dtype, device=device
        )
        x_in.requires_grad_(True)
        y = linlayer(x_in)
        target = (
            torch.ones(y.shape, dtype=x_in.dtype, device=device) * local_rank
        )  # different target per rank

        # gather x and target
        x_gathered = gather_results(
            x_in.detach(), process_mesh_dict=process_mesh_dict_2x1x2
        )
        x_gathered.requires_grad_(True)
        ref_target = gather_results(target, process_mesh_dict=process_mesh_dict_2x1x2)
        ref_y = ref_linlayer(x_gathered)
        y_gathered = gather_results(y, process_mesh_dict=process_mesh_dict_2x1x2)
        loss = loss_fn(y, target) / process_mesh_dict_2x1x2["model"].size()
        loss_ref = loss_fn(ref_y, ref_target)
        loss.backward()
        loss_ref.backward()

        optimizer.step()
        ref_optimizer.step()

    assert isinstance(linlayer.weight, torch.Tensor)
    assert isinstance(linlayer.weight.grad, torch.Tensor)
    assert isinstance(x_in.grad, torch.Tensor)

    w_gathered = gather_results(
        linlayer.weight.data, process_mesh_dict=process_mesh_dict_2x1x2
    )
    w_grad_gathered = gather_results(
        linlayer.weight.grad, process_mesh_dict=process_mesh_dict_2x1x2
    )
    if linlayer.bias is not None:
        assert isinstance(linlayer.bias, torch.Tensor)
        assert isinstance(linlayer.bias.grad, torch.Tensor)
        b_gathered = gather_1d_results(
            linlayer.bias.data, process_mesh_dict=process_mesh_dict_2x1x2, pm_key="col"
        )
        b_grad_gathered = gather_1d_results(
            linlayer.bias.grad, process_mesh_dict=process_mesh_dict_2x1x2, pm_key="col"
        )
    x_grad_gathered = gather_results(
        x_in.grad, process_mesh_dict=process_mesh_dict_2x1x2
    )

    assert isinstance(ref_y, torch.Tensor)
    assert isinstance(ref_linlayer.weight, torch.Tensor)
    assert isinstance(ref_linlayer.weight.grad, torch.Tensor)

    assert torch.allclose(y_gathered, ref_y, atol=1e-5)
    assert torch.allclose(w_gathered, ref_linlayer.weight, atol=1e-5)
    assert torch.allclose(w_grad_gathered, ref_linlayer.weight.grad, atol=1e-5)

    if linlayer.bias is not None:
        assert isinstance(ref_linlayer.bias, torch.Tensor)
        assert isinstance(ref_linlayer.bias.grad, torch.Tensor)

        assert torch.allclose(b_gathered, ref_linlayer.bias, atol=1e-5)
        assert torch.allclose(b_grad_gathered, ref_linlayer.bias.grad, atol=1e-5)
    assert torch.allclose(x_grad_gathered, x_gathered.grad, atol=1e-5)


@pytest.mark.dist(4)
def test_xwt_backward_dtp(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Test forward-backward pass of a model that employs domain parallel in one axis and channel parallel in the other."""
    local_rank = dist.get_rank(group=process_mesh_dict_2x2["col"])
    torch.manual_seed(process_mesh_dict_2x2["model"].rank())

    in_dim, out_dim = 10, 10
    seq_global = 10

    loss_fn = torch.nn.MSELoss(reduction="mean")

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        True,
        local_rank,
        device,
        process_mesh_dict_2x2["model"],
        process_mesh_dict_2x2["col"],
        process_mesh_dict_2x2["row"],
        False,
    ).to(device)

    # broadcasting
    for p in linlayer.parameters():
        dist.broadcast(p.data, group_src=0, group=process_mesh_dict_2x2["row"])

    linlayer.weight.register_post_accumulate_grad_hook(
        hooks.make_grad_sync_hook(process_mesh_dict_2x2["row"])
    )

    w_gathered = gather_1d_results(
        linlayer.weight.data,
        process_mesh_dict=process_mesh_dict_2x2,
        dim=-1,
        pm_key="col",
    )
    if linlayer.bias is not None:
        b_gathered = gather_1d_results(
            linlayer.bias.data, process_mesh_dict=process_mesh_dict_2x2, pm_key="col"
        )

    ref_linlayer = torch.nn.Linear(in_dim, out_dim, bias=True).to(device)
    ref_linlayer.weight.data = w_gathered
    if linlayer.bias is not None:
        ref_linlayer.bias.data = b_gathered
    optimizer = torch.optim.Adam(linlayer.parameters())
    ref_optimizer = torch.optim.Adam(ref_linlayer.parameters())

    for _ in range(20):
        optimizer.zero_grad()
        ref_optimizer.zero_grad()
        x_in = (
            torch.randn(
                1, seq_global, in_dim // 2, dtype=linlayer.weight.dtype, device=device
            )
            / 30
        )
        x_in.requires_grad_(True)
        y = linlayer(x_in)
        target = (
            torch.ones(y.shape, dtype=x_in.dtype, device=device) * local_rank
        )  # different target per rank

        # gather x and target
        x_gathered = gather_1d_results(
            x_in.detach(), process_mesh_dict=process_mesh_dict_2x2, dim=-1, pm_key="col"
        )
        x_gathered = gather_1d_results(
            x_gathered.detach(),
            process_mesh_dict=process_mesh_dict_2x2,
            dim=0,
            pm_key="row",
        )
        x_gathered.requires_grad_(True)
        ref_target = gather_1d_results(
            target, process_mesh_dict=process_mesh_dict_2x2, dim=-1, pm_key="col"
        )
        ref_target = gather_1d_results(
            ref_target, process_mesh_dict=process_mesh_dict_2x2, dim=0, pm_key="row"
        )
        ref_y = ref_linlayer(x_gathered)
        y_gathered = gather_1d_results(
            y, process_mesh_dict=process_mesh_dict_2x2, dim=-1, pm_key="col"
        )
        y_gathered = gather_1d_results(
            y_gathered, process_mesh_dict=process_mesh_dict_2x2, dim=0, pm_key="row"
        )
        loss = loss_fn(y, target) / process_mesh_dict_2x2["model"].size()
        loss_ref = loss_fn(ref_y, ref_target)
        loss.backward()
        loss_ref.backward()

        optimizer.step()
        ref_optimizer.step()

    assert isinstance(linlayer.weight, torch.Tensor)
    assert isinstance(linlayer.weight.grad, torch.Tensor)
    assert isinstance(x_in.grad, torch.Tensor)
    w_gathered = gather_1d_results(
        linlayer.weight.data,
        process_mesh_dict=process_mesh_dict_2x2,
        dim=-1,
        pm_key="col",
    )
    w_grad_gathered = linlayer.weight.grad.clone().unsqueeze(0)
    w_grad_gathered = gather_1d_results(
        w_grad_gathered, process_mesh_dict=process_mesh_dict_2x2, dim=-1, pm_key="col"
    )

    x_grad_gathered = gather_1d_results(
        x_in.grad, process_mesh_dict=process_mesh_dict_2x2, dim=-1, pm_key="col"
    )
    x_grad_gathered = gather_1d_results(
        x_grad_gathered, process_mesh_dict=process_mesh_dict_2x2, dim=0, pm_key="row"
    )

    assert isinstance(ref_y, torch.Tensor)
    assert isinstance(ref_linlayer.weight, torch.Tensor)
    assert isinstance(ref_linlayer.weight.grad, torch.Tensor)
    assert isinstance(x_gathered, torch.Tensor)
    assert isinstance(w_grad_gathered, torch.Tensor)
    assert isinstance(x_gathered.grad, torch.Tensor)

    assert torch.allclose(y_gathered, ref_y, atol=1e-4)
    assert torch.allclose(w_gathered, ref_linlayer.weight, atol=1e-4)
    assert torch.allclose(w_grad_gathered, ref_linlayer.weight.grad, atol=1e-4)
    assert torch.allclose(x_grad_gathered, x_gathered.grad, atol=1e-4)

    if linlayer.bias is not None:
        assert isinstance(linlayer.bias, torch.Tensor)
        assert isinstance(linlayer.bias.grad, torch.Tensor)

        b_gathered = gather_1d_results(
            linlayer.bias.data,
            process_mesh_dict=process_mesh_dict_2x2,
            pm_key="col",
            dim=-1,
        )

        # bias

        b_grad_gathered = linlayer.bias.grad.clone().unsqueeze(dim=0)
        assert isinstance(b_grad_gathered, torch.Tensor)
        assert isinstance(ref_linlayer.bias, torch.Tensor)
        assert isinstance(ref_linlayer.bias.grad, torch.Tensor)
        b_grad_gathered = gather_1d_results(
            b_grad_gathered,
            process_mesh_dict=process_mesh_dict_2x2,
            pm_key="col",
            dim=-1,
        )
        b_grad_gathered = b_grad_gathered
        assert torch.allclose(b_gathered, ref_linlayer.bias, atol=1e-3)
        assert torch.allclose(b_grad_gathered, ref_linlayer.bias.grad, atol=1e-3)


@pytest.mark.dist(8)
def test_xwt_backward_sum_8(device: torch.device, process_mesh_dict_4x2: dict) -> None:
    """Test forward and backward pass of a few iterations."""
    local_rank = dist.get_rank(group=process_mesh_dict_4x2["model"])
    torch.manual_seed(local_rank)  # for now, force the two "pairs" to have the same

    in_dim, out_dim = 8, 4
    seq_local = 4

    loss_fn = torch.nn.MSELoss(reduction="sum")

    linlayer = linear.DistributedXWT(
        in_dim,
        out_dim,
        False,
        local_rank,
        device,
        process_mesh_dict_4x2["model"],
        process_mesh_dict_4x2["col"],
        process_mesh_dict_4x2["row"],
        False,
    )
    w_gathered = gather_results(
        linlayer.weight.data, process_mesh_dict=process_mesh_dict_4x2
    )
    ref_linlayer = torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
    ref_linlayer.weight.data = w_gathered
    optimizer = torch.optim.Adam(linlayer.parameters())
    ref_optimizer = torch.optim.Adam(ref_linlayer.parameters())

    for _ in range(20):
        x_in = torch.randn(
            1, seq_local // 4, in_dim // 2, dtype=linlayer.weight.dtype, device=device
        )
        x_in.requires_grad_(True)
        y = linlayer(x_in)
        target = (
            torch.ones(y.shape, dtype=x_in.dtype, device=device) * local_rank
        )  # different target per rank

        # gather x and target
        x_gathered = gather_results(
            x_in.detach(), process_mesh_dict=process_mesh_dict_4x2
        )
        x_gathered.requires_grad_(True)
        ref_target = gather_results(target, process_mesh_dict=process_mesh_dict_4x2)
        ref_y = ref_linlayer(x_gathered)
        y_gathered = gather_results(y, process_mesh_dict=process_mesh_dict_4x2)
        loss = loss_fn(y, target)
        loss_ref = loss_fn(ref_y, ref_target)
        loss.backward()
        loss_ref.backward()

        optimizer.step()
        ref_optimizer.step()

    assert isinstance(linlayer.weight.grad, torch.Tensor)
    assert isinstance(x_in.grad, torch.Tensor)

    w_grad_gathered = gather_results(
        linlayer.weight.grad, process_mesh_dict=process_mesh_dict_4x2
    )
    x_grad_gathered = gather_results(x_in.grad, process_mesh_dict=process_mesh_dict_4x2)

    assert isinstance(ref_y, torch.Tensor)
    assert isinstance(ref_linlayer.weight.grad, torch.Tensor)
    assert isinstance(x_gathered.grad, torch.Tensor)
    assert torch.allclose(y_gathered, ref_y, atol=1e-5)
    assert torch.allclose(w_grad_gathered, ref_linlayer.weight.grad, atol=1e-5)
    assert torch.allclose(x_grad_gathered, x_gathered.grad, atol=1e-5)
