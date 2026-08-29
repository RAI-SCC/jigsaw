"""Parity tests for the distributed loss functions.

Every test builds the *global* tensors identically on all ranks (seeded), slices
out this rank's shard, runs the distributed loss on the shard and compares
against the plain ``torch.nn.functional`` loss evaluated on the global tensor.

Layout used throughout: logits of shape ``[N, C, S]`` where the class dimension
``C`` is sharded across the ``col`` group and the spatial dimension ``S`` across
the ``row`` group.  ``N`` is never sharded.  Hence ``channel_group == col`` and
``reduce_groups == (row,)`` for the classification losses; for BCE (no class
dimension) both groups shard the tensor and both appear in ``reduce_groups``.
"""

import pytest
import torch
import torch.nn.functional as F

import jigsaw.modules.loss as jloss
from jigsaw.modules.loss import functional_classification as lossc
from jigsaw.modules.loss import functional_regression as lossr
from jigsaw.utils_aggregate import gather_1d_results
from tests.process_mesh import *  # noqa: F403

pytestmark = [pytest.mark.dist_required]

MESHES_4 = [
    "process_mesh_dict_2x2",
    "process_mesh_dict_1x4",
    "process_mesh_dict_4x1",
    "process_mesh_dict_2x1x2",
    "process_mesh_dict_2x2x1",
]

FTOL = dict(atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _shard(t: torch.Tensor, pmd: dict, dims: dict) -> torch.Tensor:
    """Slice ``t`` down to this rank's shard.

    ``dims`` maps a mesh key (``"row"`` / ``"col"``) to the tensor dimension it
    partitions.
    """
    out = t
    for key, d in dims.items():
        group = pmd[key]
        if group.size() > 1:
            out = out.chunk(group.size(), dim=d)[group.rank()]
    return out.contiguous()


def _gather(t: torch.Tensor, pmd: dict, dims: list) -> torch.Tensor:
    """Reassemble a global tensor from shards; ``dims`` is ``[(mesh_key, dim), ...]``."""
    out = t
    for key, d in dims:
        out = gather_1d_results(out, pmd, dim=d, pm_key=key)
    return out


def _global_logits(device: torch.device, n: int, c: int, s: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(n, c, s, device=device)


def _global_targets(device: torch.device, n: int, c: int, s: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(1234)
    return torch.randint(0, c, (n, s), generator=g).to(device)


# --------------------------------------------------------------------------- #
# no-group degenerate path: must match torch exactly, no collectives
# --------------------------------------------------------------------------- #
def test_functional_matches_torch_without_groups(device: torch.device) -> None:
    """With no groups the functionals collapse onto their torch equivalents."""
    torch.manual_seed(device.index or 0)
    logits = torch.randn(4, 7, 5, device=device, requires_grad=True)
    target = torch.randint(0, 7, (4, 5), device=device)

    for reduction in ("mean", "sum", "none"):
        got = lossc.distributed_cross_entropy(logits, target, reduction=reduction)
        ref = F.cross_entropy(logits, target, reduction=reduction)
        assert torch.allclose(got, ref, **FTOL)

    log_probs = F.log_softmax(logits, dim=1)
    got = lossc.distributed_nll_loss(log_probs, target)
    ref = F.nll_loss(log_probs, target)
    assert torch.allclose(got, ref, **FTOL)

    probs = torch.rand(3, 6, device=device).clamp(0.02, 0.98)
    bt = torch.randint(0, 2, (3, 6), device=device).float()
    assert torch.allclose(
        lossc.distributed_binary_cross_entropy(probs, bt),
        F.binary_cross_entropy(probs, bt),
        **FTOL,
    )
    raw = torch.randn(3, 6, device=device)
    assert torch.allclose(
        lossc.distributed_bce_with_logits(raw, bt),
        F.binary_cross_entropy_with_logits(raw, bt),
        **FTOL,
    )

    pred = torch.randn(4, 7, 5, device=device)
    tgt = torch.randn(4, 7, 5, device=device)
    for reduction in ("mean", "sum", "none"):
        assert torch.allclose(
            lossr.distributed_mse_loss(pred, tgt, reduction=reduction),
            F.mse_loss(pred, tgt, reduction=reduction),
            **FTOL,
        )
        assert torch.allclose(
            lossr.distributed_l1_loss(pred, tgt, reduction=reduction),
            F.l1_loss(pred, tgt, reduction=reduction),
            **FTOL,
        )
        assert torch.allclose(
            lossr.distributed_smooth_l1_loss(pred, tgt, beta=0.7, reduction=reduction),
            F.smooth_l1_loss(pred, tgt, beta=0.7, reduction=reduction),
            **FTOL,
        )


# --------------------------------------------------------------------------- #
# cross entropy
# --------------------------------------------------------------------------- #
@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_name", MESHES_4)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cross_entropy_forward(
    request: pytest.FixtureRequest,
    device: torch.device,
    mesh_name: str,
    reduction: str,
) -> None:
    """Distributed cross entropy forward matches ``F.cross_entropy``."""
    pmd = request.getfixturevalue(mesh_name)
    n, c, s = 3, 8, 16

    logits = _global_logits(device, n, c, s)
    target = _global_targets(device, n, c, s)

    local_logits = _shard(logits, pmd, {"col": 1, "row": 2})
    local_target = _shard(target, pmd, {"row": 1})

    got = lossc.distributed_cross_entropy(
        local_logits,
        local_target,
        reduction=reduction,
        channel_group=pmd["col"],
        reduce_groups=(pmd["row"],),
    )
    ref = F.cross_entropy(logits, target, reduction=reduction)
    if reduction == "none":
        got = _gather(got, pmd, [("row", 1)])
    assert torch.allclose(got, ref, **FTOL), (got - ref).abs().max()


@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_name", MESHES_4)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_cross_entropy_backward(
    request: pytest.FixtureRequest,
    device: torch.device,
    mesh_name: str,
    reduction: str,
) -> None:
    """Distributed cross entropy gradient wrt the logits matches the reference."""
    pmd = request.getfixturevalue(mesh_name)
    n, c, s = 3, 8, 16

    logits = _global_logits(device, n, c, s)
    target = _global_targets(device, n, c, s)

    local_logits = _shard(logits, pmd, {"col": 1, "row": 2}).requires_grad_(True)
    local_target = _shard(target, pmd, {"row": 1})

    loss = lossc.distributed_cross_entropy(
        local_logits,
        local_target,
        reduction=reduction,
        channel_group=pmd["col"],
        reduce_groups=(pmd["row"],),
    )
    loss.backward()

    ref_logits = logits.detach().clone().requires_grad_(True)
    ref_loss = F.cross_entropy(ref_logits, target, reduction=reduction)
    ref_loss.backward()

    assert torch.allclose(loss.detach(), ref_loss.detach(), **FTOL)
    assert local_logits.grad is not None
    grad = _gather(local_logits.grad, pmd, [("col", 1), ("row", 2)])
    assert torch.allclose(grad, ref_logits.grad, **FTOL), (
        (grad - ref_logits.grad).abs().max()
    )


@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_name", MESHES_4)
def test_cross_entropy_weight(
    request: pytest.FixtureRequest, device: torch.device, mesh_name: str
) -> None:
    """Per-class ``weight`` is honoured in forward and backward."""
    pmd = request.getfixturevalue(mesh_name)
    n, c, s = 3, 8, 16

    logits = _global_logits(device, n, c, s)
    target = _global_targets(device, n, c, s)
    torch.manual_seed(7)
    weight = torch.rand(c, device=device) + 0.1

    local_logits = _shard(logits, pmd, {"col": 1, "row": 2}).requires_grad_(True)
    local_target = _shard(target, pmd, {"row": 1})

    loss = lossc.distributed_cross_entropy(
        local_logits,
        local_target,
        weight=weight,
        channel_group=pmd["col"],
        reduce_groups=(pmd["row"],),
    )
    loss.backward()

    ref_logits = logits.detach().clone().requires_grad_(True)
    ref_loss = F.cross_entropy(ref_logits, target, weight=weight)
    ref_loss.backward()

    assert torch.allclose(loss.detach(), ref_loss.detach(), **FTOL)
    grad = _gather(local_logits.grad, pmd, [("col", 1), ("row", 2)])
    assert torch.allclose(grad, ref_logits.grad, **FTOL)


@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_name", MESHES_4)
def test_cross_entropy_ignore_index(
    request: pytest.FixtureRequest, device: torch.device, mesh_name: str
) -> None:
    """``ignore_index`` elements drop out of both loss and normalization."""
    pmd = request.getfixturevalue(mesh_name)
    n, c, s = 3, 8, 16

    logits = _global_logits(device, n, c, s)
    target = _global_targets(device, n, c, s)
    target[:, ::4] = -100  # ignore a deterministic slice

    local_logits = _shard(logits, pmd, {"col": 1, "row": 2}).requires_grad_(True)
    local_target = _shard(target, pmd, {"row": 1})

    loss = lossc.distributed_cross_entropy(
        local_logits,
        local_target,
        ignore_index=-100,
        channel_group=pmd["col"],
        reduce_groups=(pmd["row"],),
    )
    loss.backward()

    ref_logits = logits.detach().clone().requires_grad_(True)
    ref_loss = F.cross_entropy(ref_logits, target, ignore_index=-100)
    ref_loss.backward()

    assert torch.allclose(loss.detach(), ref_loss.detach(), **FTOL)
    grad = _gather(local_logits.grad, pmd, [("col", 1), ("row", 2)])
    assert torch.allclose(grad, ref_logits.grad, **FTOL)


@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_name", MESHES_4)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_cross_entropy_label_smoothing(
    request: pytest.FixtureRequest,
    device: torch.device,
    mesh_name: str,
    reduction: str,
) -> None:
    """Label smoothing matches ``F.cross_entropy(label_smoothing=...)``."""
    pmd = request.getfixturevalue(mesh_name)
    n, c, s = 3, 8, 16

    logits = _global_logits(device, n, c, s)
    target = _global_targets(device, n, c, s)

    local_logits = _shard(logits, pmd, {"col": 1, "row": 2}).requires_grad_(True)
    local_target = _shard(target, pmd, {"row": 1})

    loss = lossc.distributed_cross_entropy(
        local_logits,
        local_target,
        reduction=reduction,
        label_smoothing=0.1,
        channel_group=pmd["col"],
        reduce_groups=(pmd["row"],),
    )
    loss.backward()

    ref_logits = logits.detach().clone().requires_grad_(True)
    ref_loss = F.cross_entropy(
        ref_logits, target, reduction=reduction, label_smoothing=0.1
    )
    ref_loss.backward()

    assert torch.allclose(loss.detach(), ref_loss.detach(), **FTOL)
    grad = _gather(local_logits.grad, pmd, [("col", 1), ("row", 2)])
    assert torch.allclose(grad, ref_logits.grad, **FTOL)


@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_name", MESHES_4)
def test_cross_entropy_probability_target(
    request: pytest.FixtureRequest, device: torch.device, mesh_name: str
) -> None:
    """Cross entropy against soft (probability) targets matches the reference."""
    pmd = request.getfixturevalue(mesh_name)
    n, c, s = 3, 8, 16

    logits = _global_logits(device, n, c, s)
    torch.manual_seed(99)
    target = F.softmax(torch.randn(n, c, s, device=device), dim=1)

    local_logits = _shard(logits, pmd, {"col": 1, "row": 2}).requires_grad_(True)
    local_target = _shard(target, pmd, {"col": 1, "row": 2})

    loss = lossc.distributed_cross_entropy(
        local_logits,
        local_target,
        channel_group=pmd["col"],
        reduce_groups=(pmd["row"],),
    )
    loss.backward()

    ref_logits = logits.detach().clone().requires_grad_(True)
    ref_loss = F.cross_entropy(ref_logits, target)
    ref_loss.backward()

    assert torch.allclose(loss.detach(), ref_loss.detach(), **FTOL)
    grad = _gather(local_logits.grad, pmd, [("col", 1), ("row", 2)])
    assert torch.allclose(grad, ref_logits.grad, **FTOL)


# --------------------------------------------------------------------------- #
# nll
# --------------------------------------------------------------------------- #
@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_name", MESHES_4)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_nll_loss_forward_backward(
    request: pytest.FixtureRequest,
    device: torch.device,
    mesh_name: str,
    reduction: str,
) -> None:
    """Distributed NLL matches ``F.nll_loss`` on pre-computed log-probabilities."""
    pmd = request.getfixturevalue(mesh_name)
    n, c, s = 3, 8, 16

    logits = _global_logits(device, n, c, s)
    log_probs = F.log_softmax(logits, dim=1)
    target = _global_targets(device, n, c, s)

    local_lp = (
        _shard(log_probs, pmd, {"col": 1, "row": 2}).detach().requires_grad_(True)
    )
    local_target = _shard(target, pmd, {"row": 1})

    loss = lossc.distributed_nll_loss(
        local_lp,
        local_target,
        reduction=reduction,
        channel_group=pmd["col"],
        reduce_groups=(pmd["row"],),
    )

    ref_lp = log_probs.detach().clone().requires_grad_(True)
    ref_loss = F.nll_loss(ref_lp, target, reduction=reduction)

    if reduction == "none":
        got = _gather(loss, pmd, [("row", 1)])
        assert torch.allclose(got, ref_loss, **FTOL)
        return

    loss.backward()
    ref_loss.backward()
    assert torch.allclose(loss.detach(), ref_loss.detach(), **FTOL)
    grad = _gather(local_lp.grad, pmd, [("col", 1), ("row", 2)])
    assert torch.allclose(grad, ref_lp.grad, **FTOL)


# --------------------------------------------------------------------------- #
# bce
# --------------------------------------------------------------------------- #
@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_name", MESHES_4)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_bce_forward_backward(
    request: pytest.FixtureRequest,
    device: torch.device,
    mesh_name: str,
    reduction: str,
) -> None:
    """Distributed BCE matches ``F.binary_cross_entropy`` over an arbitrarily split tensor."""
    pmd = request.getfixturevalue(mesh_name)
    n, c, s = 3, 8, 16

    torch.manual_seed(0)
    probs = torch.rand(n, c, s, device=device).clamp(0.02, 0.98)
    torch.manual_seed(1)
    target = torch.randint(0, 2, (n, c, s), device=device).float()

    local_probs = _shard(probs, pmd, {"col": 1, "row": 2}).detach().requires_grad_(True)
    local_target = _shard(target, pmd, {"col": 1, "row": 2})

    loss = lossc.distributed_binary_cross_entropy(
        local_probs,
        local_target,
        reduction=reduction,
        reduce_groups=(pmd["col"], pmd["row"]),
    )

    ref_probs = probs.detach().clone().requires_grad_(True)
    ref_loss = F.binary_cross_entropy(ref_probs, target, reduction=reduction)

    if reduction == "none":
        got = _gather(loss, pmd, [("col", 1), ("row", 2)])
        assert torch.allclose(got, ref_loss, **FTOL)
        return

    loss.backward()
    ref_loss.backward()
    assert torch.allclose(loss.detach(), ref_loss.detach(), **FTOL)
    grad = _gather(local_probs.grad, pmd, [("col", 1), ("row", 2)])
    assert torch.allclose(grad, ref_probs.grad, **FTOL)


@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_name", MESHES_4)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_bce_with_logits_forward_backward(
    request: pytest.FixtureRequest,
    device: torch.device,
    mesh_name: str,
    reduction: str,
) -> None:
    """Distributed BCE-with-logits matches the reference, including ``pos_weight``."""
    pmd = request.getfixturevalue(mesh_name)
    n, c, s = 3, 8, 16

    logits = _global_logits(device, n, c, s)
    torch.manual_seed(2)
    target = torch.randint(0, 2, (n, c, s), device=device).float()
    torch.manual_seed(3)
    pos_weight = torch.rand(c, 1, device=device) + 0.5

    local_logits = _shard(logits, pmd, {"col": 1, "row": 2}).requires_grad_(True)
    local_target = _shard(target, pmd, {"col": 1, "row": 2})
    local_pw = _shard(pos_weight, pmd, {"col": 0})

    loss = lossc.distributed_bce_with_logits(
        local_logits,
        local_target,
        pos_weight=local_pw,
        reduction=reduction,
        reduce_groups=(pmd["col"], pmd["row"]),
    )
    loss.backward()

    ref_logits = logits.detach().clone().requires_grad_(True)
    ref_loss = F.binary_cross_entropy_with_logits(
        ref_logits, target, pos_weight=pos_weight, reduction=reduction
    )
    ref_loss.backward()

    assert torch.allclose(loss.detach(), ref_loss.detach(), **FTOL)
    grad = _gather(local_logits.grad, pmd, [("col", 1), ("row", 2)])
    assert torch.allclose(grad, ref_logits.grad, **FTOL)


# --------------------------------------------------------------------------- #
# regression losses: l1 / l2 (mse) / smooth-l1
# --------------------------------------------------------------------------- #
_REG_LOSSES = {
    "mse": (lossr.distributed_mse_loss, F.mse_loss),
    "l1": (lossr.distributed_l1_loss, F.l1_loss),
    "smooth_l1": (lossr.distributed_smooth_l1_loss, F.smooth_l1_loss),
}


@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_name", MESHES_4)
@pytest.mark.parametrize("loss_name", list(_REG_LOSSES))
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_regression_loss_forward_backward(
    request: pytest.FixtureRequest,
    device: torch.device,
    mesh_name: str,
    loss_name: str,
    reduction: str,
) -> None:
    """Distributed L1 / MSE / smooth-L1 match the reference over an arbitrarily split tensor."""
    pmd = request.getfixturevalue(mesh_name)
    distributed_fn, ref_fn = _REG_LOSSES[loss_name]
    n, c, s = 3, 8, 16

    torch.manual_seed(0)
    pred = torch.randn(n, c, s, device=device)
    torch.manual_seed(1)
    target = torch.randn(n, c, s, device=device)

    local_pred = _shard(pred, pmd, {"col": 1, "row": 2}).detach().requires_grad_(True)
    local_target = _shard(target, pmd, {"col": 1, "row": 2})

    got = distributed_fn(
        local_pred,
        local_target,
        reduction=reduction,
        reduce_groups=(pmd["col"], pmd["row"]),
    )
    ref_pred = pred.detach().clone().requires_grad_(True)
    ref = ref_fn(ref_pred, target, reduction=reduction)

    if reduction == "none":
        assert torch.allclose(_gather(got, pmd, [("col", 1), ("row", 2)]), ref, **FTOL)
        return

    got.backward()
    ref.backward()
    assert torch.allclose(got.detach(), ref.detach(), **FTOL)
    grad = _gather(local_pred.grad, pmd, [("col", 1), ("row", 2)])
    assert torch.allclose(grad, ref_pred.grad, **FTOL)


@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_name", MESHES_4)
def test_regression_module_wrappers(
    request: pytest.FixtureRequest, device: torch.device, mesh_name: str
) -> None:
    """The regression ``nn.Module`` wrappers agree with their torch counterparts."""
    pmd = request.getfixturevalue(mesh_name)
    n, c, s = 3, 8, 16
    groups = (pmd["col"], pmd["row"])

    torch.manual_seed(0)
    pred = torch.randn(n, c, s, device=device)
    torch.manual_seed(1)
    target = torch.randn(n, c, s, device=device)

    local_pred = _shard(pred, pmd, {"col": 1, "row": 2}).detach().requires_grad_(True)
    local_target = _shard(target, pmd, {"col": 1, "row": 2})
    ref_pred = pred.detach().clone().requires_grad_(True)

    for distributed_mod, ref_mod in (
        (jloss.DistributedMSELoss(groups), torch.nn.MSELoss()),
        (jloss.DistributedL1Loss(groups), torch.nn.L1Loss()),
        (
            jloss.DistributedSmoothL1Loss(groups, beta=0.5),
            torch.nn.SmoothL1Loss(beta=0.5),
        ),
    ):
        got = distributed_mod(local_pred, local_target)
        ref = ref_mod(ref_pred, target)
        assert torch.allclose(got.detach(), ref.detach(), **FTOL)


# --------------------------------------------------------------------------- #
# eight ranks: 4-way spatial x 2-way class
# --------------------------------------------------------------------------- #
@pytest.mark.dist(8)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("loss_kind", ["cross_entropy", "nll", "bce", "mse", "l1"])
def test_eightway(
    request: pytest.FixtureRequest,
    device: torch.device,
    reduction: str,
    loss_kind: str,
) -> None:
    """All losses stay at parity on an eight-rank ``4 x 2`` mesh."""
    pmd = request.getfixturevalue("process_mesh_dict_4x2")
    row, col = "row", "col"  # size 4 (spatial) x size 2 (class)
    n, c, s = 3, 8, 16

    if loss_kind == "cross_entropy":
        global_input = _global_logits(device, n, c, s)
        target = _global_targets(device, n, c, s)
        distributed = lambda x, t: lossc.distributed_cross_entropy(  # noqa: E731
            x,
            t,
            reduction=reduction,
            channel_group=pmd[col],
            reduce_groups=(pmd[row],),
        )
        reference = lambda x, t: F.cross_entropy(x, t, reduction=reduction)  # noqa: E731
        input_dims, target_dims = {col: 1, row: 2}, {row: 1}
        none_gather = [(row, 1)]
    elif loss_kind == "nll":
        global_input = F.log_softmax(_global_logits(device, n, c, s), dim=1)
        target = _global_targets(device, n, c, s)
        distributed = lambda x, t: lossc.distributed_nll_loss(  # noqa: E731
            x,
            t,
            reduction=reduction,
            channel_group=pmd[col],
            reduce_groups=(pmd[row],),
        )
        reference = lambda x, t: F.nll_loss(x, t, reduction=reduction)  # noqa: E731
        input_dims, target_dims = {col: 1, row: 2}, {row: 1}
        none_gather = [(row, 1)]
    elif loss_kind == "bce":
        torch.manual_seed(0)
        global_input = torch.rand(n, c, s, device=device).clamp(0.02, 0.98)
        torch.manual_seed(1)
        target = torch.randint(0, 2, (n, c, s), device=device).float()
        distributed = lambda x, t: lossc.distributed_binary_cross_entropy(  # noqa: E731
            x,
            t,
            reduction=reduction,
            reduce_groups=(pmd[col], pmd[row]),
        )
        reference = lambda x, t: F.binary_cross_entropy(x, t, reduction=reduction)  # noqa: E731
        input_dims = target_dims = {col: 1, row: 2}
        none_gather = [(col, 1), (row, 2)]
    else:  # mse / l1
        torch.manual_seed(0)
        global_input = torch.randn(n, c, s, device=device)
        torch.manual_seed(1)
        target = torch.randn(n, c, s, device=device)
        dfn = (
            lossr.distributed_mse_loss
            if loss_kind == "mse"
            else lossr.distributed_l1_loss
        )
        rfn = F.mse_loss if loss_kind == "mse" else F.l1_loss
        distributed = lambda x, t: dfn(  # noqa: E731
            x, t, reduction=reduction, reduce_groups=(pmd[col], pmd[row])
        )
        reference = lambda x, t: rfn(x, t, reduction=reduction)  # noqa: E731
        input_dims = target_dims = {col: 1, row: 2}
        none_gather = [(col, 1), (row, 2)]

    local_input = _shard(global_input, pmd, input_dims).detach().requires_grad_(True)
    local_target = _shard(target, pmd, target_dims)

    got = distributed(local_input, local_target)
    ref_input = global_input.detach().clone().requires_grad_(True)
    ref = reference(ref_input, target)

    if reduction == "none":
        assert torch.allclose(_gather(got, pmd, none_gather), ref, **FTOL)
        return

    got.backward()
    ref.backward()
    assert torch.allclose(got.detach(), ref.detach(), **FTOL)
    grad = _gather(local_input.grad, pmd, [(col, 1), (row, 2)])
    assert torch.allclose(grad, ref_input.grad, **FTOL)


# --------------------------------------------------------------------------- #
# nn.Module wrappers + repeated backward
# --------------------------------------------------------------------------- #
@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_name", MESHES_4)
def test_module_wrappers_and_accumulation(
    request: pytest.FixtureRequest, device: torch.device, mesh_name: str
) -> None:
    """The ``nn.Module`` wrappers agree with the reference over several steps."""
    pmd = request.getfixturevalue(mesh_name)
    n, c, s = 3, 8, 16

    ce = jloss.DistributedCrossEntropyLoss(
        channel_group=pmd["col"], reduce_groups=(pmd["row"],)
    )
    ref_ce = torch.nn.CrossEntropyLoss()

    local_param = torch.zeros(
        _shard(torch.zeros(n, c, s), pmd, {"col": 1, "row": 2}).shape,
        device=device,
        requires_grad=True,
    )
    ref_param = torch.zeros(n, c, s, device=device, requires_grad=True)

    for step in range(3):
        logits = _global_logits(device, n, c, s) + step
        target = _global_targets(device, n, c, s)

        local_logits = _shard(logits, pmd, {"col": 1, "row": 2}) + local_param
        loss = ce(local_logits, _shard(target, pmd, {"row": 1}))
        loss.backward()

        ref_loss = ref_ce(logits + ref_param, target)
        ref_loss.backward()

        assert torch.allclose(loss.detach(), ref_loss.detach(), **FTOL)

    grad = _gather(local_param.grad, pmd, [("col", 1), ("row", 2)])
    assert torch.allclose(grad, ref_param.grad, **FTOL)
