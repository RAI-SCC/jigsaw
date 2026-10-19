"""Parity tests for the distributed loss functions.

Every test builds the *global* tensors identically on all ranks (seeded), slices
out this rank's shard, runs the distributed loss on the shard and compares
against the plain ``torch.nn.functional`` loss evaluated on the global tensor.

Layout used throughout: logits of shape ``[N, C, S]`` where the class dimension
``C`` is sharded across the ``col`` group and the spatial dimension ``S`` across
the ``row`` group.  ``N`` is never sharded.  Hence ``channel_group == col`` and
``reduce_groups == (row,)`` for the classification losses; for BCE (no class
dimension) both groups shard the tensor and both appear in ``reduce_groups``.

Comparisons run through ``tests.parity`` (``assert_close`` / ``all_ranks_ok``) so
a mismatch on a subset of ranks fails the job instead of hanging the rest in the
next collective, exactly like the linear / attention / normalization suites.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import jigsaw.modules.linear as linear
import jigsaw.modules.loss as jloss
from jigsaw.gather import gather_1d_results, gather_results
from jigsaw.modules.loss import classification as lossc
from jigsaw.modules.loss import regression as lossr
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]

MESHES_4 = [
    "process_mesh_dict_2x2",
    "process_mesh_dict_1x4",
    "process_mesh_dict_4x1",
    "process_mesh_dict_2x1x2",
    "process_mesh_dict_2x2x1",
]

FATOL, FRTOL = 1e-5, 1e-4
FTOL = dict(atol=FATOL, rtol=FRTOL)


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
class TestFunctionalNoGroups:
    """With no groups the functionals collapse onto their torch equivalents.

    These run on every rank with rank-local data and issue no collectives, so a
    bare ``allclose`` cannot hang anything -- a failing rank just fails.
    """

    def test_functional_matches_torch_without_groups(
        self, device: torch.device
    ) -> None:
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
            lossc.distributed_binary_cross_entropy_with_logits(raw, bt),
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
                lossr.distributed_smooth_l1_loss(
                    pred, tgt, beta=0.7, reduction=reduction
                ),
                F.smooth_l1_loss(pred, tgt, beta=0.7, reduction=reduction),
                **FTOL,
            )

    @pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
    def test_vmap_grad_matches_loop_without_groups(
        self, device: torch.device, label_smoothing: float
    ) -> None:
        """``vmap(grad(...))`` over the ungrouped losses matches a Python loop.

        Batches over a leading axis so each slice keeps its ``[N, C, S]`` shape;
        no groups means no collectives, so this is safe on every rank.
        """
        from torch.func import grad, vmap

        e, n, c, s = 3, 4, 6, 5
        torch.manual_seed(device.index or 0)
        logits = torch.randn(e, n, c, s, device=device, dtype=torch.double)
        target = torch.randint(0, c, (e, n, s), device=device)

        def ce(lg: torch.Tensor, tg: torch.Tensor) -> torch.Tensor:
            return lossc.distributed_cross_entropy(
                lg, tg, reduction="mean", label_smoothing=label_smoothing
            )

        got = vmap(grad(ce))(logits, target)
        ref = torch.stack([grad(ce)(logits[i], target[i]) for i in range(e)])
        assert torch.allclose(got, ref, atol=1e-10)

        pred = torch.randn(e, n, c, device=device, dtype=torch.double)
        cont = torch.randn(e, n, c, device=device, dtype=torch.double)

        def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return lossr.distributed_mse_loss(a, b, reduction="mean")

        got = vmap(grad(mse))(pred, cont)
        ref = torch.stack([grad(mse)(pred[i], cont[i]) for i in range(e)])
        assert torch.allclose(got, ref, atol=1e-10)


# --------------------------------------------------------------------------- #
# cross entropy
# --------------------------------------------------------------------------- #
class TestCrossEntropy:
    pytestmark = pytest.mark.dist(4)

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_cross_entropy_forward(
        self,
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
        assert_close("cross entropy", got, ref, FATOL, FRTOL, device, pmd["model"])

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_cross_entropy_backward(
        self,
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

        assert_close(
            "loss", loss.detach(), ref_loss.detach(), FATOL, FRTOL, device, pmd["model"]
        )
        assert local_logits.grad is not None
        grad = _gather(local_logits.grad, pmd, [("col", 1), ("row", 2)])
        assert_close(
            "logit grad", grad, ref_logits.grad, FATOL, FRTOL, device, pmd["model"]
        )

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    def test_cross_entropy_weight(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
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

        assert_close(
            "loss", loss.detach(), ref_loss.detach(), FATOL, FRTOL, device, pmd["model"]
        )
        grad = _gather(local_logits.grad, pmd, [("col", 1), ("row", 2)])
        assert_close(
            "logit grad", grad, ref_logits.grad, FATOL, FRTOL, device, pmd["model"]
        )

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    def test_cross_entropy_ignore_index(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
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

        assert_close(
            "loss", loss.detach(), ref_loss.detach(), FATOL, FRTOL, device, pmd["model"]
        )
        grad = _gather(local_logits.grad, pmd, [("col", 1), ("row", 2)])
        assert_close(
            "logit grad", grad, ref_logits.grad, FATOL, FRTOL, device, pmd["model"]
        )

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_cross_entropy_label_smoothing(
        self,
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

        assert_close(
            "loss", loss.detach(), ref_loss.detach(), FATOL, FRTOL, device, pmd["model"]
        )
        grad = _gather(local_logits.grad, pmd, [("col", 1), ("row", 2)])
        assert_close(
            "logit grad", grad, ref_logits.grad, FATOL, FRTOL, device, pmd["model"]
        )

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    def test_cross_entropy_probability_target(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
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

        assert_close(
            "loss", loss.detach(), ref_loss.detach(), FATOL, FRTOL, device, pmd["model"]
        )
        grad = _gather(local_logits.grad, pmd, [("col", 1), ("row", 2)])
        assert_close(
            "logit grad", grad, ref_logits.grad, FATOL, FRTOL, device, pmd["model"]
        )


# --------------------------------------------------------------------------- #
# nll
# --------------------------------------------------------------------------- #
class TestNLLLoss:
    pytestmark = pytest.mark.dist(4)

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_nll_loss_forward_backward(
        self,
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
            assert_close("nll", got, ref_loss, FATOL, FRTOL, device, pmd["model"])
            return

        loss.backward()
        ref_loss.backward()
        assert_close(
            "loss", loss.detach(), ref_loss.detach(), FATOL, FRTOL, device, pmd["model"]
        )
        grad = _gather(local_lp.grad, pmd, [("col", 1), ("row", 2)])
        assert_close(
            "log-prob grad", grad, ref_lp.grad, FATOL, FRTOL, device, pmd["model"]
        )


# --------------------------------------------------------------------------- #
# bce
# --------------------------------------------------------------------------- #
class TestBCE:
    pytestmark = pytest.mark.dist(4)

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_bce_forward_backward(
        self,
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

        local_probs = (
            _shard(probs, pmd, {"col": 1, "row": 2}).detach().requires_grad_(True)
        )
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
            assert_close("bce", got, ref_loss, FATOL, FRTOL, device, pmd["model"])
            return

        loss.backward()
        ref_loss.backward()
        assert_close(
            "loss", loss.detach(), ref_loss.detach(), FATOL, FRTOL, device, pmd["model"]
        )
        grad = _gather(local_probs.grad, pmd, [("col", 1), ("row", 2)])
        assert_close(
            "prob grad", grad, ref_probs.grad, FATOL, FRTOL, device, pmd["model"]
        )

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_bce_with_logits_forward_backward(
        self,
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

        loss = lossc.distributed_binary_cross_entropy_with_logits(
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

        assert_close(
            "loss", loss.detach(), ref_loss.detach(), FATOL, FRTOL, device, pmd["model"]
        )
        grad = _gather(local_logits.grad, pmd, [("col", 1), ("row", 2)])
        assert_close(
            "logit grad", grad, ref_logits.grad, FATOL, FRTOL, device, pmd["model"]
        )


# --------------------------------------------------------------------------- #
# regression losses: l1 / l2 (mse) / smooth-l1
# --------------------------------------------------------------------------- #
_REG_LOSSES = {
    "mse": (lossr.distributed_mse_loss, F.mse_loss),
    "l1": (lossr.distributed_l1_loss, F.l1_loss),
    "smooth_l1": (lossr.distributed_smooth_l1_loss, F.smooth_l1_loss),
}


class TestRegressionLosses:
    pytestmark = pytest.mark.dist(4)

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    @pytest.mark.parametrize("loss_name", list(_REG_LOSSES))
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_regression_loss_forward_backward(
        self,
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

        local_pred = (
            _shard(pred, pmd, {"col": 1, "row": 2}).detach().requires_grad_(True)
        )
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
            got = _gather(got, pmd, [("col", 1), ("row", 2)])
            assert_close(loss_name, got, ref, FATOL, FRTOL, device, pmd["model"])
            return

        got.backward()
        ref.backward()
        assert_close(
            "loss", got.detach(), ref.detach(), FATOL, FRTOL, device, pmd["model"]
        )
        grad = _gather(local_pred.grad, pmd, [("col", 1), ("row", 2)])
        assert_close(
            "pred grad", grad, ref_pred.grad, FATOL, FRTOL, device, pmd["model"]
        )

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    def test_regression_module_wrappers(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """The regression ``nn.Module`` wrappers agree with their torch counterparts."""
        pmd = request.getfixturevalue(mesh_name)
        n, c, s = 3, 8, 16
        groups = (pmd["col"], pmd["row"])

        torch.manual_seed(0)
        pred = torch.randn(n, c, s, device=device)
        torch.manual_seed(1)
        target = torch.randn(n, c, s, device=device)

        local_pred = (
            _shard(pred, pmd, {"col": 1, "row": 2}).detach().requires_grad_(True)
        )
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
            assert_close(
                type(distributed_mod).__name__,
                got.detach(),
                ref.detach(),
                FATOL,
                FRTOL,
                device,
                pmd["model"],
            )


# --------------------------------------------------------------------------- #
# eight ranks: 4-way spatial x 2-way class
# --------------------------------------------------------------------------- #
class TestEightWay:
    pytestmark = pytest.mark.dist(8)

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("loss_kind", ["cross_entropy", "nll", "bce", "mse", "l1"])
    def test_eightway(
        self,
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

        local_input = (
            _shard(global_input, pmd, input_dims).detach().requires_grad_(True)
        )
        local_target = _shard(target, pmd, target_dims)

        got = distributed(local_input, local_target)
        ref_input = global_input.detach().clone().requires_grad_(True)
        ref = reference(ref_input, target)

        if reduction == "none":
            got = _gather(got, pmd, none_gather)
            assert_close(loss_kind, got, ref, FATOL, FRTOL, device, pmd["model"])
            return

        got.backward()
        ref.backward()
        assert_close(
            "loss", got.detach(), ref.detach(), FATOL, FRTOL, device, pmd["model"]
        )
        grad = _gather(local_input.grad, pmd, [(col, 1), (row, 2)])
        assert_close(
            "input grad", grad, ref_input.grad, FATOL, FRTOL, device, pmd["model"]
        )


# --------------------------------------------------------------------------- #
# nn.Module wrappers + repeated backward
# --------------------------------------------------------------------------- #
class TestModuleWrappers:
    pytestmark = pytest.mark.dist(4)

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    def test_module_wrappers_and_accumulation(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
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

            assert_close(
                f"loss step {step}",
                loss.detach(),
                ref_loss.detach(),
                FATOL,
                FRTOL,
                device,
                pmd["model"],
            )

        grad = _gather(local_param.grad, pmd, [("col", 1), ("row", 2)])
        assert_close(
            "accumulated grad", grad, ref_param.grad, FATOL, FRTOL, device, pmd["model"]
        )


# --------------------------------------------------------------------------- #
# end-to-end: a distributed linear model trained with a distributed loss
# --------------------------------------------------------------------------- #
class TestLinearTrainingParity:
    """A ``DistributedLinear`` trained for a few SGD steps with a distributed loss
    reproduces the *gathered* ``nn.Linear`` trained with the stock ``torch`` loss
    it mirrors -- identical per-step loss values and identical weights after
    every step.

    ``DistributedLinear`` shards the weight ``(out, in)`` with ``out`` across ``row``
    and ``in`` across ``col``; activations are ``(1, seq, features)`` with ``seq``
    across ``row`` and ``features`` across ``col``.  The distributed loss is
    handed ``channel_group`` / ``reduce_groups`` matching that layout and no
    manual ``/ world_size`` rescaling -- it does the global reduction itself.
    """

    pytestmark = pytest.mark.dist(4)

    MESHES = ["process_mesh_dict_2x2", "process_mesh_dict_2x1x2"]

    @staticmethod
    def _make_pair(
        pmd: dict, in_dim: int, out_dim: int, device: torch.device
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a ``DistributedLinear`` and a gathered-weight ``nn.Linear`` twin."""
        model_rank = dist.get_rank(group=pmd["model"])
        torch.manual_seed(model_rank)
        dist_lin = linear.DistributedLinear(
            in_dim,
            out_dim,
            local_rank=model_rank,
            device=device,
            model_group=pmd["model"],
            col_group=pmd["col"],
            row_group=pmd["row"],
            bias=False,
            bayesian=False,
        ).to(device)
        w = gather_results(dist_lin.weight.data, process_mesh_dict=pmd)
        ref_lin = torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
        ref_lin.weight.data = w.clone()
        return dist_lin, ref_lin

    @pytest.mark.parametrize("mesh_name", MESHES)
    def test_mse_training_parity(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """``DistributedMSELoss`` on the sharded output == ``nn.MSELoss`` on the whole."""
        pmd = request.getfixturevalue(mesh_name)
        in_dim, out_dim, seq_local, steps = 6, 8, 5, 5
        n_col = pmd["col"].size()
        model_rank = dist.get_rank(group=pmd["model"])

        dist_lin, ref_lin = self._make_pair(pmd, in_dim, out_dim, device)
        loss_mod = jloss.DistributedMSELoss((pmd["row"], pmd["col"]), reduction="mean")
        ref_loss_mod = torch.nn.MSELoss(reduction="mean")
        opt = torch.optim.SGD(dist_lin.parameters(), lr=0.3)
        ref_opt = torch.optim.SGD(ref_lin.parameters(), lr=0.3)

        for step in range(steps):
            opt.zero_grad()
            ref_opt.zero_grad()

            torch.manual_seed(100 + 10 * step + model_rank)
            x = torch.randn(
                1,
                seq_local,
                in_dim // n_col,
                dtype=dist_lin.weight.dtype,
                device=device,
            )
            torch.manual_seed(500 + 10 * step + model_rank)
            target = torch.randn(
                1,
                seq_local,
                out_dim // n_col,
                dtype=dist_lin.weight.dtype,
                device=device,
            )

            loss = loss_mod(dist_lin(x), target)
            loss.backward()
            opt.step()

            ref_y = ref_lin(gather_results(x.detach(), process_mesh_dict=pmd))
            ref_loss = ref_loss_mod(
                ref_y, gather_results(target, process_mesh_dict=pmd)
            )
            ref_loss.backward()
            ref_opt.step()

            assert_close(
                f"mse loss step {step}",
                loss.detach(),
                ref_loss.detach(),
                FATOL,
                FRTOL,
                device,
                pmd["model"],
            )
            w_gathered = gather_results(dist_lin.weight.data, process_mesh_dict=pmd)
            assert_close(
                f"weight step {step}",
                w_gathered,
                ref_lin.weight.data,
                FATOL,
                FRTOL,
                device,
                pmd["model"],
            )

    @pytest.mark.parametrize("mesh_name", MESHES)
    def test_cross_entropy_training_parity(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """``DistributedCrossEntropyLoss`` on the sharded logits == ``F.cross_entropy``."""
        pmd = request.getfixturevalue(mesh_name)
        in_dim, out_dim, seq_local, steps = 6, 8, 5, 5
        n_col = pmd["col"].size()
        model_rank = dist.get_rank(group=pmd["model"])
        row_rank = pmd["row"].rank()

        dist_lin, ref_lin = self._make_pair(pmd, in_dim, out_dim, device)
        ce = jloss.DistributedCrossEntropyLoss(
            channel_group=pmd["col"], reduce_groups=(pmd["row"],), dim=-1
        )
        opt = torch.optim.SGD(dist_lin.parameters(), lr=0.3)
        ref_opt = torch.optim.SGD(ref_lin.parameters(), lr=0.3)

        for step in range(steps):
            opt.zero_grad()
            ref_opt.zero_grad()

            torch.manual_seed(100 + 10 * step + model_rank)
            x = torch.randn(
                1,
                seq_local,
                in_dim // n_col,
                dtype=dist_lin.weight.dtype,
                device=device,
            )
            # the label indexes a shared sequence position, so it must be
            # identical across the class (``col``) group -- seed by the row.
            torch.manual_seed(700 + 10 * step + row_rank)
            target = torch.randint(0, out_dim, (1, seq_local), device=device)

            loss = ce(dist_lin(x), target)
            loss.backward()
            opt.step()

            ref_y = ref_lin(gather_results(x.detach(), process_mesh_dict=pmd))
            ref_target = gather_results(target.unsqueeze(-1), process_mesh_dict=pmd)[
                ..., 0
            ]
            ref_loss = F.cross_entropy(ref_y.transpose(1, 2), ref_target)
            ref_loss.backward()
            ref_opt.step()

            assert_close(
                f"ce loss step {step}",
                loss.detach(),
                ref_loss.detach(),
                FATOL,
                FRTOL,
                device,
                pmd["model"],
            )
            w_gathered = gather_results(dist_lin.weight.data, process_mesh_dict=pmd)
            assert_close(
                f"weight step {step}",
                w_gathered,
                ref_lin.weight.data,
                FATOL,
                FRTOL,
                device,
                pmd["model"],
            )
