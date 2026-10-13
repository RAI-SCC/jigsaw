"""Group-free checks of the loss ``nn.Module`` wrappers and argument validation.

With no process groups every distributed loss collapses onto its ``torch`` equivalent
and issues no collective, so these run identically in the single-process session and on
every rank of an ``mpirun`` job (rank-local data, no cross-rank comparison).
"""

import pytest
import torch
import torch.nn.functional as F

import jigsaw.nn.loss as jloss
from jigsaw.nn.activation import distributed_argmax, distributed_log_softmax
from jigsaw.nn.loss import _classification as lossc
from jigsaw.nn.loss._common import (
    reduce_loss,
    reduce_scalar_loss,
    redundant_all_reduce,
)

TOL = dict(atol=1e-6, rtol=1e-5)
REDUCTIONS = ["mean", "sum", "none"]


@pytest.fixture
def data():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 7, 5, generator=g)
    target = torch.randint(0, 7, (4, 5), generator=g)
    return logits, target


class TestClassificationModules:
    @pytest.mark.parametrize("reduction", REDUCTIONS)
    def test_nll(self, data, reduction):
        logits, target = data
        log_probs = F.log_softmax(logits, dim=1)
        module = jloss.DistributedNLLLoss(reduction=reduction)
        assert module.get_parameter_count() == 0
        assert torch.allclose(
            module(log_probs, target),
            F.nll_loss(log_probs, target, reduction=reduction),
            **TOL,
        )

    def test_nll_weight_and_ignore_index_follow_the_module(self, data):
        logits, target = data
        target = target.clone()
        target[0, 0] = -100
        weight = torch.rand(7) + 0.5
        log_probs = F.log_softmax(logits, dim=1)
        module = jloss.DistributedNLLLoss(weight=weight)
        assert "weight" in dict(module.named_buffers())
        assert torch.allclose(
            module(log_probs, target),
            F.nll_loss(log_probs, target, weight=weight),
            **TOL,
        )

    @pytest.mark.parametrize("reduction", REDUCTIONS)
    @pytest.mark.parametrize("smoothing", [0.0, 0.2])
    def test_cross_entropy(self, data, reduction, smoothing):
        logits, target = data
        module = jloss.DistributedCrossEntropyLoss(
            reduction=reduction, label_smoothing=smoothing
        )
        assert module.get_parameter_count() == 0
        ref = F.cross_entropy(
            logits, target, reduction=reduction, label_smoothing=smoothing
        )
        assert torch.allclose(module(logits, target), ref, **TOL)

    def test_cross_entropy_class_dim_last(self, data):
        logits, target = data
        module = jloss.DistributedCrossEntropyLoss(dim=-1)
        moved = logits.movedim(1, -1)
        assert torch.allclose(
            module(moved, target), F.cross_entropy(logits, target), **TOL
        )

    def test_cross_entropy_probability_targets(self, data):
        logits, _ = data
        probs = torch.softmax(torch.randn_like(logits), dim=1)
        module = jloss.DistributedCrossEntropyLoss()
        assert torch.allclose(
            module(logits, probs), F.cross_entropy(logits, probs), **TOL
        )

    @pytest.mark.parametrize("reduction", REDUCTIONS)
    def test_bce(self, reduction):
        g = torch.Generator().manual_seed(1)
        probs = torch.rand(3, 6, generator=g).clamp(0.02, 0.98)
        target = torch.randint(0, 2, (3, 6), generator=g).float()
        weight = torch.rand(6, generator=g) + 0.5
        module = jloss.DistributedBCELoss(weight=weight, reduction=reduction)
        assert module.get_parameter_count() == 0
        ref = F.binary_cross_entropy(probs, target, weight=weight, reduction=reduction)
        assert torch.allclose(module(probs, target), ref, **TOL)

    @pytest.mark.parametrize("reduction", REDUCTIONS)
    def test_bce_with_logits(self, reduction):
        g = torch.Generator().manual_seed(2)
        logits = torch.randn(3, 6, generator=g)
        target = torch.randint(0, 2, (3, 6), generator=g).float()
        weight, pos_weight = torch.rand(6) + 0.5, torch.rand(6) + 0.5
        module = jloss.DistributedBCEWithLogitsLoss(
            weight=weight, pos_weight=pos_weight, reduction=reduction
        )
        assert module.get_parameter_count() == 0
        ref = F.binary_cross_entropy_with_logits(
            logits, target, weight=weight, pos_weight=pos_weight, reduction=reduction
        )
        assert torch.allclose(module(logits, target), ref, **TOL)

    def test_module_gradients_match_torch(self, data):
        logits, target = data
        ours = logits.clone().requires_grad_()
        ref = logits.clone().requires_grad_()
        jloss.DistributedCrossEntropyLoss()(ours, target).backward()
        F.cross_entropy(ref, target).backward()
        assert torch.allclose(ours.grad, ref.grad, **TOL)


class TestRegressionModules:
    @pytest.mark.parametrize("reduction", REDUCTIONS)
    def test_mse_l1_smooth_l1(self, reduction):
        g = torch.Generator().manual_seed(3)
        pred, tgt = torch.randn(4, 5, generator=g), torch.randn(4, 5, generator=g)
        cases = [
            (
                jloss.DistributedMSELoss(reduction=reduction),
                F.mse_loss(pred, tgt, reduction=reduction),
            ),
            (
                jloss.DistributedL1Loss(reduction=reduction),
                F.l1_loss(pred, tgt, reduction=reduction),
            ),
            (
                jloss.DistributedSmoothL1Loss(beta=0.5, reduction=reduction),
                F.smooth_l1_loss(pred, tgt, beta=0.5, reduction=reduction),
            ),
        ]
        for module, ref in cases:
            assert module.get_parameter_count() == 0
            assert torch.allclose(module(pred, tgt), ref, **TOL)


class TestValidation:
    def test_cross_entropy_rejects_weight_with_probability_targets(self, data):
        logits, _ = data
        probs = torch.softmax(logits, dim=1)
        with pytest.raises(NotImplementedError, match="probability targets"):
            lossc.distributed_cross_entropy(logits, probs, weight=torch.ones(7))

    def test_cross_entropy_rejects_smoothing_with_probability_targets(self, data):
        logits, _ = data
        probs = torch.softmax(logits, dim=1)
        with pytest.raises(NotImplementedError, match="probability targets"):
            lossc.distributed_cross_entropy(logits, probs, label_smoothing=0.1)

    def test_cross_entropy_rejects_smoothing_with_weight(self, data):
        logits, target = data
        with pytest.raises(NotImplementedError, match="label_smoothing with weight"):
            lossc.distributed_cross_entropy(
                logits, target, weight=torch.ones(7), label_smoothing=0.1
            )

    @pytest.mark.parametrize(
        "call",
        [
            lambda x, t: reduce_loss(x, x, "bogus", ()),
            lambda x, t: reduce_scalar_loss(x.sum(), x.numel(), "bogus", ()),
        ],
        ids=["reduce_loss", "reduce_scalar_loss"],
    )
    def test_invalid_reduction(self, data, call):
        logits, target = data
        with pytest.raises(ValueError, match="invalid reduction"):
            call(logits, target)

    def test_module_propagates_invalid_reduction(self, data):
        logits, target = data
        with pytest.raises(ValueError, match="invalid reduction"):
            jloss.DistributedCrossEntropyLoss(reduction="bogus")(logits, target)


class TestGroupFreeHelpers:
    def test_argmax_matches_torch_and_prefers_lowest_index_on_ties(self):
        x = torch.tensor([[1.0, 3.0, 3.0], [5.0, 0.0, 5.0]])
        assert torch.equal(distributed_argmax(x, dim=1), x.argmax(dim=1))
        assert torch.equal(distributed_argmax(x, dim=0), x.argmax(dim=0))

    def test_log_softmax_matches_torch_forward_and_backward(self, data):
        logits, _ = data
        ours = logits.clone().requires_grad_()
        ref = logits.clone().requires_grad_()
        weights = torch.randn_like(logits)
        out = distributed_log_softmax(ours, dim=1)
        (out * weights).sum().backward()
        expected = F.log_softmax(ref, dim=1)
        (expected * weights).sum().backward()
        assert torch.allclose(out, expected, **TOL)
        assert torch.allclose(ours.grad, ref.grad, **TOL)

    @pytest.mark.parametrize("dim", [1, -2])
    @pytest.mark.parametrize("in_dim", [0, 1, 3])
    def test_log_softmax_under_vmap_respects_batch_and_class_dims(self, dim, in_dim):
        """Regression: the batching rule used to ignore ``in_dims`` and shift ``dim``."""
        x = torch.randn(4, 7, 5, dtype=torch.double)
        batched = torch.stack([x, 2 * x, -x], dim=in_dim)
        out = torch.vmap(
            lambda a: distributed_log_softmax(a, dim=dim),
            in_dims=in_dim,
            out_dims=in_dim,
        )(batched)
        ref = torch.stack(
            [F.log_softmax(t, dim=dim) for t in (x, 2 * x, -x)], dim=in_dim
        )
        assert torch.allclose(out, ref, **TOL)

    def test_log_softmax_grad_under_vmap(self):
        from torch.func import grad, vmap

        x = torch.randn(3, 4, 7, 5, dtype=torch.double)
        w = torch.randn(4, 7, 5, dtype=torch.double)

        def loss(a):
            return (distributed_log_softmax(a, dim=1) * w).sum()

        got = vmap(grad(loss))(x)
        ref = torch.stack([grad(loss)(x[i]) for i in range(3)])
        assert torch.allclose(got, ref, **TOL)

    def test_redundant_all_reduce_without_groups_is_identity_with_identity_grad(self):
        x = torch.randn(3, requires_grad=True)
        for groups in (None, (), (None,)):
            y = redundant_all_reduce(x, groups)
            assert torch.equal(y, x) and y is not x
        y.sum().backward()
        assert torch.equal(x.grad, torch.ones(3))
