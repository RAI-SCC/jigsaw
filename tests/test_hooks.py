"""Tests for the gradient hooks in jigsaw.hooks and the averaging helpers.

Every rank contributes an identical gradient, so with ``W`` ranks a SUM all-reduce
multiplies it by ``W`` and an AVG all-reduce leaves it unchanged. That holds at any
world size, including one.
"""

import pytest
import torch
import torch.distributed as dist

from jigsaw import hooks
from jigsaw.utils_aggregate import average_weights, average_weights_module


def _param(value: float = 2.0) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.full((3,), value))


def _backward(param: torch.nn.Parameter) -> None:
    (param * torch.arange(1.0, 4.0)).sum().backward()  # grad == [1, 2, 3]


EXPECTED = torch.arange(1.0, 4.0)


class TestGradSyncHook:
    @pytest.mark.parametrize("async_op", [True, False])
    def test_single_group_sums_gradients(self, async_op):
        world = dist.get_world_size()
        param = _param()
        param.register_post_accumulate_grad_hook(
            hooks.make_grad_sync_hook(dist.group.WORLD, async_op=async_op)
        )
        _backward(param)
        assert torch.equal(param.grad, EXPECTED * world)

    def test_async_waits_before_backward_returns(self):
        # Two parameters share one ``handles`` list: the wait callback is queued once
        # (on the first gradient) and must complete both reductions.
        world = dist.get_world_size()
        a, b = _param(), _param()
        hook = hooks.make_grad_sync_hook(dist.group.WORLD, async_op=True)
        a.register_post_accumulate_grad_hook(hook)
        b.register_post_accumulate_grad_hook(hook)
        ((a + b) * torch.arange(1.0, 4.0)).sum().backward()
        assert torch.equal(a.grad, EXPECTED * world)
        assert torch.equal(b.grad, EXPECTED * world)

    def test_async_hook_is_reusable_across_backward_passes(self):
        world = dist.get_world_size()
        param = _param()
        param.register_post_accumulate_grad_hook(
            hooks.make_grad_sync_hook(dist.group.WORLD, async_op=True)
        )
        _backward(param)
        param.grad = None
        _backward(param)
        assert torch.equal(param.grad, EXPECTED * world)

    def test_group_sequence_reduces_over_each_group_in_turn(self):
        world = dist.get_world_size()
        param = _param()
        groups = [dist.group.WORLD, dist.group.WORLD]
        param.register_post_accumulate_grad_hook(
            hooks.make_grad_sync_hook(groups, async_op=True)  # async ignored
        )
        _backward(param)
        assert torch.equal(param.grad, EXPECTED * world * world)

    def test_reduce_op_is_honoured(self):
        param = _param()
        param.register_post_accumulate_grad_hook(
            hooks.make_grad_sync_hook(
                dist.group.WORLD, op=dist.ReduceOp.MAX, async_op=False
            )
        )
        _backward(param)
        assert torch.equal(param.grad, EXPECTED)

    def test_missing_gradient_is_skipped(self):
        param = _param()
        hooks.make_grad_sync_hook(dist.group.WORLD)(
            param
        )  # grad is None: no collective
        assert param.grad is None


class TestGradScalingHook:
    @pytest.mark.parametrize("parallelism", [1, 2, 4])
    def test_divides_gradient(self, parallelism):
        param = _param()
        param.register_post_accumulate_grad_hook(
            hooks.make_grad_scaling_hook(parallelism)
        )
        _backward(param)
        assert torch.allclose(param.grad, EXPECTED / parallelism)

    def test_missing_gradient_is_skipped(self):
        param = _param()
        hooks.make_grad_scaling_hook(4)(param)
        assert param.grad is None


class TestAveraging:
    def test_average_weights_keeps_identical_replicas_unchanged(self):
        tensors = [torch.full((2,), 3.0), torch.arange(4.0)]
        expected = [t.clone() for t in tensors]
        average_weights([tensors[0], None, tensors[1]], dist.group.WORLD)
        for got, want in zip(tensors, expected):
            assert torch.equal(got, want)

    @pytest.mark.parametrize("bias_only", [False, True])
    def test_average_weights_module(self, bias_only):
        torch.manual_seed(0)  # identical replicas on every rank
        model = torch.nn.Linear(3, 2)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        average_weights_module(model, dist.group.WORLD, bias_only=bias_only)
        for key, value in model.state_dict().items():
            assert torch.allclose(value, before[key])

    @pytest.mark.dist(2)
    def test_average_weights_averages_across_ranks(self):
        rank = dist.get_rank()
        weight = torch.full((2,), float(rank))
        average_weights([weight], dist.group.WORLD)
        assert torch.allclose(weight, torch.full((2,), 0.5))

    @pytest.mark.dist(2)
    def test_bias_only_leaves_weights_local(self):
        rank = dist.get_rank()
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.fill_(float(rank))
            model.bias.fill_(float(rank))
        average_weights_module(model, dist.group.WORLD, bias_only=True)
        assert torch.allclose(model.bias, torch.full((2,), 0.5))
        assert torch.all(model.weight == float(rank))
