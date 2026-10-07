"""Tests for the gradient hooks in jigsaw.hooks and the averaging helpers.

Every rank contributes an identical gradient, so with ``W`` ranks a SUM all-reduce
multiplies it by ``W`` and an AVG all-reduce leaves it unchanged. That holds at any
world size, including one.
"""

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from jigsaw import hooks
from jigsaw.utils_aggregate import average_weights, average_weights_module


def _param(
    device: torch.device = torch.device("cpu"), value: float = 2.0
) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.full((3,), value, device=device))


def _backward(param: torch.nn.Parameter) -> None:
    # grad == [1, 2, 3]
    (param * torch.arange(1.0, 4.0, device=param.device)).sum().backward()


EXPECTED = torch.arange(1.0, 4.0)


class TestGradSyncHook:
    @pytest.mark.parametrize("async_op", [True, False])
    def test_single_group_sums_gradients(self, device, async_op):
        world = dist.get_world_size()
        param = _param(device)
        param.register_post_accumulate_grad_hook(
            hooks.make_grad_sync_hook(dist.group.WORLD, async_op=async_op)
        )
        _backward(param)
        assert torch.equal(param.grad.cpu(), EXPECTED * world)

    def test_async_waits_before_backward_returns(self, device):
        # Two parameters share one ``handles`` list: the wait callback is queued once
        # (on the first gradient) and must complete both reductions.
        world = dist.get_world_size()
        a, b = _param(device), _param(device)
        hook = hooks.make_grad_sync_hook(dist.group.WORLD, async_op=True)
        a.register_post_accumulate_grad_hook(hook)
        b.register_post_accumulate_grad_hook(hook)
        ((a + b) * torch.arange(1.0, 4.0, device=device)).sum().backward()
        assert torch.equal(a.grad.cpu(), EXPECTED * world)
        assert torch.equal(b.grad.cpu(), EXPECTED * world)

    def test_async_hook_is_reusable_across_backward_passes(self, device):
        world = dist.get_world_size()
        param = _param(device)
        param.register_post_accumulate_grad_hook(
            hooks.make_grad_sync_hook(dist.group.WORLD, async_op=True)
        )
        _backward(param)
        param.grad = None
        _backward(param)
        assert torch.equal(param.grad.cpu(), EXPECTED * world)

    def test_group_sequence_reduces_over_each_group_in_turn(self, device):
        world = dist.get_world_size()
        param = _param(device)
        groups = [dist.group.WORLD, dist.group.WORLD]
        param.register_post_accumulate_grad_hook(
            hooks.make_grad_sync_hook(groups, async_op=True)  # async ignored
        )
        _backward(param)
        assert torch.equal(param.grad.cpu(), EXPECTED * world * world)

    def test_reduce_op_is_honoured(self, device):
        param = _param(device)
        param.register_post_accumulate_grad_hook(
            hooks.make_grad_sync_hook(
                dist.group.WORLD, op=dist.ReduceOp.MAX, async_op=False
            )
        )
        _backward(param)
        assert torch.equal(param.grad.cpu(), EXPECTED)

    def test_missing_gradient_is_skipped(self, device):
        param = _param(device)
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
    def test_average_weights_keeps_identical_replicas_unchanged(self, device):
        tensors = [
            torch.full((2,), 3.0, device=device),
            torch.arange(4.0, device=device),
        ]
        expected = [t.clone() for t in tensors]
        average_weights([tensors[0], None, tensors[1]], dist.group.WORLD)
        for got, want in zip(tensors, expected):
            assert torch.equal(got, want)

    @pytest.mark.parametrize("bias_only", [False, True])
    def test_average_weights_module(self, device, bias_only):
        torch.manual_seed(0)  # identical replicas on every rank
        model = torch.nn.Linear(3, 2).to(device)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        average_weights_module(model, dist.group.WORLD, bias_only=bias_only)
        for key, value in model.state_dict().items():
            assert torch.allclose(value, before[key])

    @pytest.mark.dist(2)
    def test_average_weights_averages_across_ranks(self, device):
        rank = dist.get_rank()
        weight = torch.full((2,), float(rank), device=device)
        average_weights([weight], dist.group.WORLD)
        assert torch.allclose(weight.cpu(), torch.full((2,), 0.5))

    @pytest.mark.dist(2)
    def test_bias_only_leaves_weights_local(self, device):
        rank = dist.get_rank()
        model = torch.nn.Linear(2, 2).to(device)
        with torch.no_grad():
            model.weight.fill_(float(rank))
            model.bias.fill_(float(rank))
        average_weights_module(model, dist.group.WORLD, bias_only=True)
        assert torch.allclose(model.bias.cpu(), torch.full((2,), 0.5))
        assert torch.all(model.weight == float(rank))


class TestMpDpCommHook:
    """The DDP communication hook that reduces model- then data-parallel.

    A deferred-wait post-accumulate-grad hook lets DDP reduce a gradient the
    model-parallel all-reduce has not finished writing, which shows up as
    replicas drifting apart over successive steps. These tests pin the ordering.
    """

    N = 32
    LAYERS = 3
    STEPS = 3

    @staticmethod
    def _device_ids(device: torch.device) -> "list[int] | None":
        """Return the ``device_ids`` DDP needs for this device."""
        if device.type == "cpu":
            return None
        return [torch.cuda.current_device()]

    @classmethod
    def _model(cls, device: torch.device) -> torch.nn.Module:
        """Build a model that is identical on every rank."""
        torch.manual_seed(0)
        return torch.nn.Sequential(
            *[torch.nn.Linear(cls.N, cls.N, bias=False) for _ in range(cls.LAYERS)]
        ).to(device)

    @pytest.mark.dist(4)
    def test_matches_blocking_reference_and_keeps_replicas_identical(self, device):
        # Split WORLD 2x2: model-parallel {0,1},{2,3}, data-parallel {0,2},{1,3}.
        rank, world = dist.get_rank(), dist.get_world_size()
        half = world // 2
        mp_group = [
            dist.new_group(list(range(half))),
            dist.new_group(list(range(half, world))),
        ][rank // half]
        dp_group = [dist.new_group([r, r + half]) for r in range(half)][rank % half]
        x = torch.ones(4, self.N, device=device)

        def run(comm_hook: bool) -> list[torch.Tensor]:
            """Train a few steps, reducing either via the hook or blocking."""
            model = self._model(device)
            net: torch.nn.Module = model
            if comm_hook:
                net = DistributedDataParallel(
                    model,
                    device_ids=self._device_ids(device),
                    process_group=dp_group,
                )
                net.register_comm_hook(
                    None,
                    hooks.make_mp_dp_comm_hook(lambda p: mp_group, dp_group=dp_group),
                )
            opt = torch.optim.SGD(model.parameters(), lr=1e-2)
            for _ in range(self.STEPS):
                opt.zero_grad(set_to_none=True)
                loss = (net(x) * float(rank + 1)).sum() / (self.N * self.LAYERS)
                loss.backward()
                if not comm_hook:
                    # blocking two-stage reference: model-parallel, then data-parallel
                    for param in model.parameters():
                        dist.all_reduce(param.grad, group=mp_group)
                        param.grad.div_(dp_group.size())
                        dist.all_reduce(param.grad, group=dp_group)
                opt.step()
            return [param.detach().clone() for param in model.parameters()]

        got, want = run(comm_hook=True), run(comm_hook=False)
        for have, expected in zip(got, want):
            assert torch.equal(have, expected)

        # The weights are replicated over both groups, so every rank must agree.
        for tensor in got:
            gathered = [torch.empty_like(tensor) for _ in range(world)]
            dist.all_gather(gathered, tensor)
            for other in gathered:
                assert torch.equal(other, gathered[0])

    def test_without_model_parallel_group_averages_over_data_parallel(self, device):
        # ``mp_group_for`` returning None must leave plain DDP averaging.
        world, rank = dist.get_world_size(), dist.get_rank()
        model = torch.nn.Linear(4, 4, bias=False).to(device)
        net = DistributedDataParallel(model, device_ids=self._device_ids(device))
        net.register_comm_hook(None, hooks.make_mp_dp_comm_hook(lambda p: None))
        x = torch.ones(1, 4, device=device)
        (net(x) * float(rank + 1)).sum().backward()
        expected = (world + 1) / 2
        assert torch.allclose(
            model.weight.grad.cpu(), torch.full((4, 4), float(expected))
        )
