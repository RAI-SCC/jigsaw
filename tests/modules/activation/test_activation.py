import pytest
import torch
import torch.distributed as dist

from jigsaw.modules.activation import _activation as act
from jigsaw.modules.activation import _functional
from tests.parity import all_ranks_ok, assert_close

pytestmark = [pytest.mark.dist_required]


class TestDistributedSoftmax:
    @pytest.mark.parametrize("dim", [-1, 0])
    def test_no_group_runs_locally(self, device, monkeypatch, dim):
        """Without a group the softmax is local: no all-reduce, matches torch."""
        reduced = []
        real_all_reduce = dist.all_reduce

        def recording_all_reduce(*args, **kwargs):
            reduced.append(args)
            return real_all_reduce(*args, **kwargs)

        monkeypatch.setattr(dist, "all_reduce", recording_all_reduce)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4, 6, generator=g).to(device).requires_grad_(True)
        x_ref = x.detach().clone().requires_grad_(True)
        grad = torch.randn(4, 6, generator=g).to(device)

        out = _functional._Softmax.apply(x, dim, None)
        out.backward(grad)
        no_reduce = not reduced
        monkeypatch.undo()

        ref = torch.softmax(x_ref, dim=dim)
        ref.backward(grad)
        group = dist.group.WORLD
        assert all_ranks_ok(no_reduce, device, group)
        assert_close("forward", out, ref, 1e-6, 1e-6, device, group)
        assert_close("input grad", x.grad, x_ref.grad, 1e-6, 1e-6, device, group)

    @pytest.mark.parametrize("dim", [-1, 1, 0, -2])
    def test_distributed_softmax(self, device: torch.device, dim: int) -> None:
        """Forward and backward match ``torch.softmax`` along any sharded dimension."""
        group = dist.group.WORLD
        size, rank = group.size(), group.rank()

        # The same seed on every rank gives every rank the same full tensors.
        g = torch.Generator().manual_seed(0)
        shape = [2 * size, 2 * size, 2 * size]
        x_full = torch.randn(shape, generator=g).to(device).requires_grad_(True)
        grad_full = torch.randn(shape, generator=g).to(device)
        x = x_full.detach().chunk(size, dim=dim)[rank].clone().requires_grad_(True)

        softmax = act.DistributedSoftmax(group, dim=dim)
        out = softmax(x)
        out.backward(grad_full.chunk(size, dim=dim)[rank])

        ref = torch.softmax(x_full, dim=dim)
        ref.backward(grad_full)

        assert isinstance(x.grad, torch.Tensor)
        assert isinstance(x_full.grad, torch.Tensor)
        assert_close(
            "forward", out, ref.chunk(size, dim=dim)[rank], 1e-5, 1e-5, device, group
        )
        assert_close(
            "input grad",
            x.grad,
            x_full.grad.chunk(size, dim=dim)[rank],
            1e-5,
            1e-5,
            device,
            group,
        )
