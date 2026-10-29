"""Tests for the communication primitives in jigsaw._core.comm.

Gloo's ``all_to_all`` needs equal-sized tensors on every rank, which the zero-length
placeholders used by the all-to-all exchange violate, so that exchange is
checked against a recording fake of ``dist.all_to_all`` (it is exercised for real
on NCCL). The p2p and broadcast helpers run for real on two ranks, on the session
device.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from jigsaw._core import comm


class FakeGroup:
    def __init__(self, rank: int, size: int):
        self._rank, self._size = rank, size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


@pytest.fixture
def a2a_calls(monkeypatch):
    """Replace ``dist.all_to_all`` with a recorder returning a sentinel handle."""
    calls = []

    def fake(output, inputs, group=None, async_op=False):
        calls.append(SimpleNamespace(out=output, inp=inputs, group=group, op=async_op))
        return "handle"

    monkeypatch.setattr(dist, "all_to_all", fake)
    return calls


class TestGetEmpty:
    def test_is_zero_length_with_requested_dtype(self):
        empty = comm.get_empty(torch.device("cpu"), torch.float16)
        assert empty.numel() == 0 and empty.dtype == torch.float16

    def test_is_cached_per_device_dtype(self):
        a = comm.get_empty(torch.device("cpu"), torch.float32)
        assert a is comm.get_empty(torch.device("cpu"), torch.float32)
        assert a is not comm.get_empty(torch.device("cpu"), torch.float64)


class TestAllToAllAsync:
    def test_forwards_arguments_and_returns_work(self, a2a_calls):
        out, inp, group = [torch.empty(1)], [torch.ones(1)], FakeGroup(0, 1)
        assert comm.all_to_all_async(out, inp, group=group) == "handle"
        (call,) = a2a_calls
        assert call.out is out and call.inp is inp
        assert call.group is group and call.op is True

    def test_rejects_missing_work_handle(self, monkeypatch):
        monkeypatch.setattr(dist, "all_to_all", lambda *a, **k: None)
        with pytest.raises(AssertionError):
            comm.all_to_all_async([], [])


class TestOneDAllToAll:
    def test_requires_group(self):
        with pytest.raises(ValueError, match="group is required"):
            comm.comm_1d_alltoall(torch.ones(1), torch.zeros(1), direction=1)

    def test_requires_direction_or_ranks(self):
        with pytest.raises(ValueError, match="direction is required"):
            comm.comm_1d_alltoall(torch.ones(1), torch.zeros(1), group=FakeGroup(0, 4))

    @pytest.mark.parametrize("direction", [1, -1, 2])
    def test_direction_picks_ring_neighbours(self, a2a_calls, direction):
        data, buffer = torch.ones(1), torch.zeros(1)
        out, handle = comm.comm_1d_alltoall(
            data, buffer, direction=direction, group=FakeGroup(1, 4)
        )
        (call,) = a2a_calls
        send_to, recv_from = (1 + direction) % 4, (1 - direction) % 4
        assert call.inp[send_to] is data and call.out[recv_from] is buffer
        assert sum(t.numel() > 0 for t in call.inp) == 1
        assert out is buffer and handle == "handle"

    def test_explicit_ranks_override_direction(self, a2a_calls):
        data, buffer = torch.ones(1), torch.zeros(1)
        comm.comm_1d_alltoall(
            data, buffer, send_recv_ranks=[3, 0], group=FakeGroup(1, 4), async_op=False
        )
        (call,) = a2a_calls
        assert call.inp[3] is data and call.out[0] is buffer and call.op is False

    def test_group_size_override_wraps_ring(self, a2a_calls):
        data, buffer = torch.ones(1), torch.zeros(1)
        comm.comm_1d_alltoall(
            data, buffer, direction=1, group=FakeGroup(2, 4), group_size=3
        )
        (call,) = a2a_calls
        assert call.inp[0] is data  # (2 + 1) % 3


P2P_1D_FNS = [comm.comm_1d_isend, comm.comm_1d_batch_isend_irecv]


@pytest.mark.parametrize("fn", P2P_1D_FNS)
class TestOneDP2PValidation:
    def test_requires_group(self, fn):
        with pytest.raises(ValueError, match="group is required"):
            fn(torch.ones(1), torch.zeros(1), direction=1)

    def test_requires_direction_or_ranks(self, fn):
        with pytest.raises(ValueError, match="direction is required"):
            fn(torch.ones(1), torch.zeros(1), group=FakeGroup(0, 2))


class TestResolveCommOp:
    cpu, cuda = torch.device("cpu"), torch.device("cuda")

    def test_auto_picks_by_device(self):
        assert comm.resolve_comm_op("auto", self.cuda) is comm.CommOp.ALL_TO_ALL
        assert comm.resolve_comm_op("auto", self.cpu) is comm.CommOp.BATCH_ISEND_IRECV

    @pytest.mark.parametrize("value", ["isend", "ISEND", comm.CommOp.ISEND])
    def test_strings_in_any_case_and_members(self, value):
        assert comm.resolve_comm_op(value, self.cpu) is comm.CommOp.ISEND

    def test_all_to_all_off_cuda_is_rejected(self):
        with pytest.raises(ValueError, match="needs CUDA"):
            comm.resolve_comm_op("all_to_all", self.cpu)

    def test_unknown_name_is_rejected(self):
        with pytest.raises(ValueError, match="comm must be one of"):
            comm.resolve_comm_op("p2p", self.cpu)

    def test_unknown_device_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported device meta"):
            comm.resolve_comm_op("auto", torch.device("meta"))

    @pytest.mark.parametrize(
        "op, fn",
        [
            (comm.CommOp.ALL_TO_ALL, comm.comm_1d_alltoall),
            (comm.CommOp.ISEND, comm.comm_1d_isend),
            (comm.CommOp.BATCH_ISEND_IRECV, comm.comm_1d_batch_isend_irecv),
        ],
    )
    def test_select_comm_1d(self, op, fn):
        assert comm.select_comm_1d(op) is fn


class TestBatchWork:
    class _Work:
        def __init__(self, done: bool):
            self.done, self.waited = done, []

        def wait(self, *args, **kwargs):
            self.waited.append((args, kwargs))
            self.done = True

        def is_completed(self) -> bool:
            return self.done

    def test_wait_waits_on_every_work_and_forwards_arguments(self):
        works = [self._Work(False), self._Work(False)]
        assert comm._BatchWork(works).wait(timeout=3) is True
        assert all(w.waited == [((), {"timeout": 3})] for w in works)

    def test_is_completed_needs_every_work_to_be_done(self):
        works = [self._Work(True), self._Work(False)]
        batch = comm._BatchWork(works)
        assert not batch.is_completed()
        batch.wait()
        assert batch.is_completed()


@pytest.mark.dist(2)
class TestTwoRanks:
    @pytest.fixture
    def group(self):
        return dist.group.WORLD

    @pytest.mark.parametrize("fn", P2P_1D_FNS)
    @pytest.mark.parametrize("async_op", [True, False])
    @pytest.mark.parametrize("direction", [1, -1])
    def test_1d_p2p_ring_shift(self, device, group, fn, async_op, direction):
        rank = dist.get_rank()
        data = torch.full((2,), float(rank), device=device)
        buffer = torch.zeros(2, device=device)
        out, handle = fn(
            data, buffer, direction=direction, group=group, async_op=async_op
        )
        assert (handle is None) == (not async_op)
        if handle is not None:
            handle.wait()
        assert out is buffer
        assert torch.equal(
            buffer.cpu(), torch.full((2,), float((rank - direction) % 2))
        )

    @pytest.mark.parametrize("fn", P2P_1D_FNS)
    @pytest.mark.parametrize("async_op", [True, False])
    def test_1d_p2p_sends_a_strided_view(self, device, group, fn, async_op):
        rank = dist.get_rank()
        data = torch.arange(6.0, device=device).reshape(2, 3).add(10 * rank)[:, 1]
        assert not data.is_contiguous()
        buffer = torch.zeros(2, device=device)
        _, handle = fn(data, buffer, direction=1, group=group, async_op=async_op)
        if handle is not None:
            handle.wait()
        expected = torch.tensor([1.0, 4.0]) + 10 * (1 - rank)
        assert torch.equal(buffer.cpu(), expected)

    @pytest.mark.parametrize("fn", P2P_1D_FNS)
    def test_1d_p2p_explicit_ranks(self, device, group, fn):
        rank = dist.get_rank()
        buffer = torch.zeros(1, device=device)
        fn(
            torch.tensor([float(rank)], device=device),
            buffer,
            send_recv_ranks=[1 - rank, 1 - rank],
            group=group,
            async_op=False,
        )
        assert buffer.item() == float(1 - rank)

    @pytest.mark.parametrize("src", [0, 1])
    @pytest.mark.parametrize("async_op", [True, False])
    def test_broadcast(self, device, group, src, async_op):
        data = torch.full((4,), 5.0) if dist.get_rank() == src else torch.zeros(4)
        data = data.to(device)
        out, handle = comm.comm_broadcast(data, src, group, async_op=async_op)
        if handle is not None:
            handle.wait()
        assert out is data
        assert torch.equal(data.cpu(), torch.full((4,), 5.0))
