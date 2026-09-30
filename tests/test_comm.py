"""Tests for the communication primitives in jigsaw.comm.

Gloo's ``all_to_all`` needs equal-sized tensors on every rank, which the zero-length
placeholders used by the all-to-all halo/ring helpers violate, so those helpers are
checked against a recording fake of ``dist.all_to_all`` (they are exercised for real
on NCCL). The p2p and broadcast helpers run for real on two ranks.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from jigsaw import comm


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


class TestResolveMatmulDtype:
    cpu = torch.device("cpu")

    def test_single_operand(self):
        assert comm.resolve_matmul_dtype(self.cpu, torch.zeros(1)) == torch.float32

    def test_promotes_across_operands(self):
        a = torch.zeros(1, dtype=torch.float16)
        b = torch.zeros(1, dtype=torch.float32)
        c = torch.zeros(1, dtype=torch.float64)
        assert comm.resolve_matmul_dtype(self.cpu, a, b) == torch.float32
        assert comm.resolve_matmul_dtype(self.cpu, a, b, c) == torch.float64

    def test_autocast_dtype_wins(self):
        x = torch.zeros(1, dtype=torch.float32)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert comm.resolve_matmul_dtype(self.cpu, x) == torch.bfloat16


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


class TestTwoWayAllToAll:
    def _run(self, rank, async_op=True):
        data, buffer, empty = torch.ones(2), torch.zeros(2), torch.empty(0)
        result = comm.comm_2way_alltoall(
            data, buffer, empty, FakeGroup(rank, 2), async_op=async_op
        )
        return data, buffer, empty, result

    def test_rank0_sends_to_slot_one(self, a2a_calls):
        data, buffer, empty, (out, handles) = self._run(0)
        (call,) = a2a_calls
        assert call.out == [empty, buffer] and call.inp == [empty, data]
        assert out is buffer and handles == ["handle"] and call.op is True

    def test_rank1_sends_to_slot_zero(self, a2a_calls):
        data, buffer, empty, _ = self._run(1, async_op=False)
        (call,) = a2a_calls
        assert call.out == [buffer, empty] and call.inp == [data, empty]
        assert call.op is False

    def test_invalid_rank(self, a2a_calls):
        with pytest.raises(ValueError, match="not valid"):
            self._run(2)


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


class TestOneDP2PValidation:
    def test_requires_group(self):
        with pytest.raises(ValueError, match="group is required"):
            comm.comm_1d_p2p(torch.ones(1), torch.zeros(1), direction=1)

    def test_requires_direction_or_ranks(self):
        with pytest.raises(ValueError, match="direction is required"):
            comm.comm_1d_p2p(torch.ones(1), torch.zeros(1), group=FakeGroup(0, 2))


def test_2way_p2p_invalid_rank():
    with pytest.raises(ValueError, match="not valid"):
        comm.comm_2way_p2p(
            torch.ones(1), torch.zeros(1), torch.empty(0), FakeGroup(2, 3)
        )


@pytest.mark.dist(2)
class TestTwoRanks:
    @pytest.fixture
    def group(self):
        return dist.group.WORLD

    @pytest.mark.parametrize("async_op", [True, False])
    def test_2way_p2p_swaps_data(self, group, async_op):
        rank = dist.get_rank()
        data = torch.full((3,), float(rank + 1))
        buffer = torch.zeros(3)
        out, handles = comm.comm_2way_p2p(
            data, buffer, torch.empty(0), group, async_op=async_op
        )
        assert len(handles) == 2
        for handle in handles:
            if handle is not None:
                handle.wait()
        assert out is buffer
        assert torch.equal(buffer, torch.full((3,), float(2 - rank)))

    @pytest.mark.parametrize("async_op", [True, False])
    @pytest.mark.parametrize("direction", [1, -1])
    def test_1d_p2p_ring_shift(self, group, async_op, direction):
        rank = dist.get_rank()
        data = torch.full((2,), float(rank))
        buffer = torch.zeros(2)
        out, handle = comm.comm_1d_p2p(
            data, buffer, direction=direction, group=group, async_op=async_op
        )
        assert (handle is None) == (not async_op)
        if handle is not None:
            handle.wait()
        assert out is buffer
        assert torch.equal(buffer, torch.full((2,), float((rank - direction) % 2)))

    def test_1d_p2p_explicit_ranks(self, group):
        rank = dist.get_rank()
        buffer = torch.zeros(1)
        comm.comm_1d_p2p(
            torch.tensor([float(rank)]),
            buffer,
            send_recv_ranks=[1 - rank, 1 - rank],
            group=group,
            async_op=False,
        )
        assert buffer.item() == float(1 - rank)

    @pytest.mark.parametrize("src", [0, 1])
    @pytest.mark.parametrize("async_op", [True, False])
    def test_broadcast(self, group, src, async_op):
        data = torch.full((4,), 5.0) if dist.get_rank() == src else torch.zeros(4)
        out, handle = comm.comm_broadcast(data, src, group, async_op=async_op)
        if handle is not None:
            handle.wait()
        assert out is data
        assert torch.equal(data, torch.full((4,), 5.0))
