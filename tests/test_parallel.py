"""Unit tests for the in-graph gradient sync in :mod:`jigsaw.parallel`.

Rank ``r`` contributes the gradient ``(r + 1) * [1, 2, 3]``, so a SUM over a
group of global ranks ``R`` is ``sum(r + 1 for r in R) * [1, 2, 3]`` -- exact in
floating point, and different from any single rank's share.
"""

import pytest
import torch
import torch.distributed as dist

from jigsaw import parallel
from jigsaw.mesh import make_process_mesh

BASE = torch.arange(1.0, 4.0)


def _scale() -> float:
    return float(dist.get_rank() + 1)


def _world_sum() -> float:
    world = dist.get_world_size()
    return world * (world + 1) / 2


def _param(device: torch.device, dtype: torch.dtype = torch.float32):
    return torch.nn.Parameter(torch.zeros(3, device=device, dtype=dtype))


_MESH_CACHE: dict = {}


def _mesh_2x2(device: torch.device) -> dict:
    # Every mesh opens new communicators and nothing frees them, so the
    # 4-rank GPU suite runs out of memory if each test builds its own.
    if device.type not in _MESH_CACHE:
        _MESH_CACHE[device.type] = make_process_mesh(
            {"a": 2, "b": 2}, device_type=device.type
        )
    return _MESH_CACHE[device.type]


def _loss(t: torch.Tensor) -> torch.Tensor:
    # d loss / d t == (rank + 1) * [1, 2, 3]
    return (t * BASE.to(t.device, t.dtype) * _scale()).sum()


@pytest.fixture
def count_all_reduce(monkeypatch):
    """Count ``dist.all_reduce`` calls."""
    calls = []
    orig = dist.all_reduce

    def counting(tensor, *args, **kwargs):
        calls.append(kwargs.get("group"))
        return orig(tensor, *args, **kwargs)

    monkeypatch.setattr(dist, "all_reduce", counting)
    return calls


class TestMark:
    def test_double_mark_raises(self, device):
        if dist.get_world_size() == 1:
            pytest.skip("a size-1 group is never marked")
        p = _param(device)
        parallel.mark_replicated(p, dist.group.WORLD)
        with pytest.raises(RuntimeError, match="already marked"):
            parallel.mark_replicated(p, dist.group.WORLD)

    @pytest.mark.dist(4)
    def test_extend_appends_new_groups(self, device):
        mesh = _mesh_2x2(device)
        p = _param(device)
        parallel.mark_replicated(p, mesh["a"])
        parallel.mark_replicated(p, mesh["b"], extend=True)
        assert parallel.replica_groups(p) == (mesh["a"], mesh["b"])
        (q,) = parallel.sync_grad(p)
        _loss(q).backward()
        assert torch.equal(p.grad.cpu(), BASE * _world_sum())

    def test_extend_drops_groups_with_the_same_ranks(self, device):
        if dist.get_world_size() == 1:
            pytest.skip("a size-1 group is never marked")
        p = _param(device)
        parallel.mark_replicated(p, dist.group.WORLD)
        same = dist.new_group(list(range(dist.get_world_size())))
        parallel.mark_replicated(p, [same, dist.group.WORLD], extend=True)
        assert parallel.replica_groups(p) == (dist.group.WORLD,)
        (q,) = parallel.sync_grad(p)
        _loss(q).backward()
        assert torch.equal(p.grad.cpu(), BASE * _world_sum())

    def test_extend_marks_an_unmarked_parameter(self, device):
        if dist.get_world_size() == 1:
            pytest.skip("a size-1 group is never marked")
        p = _param(device)
        parallel.mark_replicated(p, dist.group.WORLD, extend=True)
        assert parallel.replica_groups(p) == (dist.group.WORLD,)

    def test_size_one_groups_are_dropped(self, device):
        p = _param(device)
        parallel.mark_replicated(p, [None, dist.new_group([dist.get_rank()])])
        assert parallel.replica_groups(p) == ()

    def test_unmarked_has_no_groups(self, device):
        assert parallel.replica_groups(_param(device)) == ()
        assert parallel.replica_groups(None) == ()


class TestSyncGrad:
    def test_single_group_sums(self, device):
        p = _param(device)
        parallel.mark_replicated(p, dist.group.WORLD)
        (q,) = parallel.sync_grad(p)
        _loss(q).backward()
        assert torch.equal(p.grad.cpu(), BASE * _world_sum())

    @pytest.mark.dist(4)
    def test_group_sequence_sums_over_product(self, device):
        mesh = _mesh_2x2(device)
        p = _param(device)
        parallel.mark_replicated(p, [mesh["a"], mesh["b"]])
        (q,) = parallel.sync_grad(p)
        _loss(q).backward()
        assert torch.equal(p.grad.cpu(), BASE * _world_sum())

    @pytest.mark.dist(4)
    def test_sums_only_over_the_marked_group(self, device):
        mesh = _mesh_2x2(device)
        p = _param(device)
        parallel.mark_replicated(p, mesh["b"])
        (q,) = parallel.sync_grad(p)
        _loss(q).backward()
        members = dist.get_process_group_ranks(mesh["b"])
        assert torch.equal(p.grad.cpu(), BASE * sum(r + 1 for r in members))

    def test_unmarked_none_and_no_grad_pass_through(self, device, count_all_reduce):
        marked, plain = _param(device), _param(device)
        parallel.mark_replicated(marked, dist.group.WORLD)
        frozen = _param(device).requires_grad_(False)
        parallel.mark_replicated(frozen, dist.group.WORLD)

        out = parallel.sync_grad(plain, None, frozen)
        assert out[0] is plain and out[1] is None and out[2] is frozen

        with torch.no_grad():
            (q,) = parallel.sync_grad(marked)
        assert q is marked
        assert count_all_reduce == []

    def test_coalesces_one_collective_per_group_and_dtype(
        self, device, count_all_reduce
    ):
        if dist.get_world_size() == 1:
            pytest.skip("a size-1 group is never marked")
        a, b = _param(device), _param(device)
        c = _param(device, torch.float64)
        for p in (a, b, c):
            parallel.mark_replicated(p, dist.group.WORLD)
        qa, qb, qc = parallel.sync_grad(a, b, c)
        (_loss(qa) + _loss(qb) + _loss(qc)).backward()
        # one flat all-reduce for (a, b), one for c
        assert len(count_all_reduce) == 2
        for p in (a, b, c):
            assert torch.equal(p.grad.cpu().float(), BASE * _world_sum())

    def test_parameter_used_twice(self, device):
        # Two sync nodes on one parameter: S(g_a) + S(g_b) == S(g_a + g_b).
        p = _param(device)
        parallel.mark_replicated(p, dist.group.WORLD)
        (q1,) = parallel.sync_grad(p)
        (q2,) = parallel.sync_grad(p)
        (_loss(q1) + 2 * _loss(q2)).backward()
        assert torch.equal(p.grad.cpu(), 3 * BASE * _world_sum())

    def test_unused_output_gets_zero_gradient(self, device):
        # The zero keeps the flat buffer the same size on every rank.
        a, b = _param(device), _param(device)
        for p in (a, b):
            parallel.mark_replicated(p, dist.group.WORLD)
        qa, _ = parallel.sync_grad(a, b)
        _loss(qa).backward()
        assert torch.equal(a.grad.cpu(), BASE * _world_sum())
        if dist.get_world_size() > 1:
            assert torch.equal(b.grad.cpu(), torch.zeros(3))

    def test_vmap_over_the_input(self, device):
        p = _param(device)
        parallel.mark_replicated(p, dist.group.WORLD)
        x = torch.ones(4, 3, device=device)

        def f(xi):
            (q,) = parallel.sync_grad(p)
            return _loss(q * xi)

        torch.vmap(f)(x).sum().backward()
        assert torch.equal(p.grad.cpu(), 4 * BASE * _world_sum())


class TestGradientAccumulation:
    """Each microbatch is reduced on its own, never the running ``p.grad`` sum."""

    @pytest.mark.parametrize("microbatches", [1, 2, 3])
    def test_n_microbatches_give_n_times_one(self, device, microbatches):
        p = _param(device)
        parallel.mark_replicated(p, dist.group.WORLD)
        for _ in range(microbatches):
            (q,) = parallel.sync_grad(p)
            _loss(q).backward()
        assert torch.equal(p.grad.cpu(), microbatches * BASE * _world_sum())


class TestSyncedParameters:
    def test_only_marked_parameters_are_returned(self, device):
        lin = torch.nn.Linear(3, 3, device=device)
        parallel.mark_replicated(lin.weight, dist.group.WORLD)
        synced = parallel.synced_parameters(lin)
        if dist.get_world_size() == 1:
            assert synced == {}
        else:
            assert set(synced) == {"weight"}
            assert synced["weight"] is not lin.weight

    def test_functional_call_sums_foreign_module(self, device):
        lin = torch.nn.Linear(3, 1, bias=False, device=device)
        parallel.mark_replicated(lin.weight, dist.group.WORLD)
        x = BASE.to(device)[None] * _scale()
        torch.func.functional_call(
            lin, parallel.synced_parameters(lin), (x,)
        ).sum().backward()
        assert torch.equal(lin.weight.grad.cpu(), BASE[None] * _world_sum())


class TestSyncedCall:
    def test_sums_marked_and_keeps_unmarked(self, device):
        lin = torch.nn.Linear(3, 1, device=device)
        lin.bias.data.zero_()
        parallel.mark_replicated(lin.weight, dist.group.WORLD)
        x = BASE.to(device)[None] * _scale()
        parallel.synced_call(lin, x).sum().backward()
        assert torch.equal(lin.weight.grad.cpu(), BASE[None] * _world_sum())
        assert torch.equal(lin.bias.grad.cpu(), torch.ones(1))

    def test_passes_keyword_inputs(self, device):
        attn = torch.nn.MultiheadAttention(3, 1, bias=False, device=device)
        parallel.mark_replicated(attn.in_proj_weight, dist.group.WORLD)
        x = BASE.to(device)[None, None]
        out, weights = parallel.synced_call(attn, x, x, x, need_weights=False)
        assert out.shape == (1, 1, 3)
        assert weights is None
