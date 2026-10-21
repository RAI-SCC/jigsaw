"""Unit tests for :func:`jigsaw.parallel.clip_grad_norm_`.

The 4-rank tests split three parameters over a 2 x 2 mesh: one sharded over all
ranks (rank ``r`` holds ``(r + 1) * [1, 2, 3]``), one replicated over ``a``
(its copy depends only on the rank's ``b`` coordinate), and one replicated over
both axes. The expected norm treats them as one vector with every shard and
every replica counted once.
"""

import math

import pytest
import torch
import torch.distributed as dist

from jigsaw import parallel
from jigsaw.mesh import make_process_mesh

NORM_TYPES = [1.0, 2.0, 3.0, math.inf]

_MESH_CACHE: dict = {}


def _mesh_2x2(device: torch.device) -> dict:
    # Every mesh opens new communicators and nothing frees them; build it once.
    if device.type not in _MESH_CACHE:
        _MESH_CACHE[device.type] = make_process_mesh(
            {"a": 2, "b": 2}, device_type=device.type
        )
    return _MESH_CACHE[device.type]


def _with_grad(grad: torch.Tensor) -> torch.nn.Parameter:
    p = torch.nn.Parameter(torch.zeros_like(grad))
    p.grad = grad.clone()
    return p


def _mixed_params(mesh: dict, device: torch.device) -> list[torch.nn.Parameter]:
    """Return the sharded, ``a``-replicated and fully replicated parameters."""
    r = dist.get_rank()
    b = dist.get_rank(mesh["b"])
    dt = torch.float64
    sharded = _with_grad((r + 1) * torch.tensor([1.0, 2.0, 3.0], dtype=dt, device=device))
    rep_a = _with_grad((b + 1) * torch.ones(2, dtype=dt, device=device))
    rep_ab = _with_grad(torch.tensor([5.0], dtype=dt, device=device))
    parallel.mark_replicated(rep_a, mesh["a"])
    parallel.mark_replicated(rep_ab, [mesh["a"], mesh["b"]])
    return [sharded, rep_a, rep_ab]


def _expected_norm(p: float) -> float:
    """Return the norm of the mixed parameters' gradients as one vector."""
    if math.isinf(p):
        return 12.0  # 4 * 3, the largest shard entry
    sharded = sum((r + 1) ** p for r in range(4)) * (1 + 2**p + 3**p)
    rep_a = sum((b + 1) ** p for b in range(2)) * 2
    return (sharded + rep_a + 5.0**p) ** (1 / p)


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_matches_torch_on_replicas(norm_type: float, device: torch.device) -> None:
    """Parameters replicated over the whole group clip like torch on one rank."""
    g = torch.Generator().manual_seed(0)
    grads = [torch.randn(5, 3, generator=g), torch.randn(7, generator=g)]
    ours = [_with_grad(x.to(device)) for x in grads]
    ref = [_with_grad(x.to(device)) for x in grads]
    for p in ours:
        parallel.mark_replicated(p, dist.group.WORLD)

    total = parallel.clip_grad_norm_(ours, 0.5, dist.group.WORLD, norm_type)
    expected = torch.nn.utils.clip_grad_norm_(ref, 0.5, norm_type)

    torch.testing.assert_close(total, expected)
    for p, q in zip(ours, ref):
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.dist(4)
@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_counts_shards_and_replicas_once(
    norm_type: float, device: torch.device
) -> None:
    mesh = _mesh_2x2(device)
    params = _mixed_params(mesh, device)
    before = [p.grad.clone() for p in params]
    expected = _expected_norm(norm_type)

    total = parallel.clip_grad_norm_(params, expected / 2, mesh["model"], norm_type)

    assert total.item() == pytest.approx(expected, rel=1e-12)
    coef = (expected / 2) / (expected + 1e-6)
    for p, g in zip(params, before):
        torch.testing.assert_close(p.grad, g * coef)


@pytest.mark.dist(4)
def test_no_clipping_below_max_norm(device: torch.device) -> None:
    mesh = _mesh_2x2(device)
    params = _mixed_params(mesh, device)
    before = [p.grad.clone() for p in params]

    parallel.clip_grad_norm_(params, 2 * _expected_norm(2.0), mesh["model"])

    for p, g in zip(params, before):
        torch.testing.assert_close(p.grad, g)


def test_rank_without_gradients(device: torch.device) -> None:
    """A rank without gradients still joins the collective and gets the norm."""
    p = torch.nn.Parameter(torch.zeros(2, device=device))
    if dist.get_rank() != 0:
        p.grad = torch.full((2,), 3.0, device=device)

    total = parallel.clip_grad_norm_([p], 1.0, dist.group.WORLD)

    world = dist.get_world_size()
    expected = math.sqrt(18 * (world - 1))
    assert total.item() == pytest.approx(expected, rel=1e-6)


@pytest.mark.dist(4)
def test_replica_group_outside_clipping_group_raises(device: torch.device) -> None:
    mesh = _mesh_2x2(device)
    p = _with_grad(torch.ones(2, device=device))
    parallel.mark_replicated(p, mesh["a"])
    with pytest.raises(ValueError, match="model-parallel group"):
        parallel.clip_grad_norm_([p], 1.0, mesh["b"])


def test_error_if_nonfinite(device: torch.device) -> None:
    p = _with_grad(torch.tensor([1.0, math.inf], device=device))
    with pytest.raises(RuntimeError, match="non-finite"):
        parallel.clip_grad_norm_([p], 1.0, dist.group.WORLD, error_if_nonfinite=True)
