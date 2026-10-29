"""Unit tests for :func:`jigsaw.parallel.clip_grad_norm_`.

The 4-rank tests split three parameters over a 2 x 2 mesh: one sharded over all
ranks (rank ``r`` holds ``(r + 1) * [1, 2, 3]``), one replicated over ``a``
(its copy depends only on the rank's ``b`` coordinate), and one replicated over
both axes. The expected norm treats them as one vector with every shard and
every replica counted once.

The model tests convert a serial model with every kind of jigsaw parameter on
several meshes and compare against ``torch.nn.utils.clip_grad_norm_`` on the
serial model, which fails if a module leaves a replicated parameter unmarked.
"""

import math

import pytest
import torch
import torch.distributed as dist

from jigsaw import parallel
from jigsaw.convert import convert_to_jigsaw
from jigsaw.modules.attention import Attention
from tests.parity import assert_close
from tests.process_mesh import _get_mesh

NORM_TYPES = [1.0, 2.0, 3.0, math.inf]

# torch.testing.assert_close's defaults per dtype
F32_ATOL, F32_RTOL = 1e-5, 1.3e-6
F64_ATOL, F64_RTOL = 1e-7, 1e-7


def _with_grad(grad: torch.Tensor) -> torch.nn.Parameter:
    p = torch.nn.Parameter(torch.zeros_like(grad))
    p.grad = grad.clone()
    return p


def _mixed_params(mesh: dict, device: torch.device) -> list[torch.nn.Parameter]:
    """Return the sharded, ``a``-replicated and fully replicated parameters."""
    r = dist.get_rank()
    b = dist.get_rank(mesh["b"])
    dt = torch.float64
    sharded = _with_grad(
        (r + 1) * torch.tensor([1.0, 2.0, 3.0], dtype=dt, device=device)
    )
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

    group = dist.group.WORLD
    assert_close("norm", total, expected, F32_ATOL, F32_RTOL, device, group)
    for i, (p, q) in enumerate(zip(ours, ref, strict=True)):
        assert_close(f"grad {i}", p.grad, q.grad, F32_ATOL, F32_RTOL, device, group)


@pytest.mark.dist(4)
@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_counts_shards_and_replicas_once(
    norm_type: float, device: torch.device
) -> None:
    mesh = _get_mesh(device.type, {"a": 2, "b": 2})
    params = _mixed_params(mesh, device)
    before = [p.grad.clone() for p in params]
    expected = _expected_norm(norm_type)

    total = parallel.clip_grad_norm_(params, expected / 2, mesh["model"], norm_type)

    group = mesh["model"]
    ref = torch.tensor(expected, dtype=total.dtype, device=device)
    assert_close("norm", total, ref, 0.0, 1e-12, device, group)
    coef = (expected / 2) / (expected + 1e-6)
    for i, (p, g) in enumerate(zip(params, before, strict=True)):
        assert_close(f"grad {i}", p.grad, g * coef, F64_ATOL, F64_RTOL, device, group)


@pytest.mark.dist(4)
def test_no_clipping_below_max_norm(device: torch.device) -> None:
    mesh = _get_mesh(device.type, {"a": 2, "b": 2})
    params = _mixed_params(mesh, device)
    before = [p.grad.clone() for p in params]

    parallel.clip_grad_norm_(params, 2 * _expected_norm(2.0), mesh["model"])

    for i, (p, g) in enumerate(zip(params, before, strict=True)):
        assert_close(f"grad {i}", p.grad, g, 0.0, 0.0, device, mesh["model"])


def test_rank_without_gradients(device: torch.device) -> None:
    """A rank without gradients still joins the collective and gets the norm."""
    p = torch.nn.Parameter(torch.zeros(2, device=device))
    if dist.get_rank() != 0:
        p.grad = torch.full((2,), 3.0, device=device)

    total = parallel.clip_grad_norm_([p], 1.0, dist.group.WORLD)

    world = dist.get_world_size()
    expected = torch.tensor(
        math.sqrt(18 * (world - 1)), dtype=total.dtype, device=device
    )
    assert_close("norm", total, expected, 0.0, 1e-6, device, dist.group.WORLD)


@pytest.mark.dist(4)
@pytest.mark.parametrize("norm_type", [2.0, math.inf])
def test_replica_group_outside_clipping_group_raises(
    norm_type: float, device: torch.device
) -> None:
    """Every rank raises, also rank 0 without a gradient, so none hangs."""
    mesh = _get_mesh(device.type, {"a": 2, "b": 2})
    p = torch.nn.Parameter(torch.ones(2, device=device))
    if dist.get_rank() != 0:
        p.grad = torch.ones_like(p)
    parallel.mark_replicated(p, mesh["a"])
    with pytest.raises(ValueError, match="model-parallel group"):
        parallel.clip_grad_norm_([p], 1.0, mesh["b"], norm_type)


def test_error_if_nonfinite(device: torch.device) -> None:
    p = _with_grad(torch.tensor([1.0, math.inf], device=device))
    with pytest.raises(RuntimeError, match="non-finite"):
        parallel.clip_grad_norm_([p], 1.0, dist.group.WORLD, error_if_nonfinite=True)


def test_half_precision_norm_does_not_overflow(device: torch.device) -> None:
    """A float16 gradient whose norm exceeds float16's range still clips.

    The norm comes back in float32, so it is finite too.
    """
    # norm 1000 * sqrt(10000) = 1e5 > 65504; each element and the clipped
    # result (1000 / 1e5 = 0.01) are representable
    p = _with_grad(torch.full((10000,), 1000.0, dtype=torch.float16, device=device))
    parallel.mark_replicated(p, dist.group.WORLD)

    total = parallel.clip_grad_norm_([p], 1.0, dist.group.WORLD)

    group = dist.group.WORLD
    assert total.dtype == torch.float32
    ref = torch.tensor(1e5, device=device)
    # float32 accumulation of 10000 squares of 1e6
    assert_close("norm", total, ref, 0.0, 1e-5, device, group)
    expected = torch.full_like(p.grad, 0.01)
    assert_close("grad", p.grad, expected, 0.0, 1e-3, device, group)


def test_norm_dtype_promotes_over_all_parameters(device: torch.device) -> None:
    """The norm's dtype comes from every parameter, also those without grads.

    Rank 0's float64 parameter has no gradient, yet every rank returns float64.
    """
    half = _with_grad(torch.ones(4, dtype=torch.bfloat16, device=device))
    wide = torch.nn.Parameter(torch.ones(2, dtype=torch.float64, device=device))
    if dist.get_rank() != 0:
        wide.grad = torch.ones_like(wide)
    for p in (half, wide):
        parallel.mark_replicated(p, dist.group.WORLD)

    total = parallel.clip_grad_norm_([half, wide], 100.0, dist.group.WORLD)

    assert total.dtype == torch.float64


class _Model(torch.nn.Module):
    """Every kind of jigsaw parameter: sharded, col-sharded, and replicated."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, device=device)
        self.norm = torch.nn.LayerNorm(32, device=device)
        self.attn = Attention(32, 32, 32, 4, device, qk_norm=True)
        self.conv = torch.nn.Conv2d(8, 8, 3, device=device)
        self.group_norm = torch.nn.GroupNorm(4, 8, device=device)


@pytest.mark.dist(4)
@pytest.mark.parametrize("shape", [(2, 2), (4, 1), (1, 4)], ids=str)
@pytest.mark.parametrize("norm_type", [1.0, 2.0, math.inf])
def test_converted_model_matches_serial(
    shape: tuple[int, int], norm_type: float, device: torch.device
) -> None:
    """The norm of a converted model is the serial model's on every mesh.

    Each gradient is set to its parameter's value, so the converter's weight
    shards are also the gradient shards and no forward pass is needed.
    """
    mesh = _get_mesh(device.type, {"row": shape[0], "col": shape[1]})
    torch.manual_seed(0)
    serial = _Model(device)
    with torch.no_grad():
        for p in serial.parameters():
            p.normal_()
            p.grad = p.detach().clone()
    max_norm = 1.0
    expected = torch.nn.utils.clip_grad_norm_(serial.parameters(), max_norm, norm_type)

    # ring attention splits the sequence over rows, so needs more than one
    kwargs = {} if shape[0] > 1 else {"ring": False}
    model = convert_to_jigsaw(serial, mesh, device, **kwargs)
    params = list(model.parameters())
    for p in params:
        p.grad = p.detach().clone()
    total = parallel.clip_grad_norm_(params, max_norm, mesh["model"], norm_type)

    group = mesh["model"]
    assert_close("norm", total, expected, 0.0, 1e-5, device, group)
    coef = min(max_norm / (expected.item() + 1e-6), 1.0)
    for i, p in enumerate(params):
        assert_close(
            f"grad {i}", p.grad, p.detach() * coef, F32_ATOL, 1e-5, device, group
        )
