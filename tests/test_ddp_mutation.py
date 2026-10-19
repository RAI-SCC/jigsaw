"""The DDP tests notice when the replica sync is missing or misscaled.

``JIGSAW_DISABLE_GRAD_SYNC=1`` (``jigsaw.parallel._sync._SYNC_DISABLED``) turns
:func:`jigsaw.parallel.sync_grad` into a pass-through: replicated parameters keep
their local gradient share. Here the switch is flipped in-process and the
gradient check (``test_ddp``), the multi-step identity check
(``test_ddp_steps``) and the serial equivalence (``test_ddp_serial``) must each
fail. A pure data-parallel mesh replicates nothing, so it must still pass.

The standing diagnostic: at a fixed global batch the gradient norm does not
depend on how many tiles an image is cut into. An unsummed gradient falls off
with the number of tiles instead.
"""

import pytest
import torch
import torch.distributed as dist

import jigsaw
from jigsaw import parallel
from tests import test_ddp_serial
from tests.parity import all_ranks_ok
from tests.test_ddp import (
    KINDS,
    MESHES,
    _build,
    _disable_sync,
    _expected,
    _local_grads,
    _mesh,
    _step,
    _wrap,
)
from tests.test_ddp_steps import train
from tests.test_examples import _load, _tile


@pytest.fixture
def sync_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_sync(monkeypatch)


def _cases(shapes):
    for shape in shapes:
        for kind in KINDS:
            yield pytest.param(
                shape,
                kind,
                id=f"{'x'.join(map(str, shape))}-{kind}",
                marks=pytest.mark.dist(shape[0] * shape[1] * shape[2]),
            )


MODEL_PARALLEL = [s for s in MESHES if s[1] * s[2] > 1]
DATA_PARALLEL = [s for s in MESHES if s[1] * s[2] == 1]


@pytest.mark.parametrize("shape, kind", list(_cases(MODEL_PARALLEL)))
def test_gradient_check_fails(shape, kind, device, monkeypatch) -> None:
    """Layer 2: some replicated gradient is not ``AVG_dp(SUM_replica(local))``."""
    mesh = _mesh(shape, device)
    model, make_input = _build(kind, mesh, device)
    local = _local_grads(model, make_input, 1, monkeypatch)
    expected = _expected(model, local, mesh["dp"], ddp=True)
    _disable_sync(monkeypatch)
    _step(_wrap(model, "ddp", mesh["dp"]), make_input, 1)
    ok = all(
        torch.allclose(p.grad, e, rtol=1e-5, atol=1e-6)
        for p, e in zip(model.parameters(), expected)
    )
    assert not all_ranks_ok(ok, device, dist.group.WORLD)


@pytest.mark.parametrize("shape, kind", list(_cases(MODEL_PARALLEL)))
def test_replica_identity_fails(shape, kind, device, sync_disabled) -> None:
    """Layer 3: the replicas diverge on the first step."""
    with pytest.raises(AssertionError, match="step 0|differ"):
        train(shape, kind, device, steps=2)


@pytest.mark.parametrize("shape, kind", list(_cases(DATA_PARALLEL)))
def test_replica_identity_unaffected_without_replicas(
    shape, kind, device, sync_disabled
) -> None:
    """Control: nothing is replicated on a pure dp mesh, so nothing changes."""
    train(shape, kind, device, steps=2)


@pytest.mark.dist(8)
def test_serial_equivalence_fails(device, sync_disabled) -> None:
    """Layer 4: without the sync, training drifts from the serial run."""
    with pytest.raises(AssertionError, match="serial reference"):
        test_ddp_serial.compare("distributed_mse", device)


@pytest.mark.dist(8)
def test_serial_equivalence_catches_unscaled_loss(device) -> None:
    """Layer 4: the local mean loss, not divided by the number of tiles, is wrong."""
    with pytest.raises(AssertionError, match="serial reference"):
        test_ddp_serial.compare("local_mse", device)


# (dp, sh, sw) on 4 ranks: one image cut into 1, 2 and 4 tiles
TILINGS = [(4, 1, 1), (2, 2, 1), (1, 2, 2)]
_TILING_MESHES: dict = {}


def _grad_norm(tiling: tuple[int, int, int], state: dict, device) -> float:
    """Global gradient norm of the domain-parallel tutorial at a fixed global batch."""
    dp_size, sh, sw = tiling
    key = (tiling, device.type)
    if key not in _TILING_MESHES:
        _TILING_MESHES[key] = jigsaw.make_process_mesh(
            {"dp": dp_size, "sh": sh, "sw": sw, "col": 1},
            device_type=device.type,
            model_parallel_axes=("sh", "sw", "col"),
        )
    mesh = _TILING_MESHES[key]
    model = _load("domain_parallel").build_model(mesh, device)
    model.load_state_dict(state)
    ddp = parallel.wrap_ddp(model, mesh)

    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 1, 64, 64, generator=g).to(device)
    y = torch.randn(4, 1, 64, 64, generator=g).to(device)
    per_replica = 4 // dp_size
    mine = slice(mesh["dp"].rank() * per_replica, (mesh["dp"].rank() + 1) * per_replica)
    tiles = [(-2, "sh"), (-1, "sw")]
    x, y = _tile(x[mine], mesh, tiles), _tile(y[mine], mesh, tiles)
    loss_fn = jigsaw.modules.DistributedMSELoss(reduce_groups=(mesh["sh"], mesh["sw"]))
    loss_fn(ddp(x), y).backward()
    # Every weight is replicated, so the local norm is the global one.
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()]).norm().item()


def _norms(device) -> list[float]:
    # The weights are replicated, so the undivided model's state fits every tiling.
    torch.manual_seed(0)
    model = _load("domain_parallel").build_model(
        test_ddp_serial._mesh({"dp": 4, "sh": 1, "sw": 1, "col": 1}, device), device
    )
    state = {k: v.clone() for k, v in model.state_dict().items()}
    return [_grad_norm(t, state, device) for t in TILINGS]


@pytest.mark.dist(4)
def test_grad_norm_invariant_to_tiling(device) -> None:
    norms = _norms(device)
    ok = all(abs(n - norms[0]) <= 1e-5 * norms[0] for n in norms)
    assert all_ranks_ok(ok, device, dist.group.WORLD), f"norm per tiling {norms}"


@pytest.mark.dist(4)
def test_grad_norm_falls_off_without_sync(device, sync_disabled) -> None:
    norms = _norms(device)
    ok = all(n < 0.9 * norms[0] for n in norms[1:])
    assert all_ranks_ok(ok, device, dist.group.WORLD), f"norm per tiling {norms}"
