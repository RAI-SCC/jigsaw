"""The guards against silently wrong DDP setups.

``DistributedDataParallel`` on the wrong process group, or around replicas that
were never made identical, trains without an error and produces a wrong model.
These tests check that each such setup raises -- on every rank, so nothing
hangs -- and that the debug check catches a perturbed replica.
"""

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from jigsaw import debug, hooks, parallel
from jigsaw.modules import convolution
from tests.test_ddp import _build, _expected, _int_tensor, _local_grads, _mesh

# dp=2 over a 2x1 model: replicated conv weights over ``row``
SHAPE = (2, 2, 1)


def _perturb(p: torch.Tensor) -> None:
    with torch.no_grad():
        p.add_(1.0)


def _perturb_on(rank: int, p: torch.Tensor) -> None:
    if dist.get_rank() == rank:
        _perturb(p)


@pytest.mark.dist(4)
@pytest.mark.parametrize("kind", ["xwt", "conv_halo", "attention"])
def test_ddp_on_default_group_raises(kind, device) -> None:
    mesh = _mesh(SHAPE, device)
    model, make_input = _build(kind, mesh, device)
    ddp = DistributedDataParallel(model)  # WORLD: spans the row axis
    with pytest.raises(RuntimeError, match="spans model-parallel ranks"):
        ddp(make_input(0))
    with pytest.raises(RuntimeError, match="spans model-parallel ranks"):
        parallel.check_ddp(ddp)


@pytest.mark.dist(4)
def test_ddp_on_dp_group_passes(device) -> None:
    mesh = _mesh(SHAPE, device)
    model, make_input = _build("conv_halo", mesh, device)
    ddp = DistributedDataParallel(model, process_group=mesh["dp"])
    ddp(make_input(0)).sum().backward()
    assert getattr(ddp, parallel._CHECKED)


@pytest.mark.dist(4)
def test_ddp_on_whole_world_is_fine_without_model_parallelism(device) -> None:
    """On a pure data-parallel mesh the default group *is* the dp axis."""
    mesh = _mesh((4, 1, 1), device)
    model, make_input = _build("conv_halo", mesh, device)
    DistributedDataParallel(model)(make_input(0)).sum().backward()


@pytest.mark.dist(4)
def test_ddp_before_replica_broadcast_raises(device) -> None:
    """DDP only broadcasts over dp; replicas that differ inside a model stay so."""
    mesh = _mesh(SHAPE, device)
    model, make_input = _build("conv_halo", mesh, device)
    _perturb_on(1, model.weight)  # dp 0, row 1
    ddp = DistributedDataParallel(model, process_group=mesh["dp"])
    with pytest.raises(RuntimeError, match="differ across their replica groups"):
        ddp(make_input(0))


@pytest.mark.dist(4)
def test_wrap_ddp_broadcasts_replicas_first(device) -> None:
    mesh = _mesh(SHAPE, device)
    model, make_input = _build("conv_halo", mesh, device)
    _perturb_on(1, model.weight)
    ddp = parallel.wrap_ddp(model, mesh)
    debug.assert_replica_consistency(model, mesh)
    ddp(make_input(0)).sum().backward()


@pytest.mark.dist(4)
def test_wrap_ddp_rejects_process_group(device) -> None:
    mesh = _mesh(SHAPE, device)
    model, _ = _build("conv_halo", mesh, device)
    with pytest.raises(TypeError):
        parallel.wrap_ddp(model, mesh, process_group=mesh["dp"])


@pytest.mark.dist(4)
def test_hook_synced_parameter_warns(device) -> None:
    mesh = _mesh(SHAPE, device)
    model, _ = _build("conv_halo", mesh, device)
    model.bias.register_post_accumulate_grad_hook(
        hooks.make_grad_sync_hook(mesh["row"])
    )
    with pytest.warns(UserWarning, match="make_grad_sync_hook"):
        parallel.wrap_ddp(model, mesh)


@pytest.mark.dist(8)
@pytest.mark.parametrize("axis", ["replica", "dp"])
def test_assert_replica_consistency_fails_on_every_rank(axis, device) -> None:
    mesh = _mesh((2, 2, 2), device)
    model, _ = _build("attention", mesh, device)
    debug.assert_replica_consistency(model, mesh)

    # "replica": one rank of one model replica differs from the rest of it.
    # "dp": a whole model replica differs, consistently, from the other one.
    name, p = next(iter(model.named_parameters()))
    if axis == "replica" and dist.get_rank() == 7:
        _perturb(p)
    if axis == "dp" and mesh["dp"].rank() == 1:
        _perturb(p)
    with pytest.raises(AssertionError, match=name.replace(".", r"\.")):
        debug.assert_replica_consistency(model, mesh)
    with pytest.raises(AssertionError):
        debug.assert_replica_consistency(model, mesh, atol=0.5)
    debug.assert_replica_consistency(model, mesh, atol=1.0)


@pytest.mark.dist(8)
def test_reshard_remarks_onto_new_groups(device, monkeypatch) -> None:
    """After resharding, the gradient is summed over the new spatial group."""
    old, new = _mesh((2, 2, 2), device), _mesh((2, 4, 1), device)
    conv = convolution.DistributedConv2D(
        4,
        4,
        3,
        device=device,
        spatial_group=old["row"],
        channel_group=old["col"],
        padding=(1, 1),
    )
    for i, p in enumerate(conv.parameters()):  # same on every dp replica
        p.data = _int_tensor(p.shape, 1000 + i, device, p.dtype)
    conv.reshard_parameters(old, new)
    for p in conv.parameters():
        assert parallel.replica_groups(p) == (new["row"],)

    def make_input(step: int) -> torch.Tensor:
        return _int_tensor((2, 4, 4, 8 // 4), 10 * dist.get_rank() + step, device)

    local = _local_grads(conv, make_input, 1, monkeypatch)
    expected = _expected(conv, local, new["dp"], ddp=True)
    ddp = parallel.wrap_ddp(conv, new)
    y = ddp(make_input(0))
    y.backward(_int_tensor(y.shape, 500 + 10 * dist.get_rank(), device))
    for (name, p), e in zip(conv.named_parameters(), expected):
        assert torch.equal(p.grad, e), name
