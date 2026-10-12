"""The tutorials with data parallelism on top (``--ddp``, ``--joint-hook``, ``ddp.py``).

Run under ``mpirun -n 8 pytest tests``: two data-parallel copies of each 4-rank model.

* **Smoke** -- ``main()`` of every tutorial runs in every DDP mode, the loss goes
  down, and the copies train on different images.
* **Parity with data-serial training** -- the model of each tutorial, wrapped in DDP
  over ``dp`` the three ways the tutorials and ``docs/ddp.md`` show (a hand-built
  ``DistributedDataParallel``, ``wrap_ddp``, ``wrap_ddp(joint_hook=True)``), with each
  copy training on its half of the global batch, against the same model without DDP
  training on the whole batch. Both use SGD, which unlike Adam does not hide a wrong
  gradient scale. The reference is still model-parallel; ``test_examples.py`` checks
  that against plain PyTorch, so together they pin the DDP model to a serial one.
  ``ddp.py``'s training step (global batch split over ``dp``, gradient accumulation
  with ``no_sync``) is checked the same way.

After training, every replica must be bitwise identical to its copies
(:func:`jigsaw.debug.assert_replica_consistency`, unmarked layers included).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import jigsaw
from jigsaw import debug, parallel
from jigsaw.launch import set_all_seeds
from tests.parity import assert_close, no_tf32
from tests.test_examples import (
    REL,
    REL_FP16_ATTENTION,
    _agree,
    _load,
    _mesh,
    _tile,
)

pytestmark = [pytest.mark.dist_required, pytest.mark.dist(8)]

STEPS = 5
BATCH = 4  # global, split over the two copies
LR = 0.05


@pytest.fixture
def example(
    monkeypatch: pytest.MonkeyPatch, device: torch.device
) -> Callable[[str], ModuleType]:
    """Load a tutorial on the test session's process group, with cached meshes."""

    def make_process_mesh(axes, device_type=None, model_parallel_axes=None):
        return _mesh(axes, model_parallel_axes, device)

    monkeypatch.setattr(jigsaw, "make_process_mesh", make_process_mesh)

    def load(name: str) -> ModuleType:
        module = _load(name)
        monkeypatch.setattr(
            module, "setup", lambda: (device, dist.get_rank(), dist.get_world_size())
        )
        return module

    return load


# --- smoke ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint_hook", [False, True], ids=["ddp", "joint"])
@pytest.mark.parametrize("name", ["domain_parallel", "vit", "swin"])
def test_training_reduces_loss(
    example: Callable[[str], ModuleType],
    device: torch.device,
    name: str,
    joint_hook: bool,
) -> None:
    torch.manual_seed(0)
    losses = example(name).main(steps=30, data_parallel=True, joint_hook=joint_hook)
    first, last = sum(losses[:3]) / 3, sum(losses[-3:]) / 3
    _agree(all(map(math.isfinite, losses)), device, "non-finite loss")
    _agree(last < first, device, f"loss did not go down: {first:.4f} -> {last:.4f}")

    # The copies must draw different images (the losses are per copy here). The
    # tutorial's mesh is the cached one of the parity tests below.
    t = TUTORIALS[name]
    mesh = _mesh(t.axes, t.model_axes, device)
    seen = torch.tensor([losses[0]], dtype=torch.float64, device=device)
    both = [torch.empty_like(seen) for _ in range(2)]
    dist.all_gather(both, seen, group=mesh["dp"])
    _agree(not torch.equal(*both), device, "both copies trained on the same images")


@pytest.mark.parametrize("joint_hook", [False, True], ids=["plain", "joint"])
def test_ddp_tutorial_runs(
    example: Callable[[str], ModuleType], device: torch.device, joint_hook: bool
) -> None:
    torch.manual_seed(0)
    losses = example("ddp").main(steps=30, joint_hook=joint_hook)  # checks replicas
    first, last = sum(losses[:3]) / 3, sum(losses[-3:]) / 3
    _agree(all(map(math.isfinite, losses)), device, "non-finite loss")
    _agree(last < first, device, f"loss did not go down: {first:.4f} -> {last:.4f}")
    # The logged loss is the global batch's: the same on every rank.
    mine = torch.tensor(losses, dtype=torch.float64, device=device)
    ref = mine.clone()
    dist.broadcast(ref, src=0)
    _agree(torch.equal(mine, ref), device, "ranks log different losses")


# --- parity with data-serial training ------------------------------------------------


@dataclass(frozen=True)
class Tutorial:
    """How to build, feed and score the model of one tutorial on a ``dp`` mesh."""

    name: str
    axes: dict
    model_axes: tuple[str, ...]
    tiles: tuple[tuple[int, str], ...]  # (dim, mesh axis) splits of an input
    shape: Callable[[ModuleType], tuple[int, int, int]]  # (C, H, W) of an image
    build: Callable[[ModuleType, dict, torch.device], nn.Module]


def _seeded(build: Callable) -> Callable:
    def seeded(module: ModuleType, mesh: dict, device: torch.device) -> nn.Module:
        set_all_seeds(mesh["model"].rank())  # as the ViT tutorial does
        return build(module, mesh, device)

    return seeded


SPATIAL = {"dp": 2, "sh": 2, "sw": 2, "col": 1}
TUTORIALS = {
    t.name: t
    for t in [
        Tutorial(
            "domain_parallel",
            SPATIAL,
            ("sh", "sw", "col"),
            ((-2, "sh"), (-1, "sw")),
            lambda m: (1, 64, 64),
            _seeded(lambda m, mesh, d: m.build_model(mesh, d)),
        ),
        Tutorial(
            "swin",
            SPATIAL,
            ("sh", "sw", "col"),
            ((-2, "sh"), (-1, "sw")),
            lambda m: (m.IN_CHANNELS, m.H, m.W),
            _seeded(lambda m, mesh, d: m.Swin(mesh, d)),
        ),
        Tutorial(
            "vit",
            {"dp": 2, "row": 2, "col": 2},
            ("row", "col"),
            ((1, "col"), (-1, "row")),
            lambda m: (m.IN_CHANNELS, m.H, m.W),
            _seeded(lambda m, mesh, d: m.ViT(mesh, d)),
        ),
    ]
}


def _wrap(model: nn.Module, how: str, mesh: dict) -> nn.Module:
    if how == "ddp":  # domain_parallel.py: checked in the first forward
        return DistributedDataParallel(model, process_group=mesh["dp"])
    return parallel.wrap_ddp(model, mesh, joint_hook=how == "joint")


def _batch(step: int, shape: tuple[int, int, int], device: torch.device):
    g = torch.Generator().manual_seed(step)
    x = torch.randn(BATCH, *shape, generator=g)
    return x.to(device), torch.randn(BATCH, *shape, generator=g).to(device)


def _rel(name: str, device: torch.device) -> float:
    return REL_FP16_ATTENTION if name == "vit" and device.type == "cuda" else REL


def _compare(
    name: str,
    losses: list[float],
    ref_losses: list[float],
    model: nn.Module,
    ref: nn.Module,
    init: list[torch.Tensor],
    device: torch.device,
) -> None:
    """Fail on every rank unless the loss curves and the weight updates match."""
    world, rel = dist.group.WORLD, _rel(name, device)
    got, want = torch.tensor(losses), torch.tensor(ref_losses)
    assert_close(
        "loss curve", got, want, rel * want.abs().max().item(), rel, device, world
    )
    for (pname, p), q, p0 in zip(
        model.named_parameters(), ref.parameters(), init, strict=True
    ):
        # The updates, not the weights: they are small next to the weights.
        upd, ref_upd = p.detach() - p0, q.detach() - p0
        atol = rel * ref_upd.abs().max().item() + 1e-7
        assert_close(f"{pname} update", upd, ref_upd, atol, rel, device, world)


@pytest.mark.parametrize("how", ["ddp", "wrap_ddp", "joint"])
@pytest.mark.parametrize("name", list(TUTORIALS))
def test_matches_data_serial_training(
    example: Callable[[str], ModuleType], device: torch.device, name: str, how: str
) -> None:
    """Each copy on half the batch, under DDP == one copy on the whole batch."""
    t = TUTORIALS[name]
    module = example(name)
    mesh = _mesh(t.axes, t.model_axes, device)
    dp = mesh["dp"]
    model, ref = t.build(module, mesh, device), t.build(module, mesh, device)
    with torch.no_grad():
        for p, q in zip(ref.parameters(), model.parameters(), strict=True):
            p.copy_(q)
    init = [p.detach().clone() for p in model.parameters()]
    wrapped = _wrap(model, how, mesh)
    opt = torch.optim.SGD(wrapped.parameters(), lr=LR, momentum=0.9)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=LR, momentum=0.9)
    loss_fn = jigsaw.loss.DistributedMSELoss(
        reduce_groups=tuple(mesh[a] for a in t.model_axes)
    )

    half = BATCH // dp.size()
    mine = slice(dp.rank() * half, (dp.rank() + 1) * half)
    losses, ref_losses = [], []
    with no_tf32():
        for step in range(STEPS):
            x, y = _batch(step, t.shape(module), device)
            for net, optim, rows, out in (
                (wrapped, opt, mine, losses),
                (ref, ref_opt, slice(None), ref_losses),
            ):
                loss = loss_fn(
                    net(_tile(x[rows], mesh, t.tiles)), _tile(y[rows], mesh, t.tiles)
                )
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                out.append(loss.detach())
    # Under DDP each copy saw its half: the global loss is the mean over dp.
    losses_t = torch.stack(losses)
    dist.all_reduce(losses_t, group=dp)
    _compare(
        name,
        (losses_t / dp.size()).tolist(),
        torch.stack(ref_losses).tolist(),
        model,
        ref,
        init,
        device,
    )
    debug.assert_replica_consistency(model, mesh, unmarked=True)


@pytest.mark.parametrize("joint_hook", [False, True], ids=["plain", "joint"])
def test_ddp_tutorial_step_matches_full_batch(
    example: Callable[[str], ModuleType], device: torch.device, joint_hook: bool
) -> None:
    """``ddp.py``'s step -- split over dp, accumulated over microbatches -- is exact.

    Each copy gets half of the global batch in two microbatches, accumulated under
    ``no_sync``; the reference takes the whole batch in one forward pass, no DDP.
    """
    module = example("ddp")
    t = TUTORIALS["vit"]
    mesh = _mesh(t.axes, t.model_axes, device)
    share = module.GLOBAL_BATCH // mesh["dp"].size()
    assert share // module.MICRO_BATCH == 2, "the test wants two microbatches"
    model, ref = t.build(module, mesh, device), t.build(module, mesh, device)
    with torch.no_grad():
        for p, q in zip(ref.parameters(), model.parameters(), strict=True):
            p.copy_(q)
    init = [p.detach().clone() for p in model.parameters()]
    ddp = parallel.wrap_ddp(model, mesh, joint_hook=joint_hook)
    opt = torch.optim.SGD(ddp.parameters(), lr=LR, momentum=0.9)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=LR, momentum=0.9)
    loss_fn = jigsaw.loss.DistributedMSELoss(reduce_groups=(mesh["row"], mesh["col"]))

    losses, ref_losses = [], []
    with no_tf32():
        for step in range(STEPS):
            x, y = module.global_batch(step)
            share_x, share_y = (module.my_share(v, mesh, device) for v in (x, y))
            losses.append(module.train_step(ddp, opt, loss_fn, share_x, share_y, mesh))

            whole = [_tile(v.to(device), mesh, t.tiles) for v in (x, y)]
            loss = loss_fn(ref(whole[0]), whole[1])
            ref_opt.zero_grad(set_to_none=True)
            loss.backward()
            ref_opt.step()
            ref_losses.append(loss.item())
    _compare("vit", losses, ref_losses, model, ref, init, device)
    debug.assert_replica_consistency(model, mesh, unmarked=True)
