"""Data- plus domain-parallel training gives the same model as serial training.

The Swin tutorial is trained for :data:`STEPS` steps on ``dp=2 x (sh=2 x sw=2)``:
every data-parallel replica gets half of the global batch, and every image is cut
into four tiles. The reference is the same model, undivided, trained on the whole
global batch on one rank (computed redundantly on every rank). The loss curves
and the weight updates have to match.

This pins the loss-scaling convention. The replica sync is a SUM over the tiles
and DDP a mean over ``dp``, so the loss each rank backpropagates must be the
replica's mean loss *as seen from its tile*: either a loss that reduces over the
tiles with an identity backward (``DistributedMSELoss``), or the local mean
divided by the number of tiles. The optimizer is SGD because Adam normalizes the
gradient and would hide a wrong scale.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import jigsaw
from jigsaw import parallel
from jigsaw.modules.roll import DistributedRoll
from tests.parity import assert_close, no_tf32
from tests.test_examples import _load, _tile, _TorchRoll

pytestmark = pytest.mark.dist(8)

STEPS = 8
BATCH = 4  # global
LR = 0.05
# fp32 noise after 8 steps is ~1e-6 of the largest entry; a wrong scale is O(1).
REL = 1e-4

# Every axis of the Swin mesh is split: ``DistributedRoll`` cannot roll over a
# group of size 1, so ``dp=2 x sh=2`` on 4 ranks does not run.
PARALLEL = {"dp": 2, "sh": 2, "sw": 2, "col": 1}
SERIAL = {"dp": 8, "sh": 1, "sw": 1, "col": 1}
LOSSES = ["distributed_mse", "local_mse_scaled"]

_MESH_CACHE: dict = {}


def _mesh(axes: dict, device: torch.device) -> dict:
    key = (tuple(axes.items()), device.type)
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = jigsaw.make_process_mesh(
            axes, device_type=device.type, model_parallel_axes=("sh", "sw", "col")
        )
    return _MESH_CACHE[key]


def _batch(step: int, size: tuple[int, int], device: torch.device):
    g = torch.Generator().manual_seed(step)
    x = torch.randn(BATCH, 1, *size, generator=g)
    y = torch.randn(BATCH, 1, *size, generator=g)
    return x.to(device), y.to(device)


def _loss(kind: str, out: torch.Tensor, y: torch.Tensor, mesh: dict) -> torch.Tensor:
    tiles = (mesh["sh"], mesh["sw"])
    if kind == "distributed_mse":
        return jigsaw.loss.DistributedMSELoss(reduce_groups=tiles)(out, y)
    local = F.mse_loss(out, y)
    if kind == "local_mse_scaled":
        return local / (tiles[0].size() * tiles[1].size())
    if kind == "local_mse":  # unscaled: wrong by the number of tiles
        return local
    raise ValueError(kind)


def train_parallel(loss_kind: str, device: torch.device, steps: int = STEPS):
    """Train the Swin tutorial on :data:`PARALLEL` through ``wrap_ddp``.

    Returns the global loss of every step, and the names, initial and final values
    of the parameters.
    """
    module = _load("swin")
    mesh = _mesh(PARALLEL, device)
    dp, model_group = mesh["dp"], mesh["model"]
    torch.manual_seed(0)
    model = module.Swin(mesh, device)
    ddp = parallel.wrap_ddp(model, mesh)
    init = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(ddp.parameters(), lr=LR, momentum=0.9)

    per_replica = BATCH // dp.size()
    losses = []
    with no_tf32():
        for step in range(steps):
            x, y = _batch(step, (module.H, module.W), device)
            mine = slice(dp.rank() * per_replica, (dp.rank() + 1) * per_replica)
            tiles = [(-2, "sh"), (-1, "sw")]
            x, y = _tile(x[mine], mesh, tiles), _tile(y[mine], mesh, tiles)
            loss = _loss(loss_kind, ddp(x), y, mesh)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            seen = loss.detach().clone()
            if loss_kind != "distributed_mse":  # the replica's loss is the tile sum
                dist.all_reduce(seen, group=model_group)
            dist.all_reduce(seen, group=dp)
            losses.append(seen.item() / dp.size())
    named = list(model.named_parameters())
    return losses, [n for n, _ in named], init, [p.detach() for _, p in named]


def train_serial(init: list, device: torch.device, steps: int = STEPS):
    """Train the undivided Swin tutorial on the whole batch, from ``init``."""
    module = _load("swin")
    model = module.Swin(_mesh(SERIAL, device), device)
    for p, q in zip(model.parameters(), init, strict=True):
        p.data.copy_(q)
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, DistributedRoll):
                setattr(parent, name, _TorchRoll(child.shifts, child.dims))
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    losses = []
    with no_tf32():
        for step in range(steps):
            x, y = _batch(step, (module.H, module.W), device)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return losses, [p.detach() for p in model.parameters()]


def compare(loss_kind: str, device: torch.device) -> None:
    """Train both ways and fail on every rank if they differ."""
    losses, names, init, final = train_parallel(loss_kind, device)
    ref_losses, ref_final = train_serial(init, device)
    world = dist.group.WORLD

    got, want = torch.tensor(losses), torch.tensor(ref_losses)
    atol = REL * want.abs().max().item()
    assert_close("loss curve", got, want, atol, REL, device, world)
    for name, p, q, p0 in zip(names, final, ref_final, init, strict=True):
        # Compare the updates: they are small next to the weights themselves.
        upd, ref_upd = p - p0, q - p0
        atol = REL * ref_upd.abs().max().item() + 1e-7
        assert_close(f"{name} update", upd, ref_upd, atol, REL, device, world)


@pytest.mark.parametrize("loss_kind", LOSSES)
def test_matches_serial_training(loss_kind: str, device: torch.device) -> None:
    compare(loss_kind, device)
