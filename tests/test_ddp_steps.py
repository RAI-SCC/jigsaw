"""Replicas stay bitwise identical over many optimizer steps under DDP.

A wrong replica sum shows up as a gradient that differs between replicas, and
from there as parameters that drift apart. One step can hide a small error that
AdamW's normalization amplifies later, so this trains for :data:`STEPS` steps
and checks after *every* ``optimizer.step()`` that each marked parameter is
identical across its replica groups and every parameter across ``dp``. Bitwise
identity depends on the backend's reduction order, so it runs on gloo and NCCL.
"""

import pytest
import torch

from jigsaw import debug, parallel
from tests.test_ddp import KINDS, MESHES, _assert_identical_over, _build, _mesh

STEPS = 20


def _cases():
    for shape in MESHES:
        world = shape[0] * shape[1] * shape[2]
        for kind in KINDS:
            yield pytest.param(
                shape,
                kind,
                id=f"{'x'.join(map(str, shape))}-{kind}",
                marks=pytest.mark.dist(world),
            )


def train(shape, kind, device, steps: int = STEPS) -> list:
    """Train through ``wrap_ddp`` and check the replicas after every step.

    The global check runs first, so a divergence raises on every rank at the same
    step instead of on some ranks only. Returns the initial parameters.
    """
    mesh = _mesh(shape, device)
    model, make_input = _build(kind, mesh, device)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.1)  # keep the integer init from saturating attention
    ddp = parallel.wrap_ddp(model, mesh)
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in model.parameters()]

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ddp(make_input(step)).float().pow(2).mean()
        loss.backward()
        opt.step()
        debug.assert_replica_consistency(model, mesh)
        for name, p in model.named_parameters():
            for grp in parallel.replica_groups(p):
                _assert_identical_over(p.detach(), grp, f"step {step}: {name}")
            _assert_identical_over(p.detach(), mesh["dp"], f"step {step}: {name}")
    return [(name, p, b) for (name, p), b in zip(model.named_parameters(), before)]


@pytest.mark.parametrize("shape, kind", list(_cases()))
def test_replicas_stay_identical(shape, kind, device) -> None:
    for name, p, b in train(shape, kind, device):
        assert torch.isfinite(p).all(), name
        assert not torch.equal(p, b), f"{name} never moved"
