"""A data-parallel training loop: global batch, gradient accumulation, replica checks.

``vit.py --ddp`` and ``swin.py --ddp`` show how to wrap a jigsaw model in
``DistributedDataParallel``. This tutorial is the rest of the training loop, for the
ViT of ``vit.py`` (split 2x2 over "row" and "col") with a "dp" axis of copies on top:

  * The **global batch** is split over "dp": every copy trains on its own share, and
    every rank of one copy reads the same images (keeping only its shard of them).
  * **Gradient accumulation**: a copy's share is cut into microbatches. ``no_sync()``
    skips DDP's all-reduce on all but the last one, and each microbatch's loss is
    divided by their number, so the step uses the gradient of the whole global batch.
  * The **joint comm hook** (``--plain`` switches it off): DDP reduces each replicated
    gradient over all of its copies at once, ``dp`` x "row", in its buckets. Same
    gradients as plain DDP, one collective fewer; see ``docs/ddp.md``.
  * The **loss to log** is the mean over the copies, not just that of rank 0's copy.
  * A **replica check** every few steps: ``jigsaw.debug.assert_replica_consistency``
    raises on every rank once a replicated weight has drifted apart across its
    copies -- the symptom of a parameter whose gradient is not summed over them.

Checkpointing needs nothing special: ``ddp.module.state_dict()`` holds this rank's
shards, which are the same on every copy, so only the ranks of copy 0
(``mesh["dp"].rank() == 0``) need to save, one file each.

Run with a multiple of 4 processes, one per GPU (or on the CPU):

    torchrun --nproc-per-node 8 examples/ddp.py

With 4 processes there is a single copy and DDP has nothing to average; the script
still runs.
"""

import contextlib
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from _common import setup, shard
from vit import H, IN_CHANNELS, W, ViT

import jigsaw
from jigsaw import log
from jigsaw.launch import set_all_seeds

Mesh = dict[str, dist.ProcessGroup]

GLOBAL_BATCH = 8  # images per optimizer step, over all copies
MICRO_BATCH = 2  # images per forward pass of one copy
CHECK_EVERY = 25  # steps between replica checks (a check costs a few collectives)


def global_batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The noisy inputs and clean targets of ``step``, the same on every rank."""
    g = torch.Generator().manual_seed(step)
    clean = torch.nn.functional.avg_pool2d(
        torch.randn(GLOBAL_BATCH, IN_CHANNELS, H, W, generator=g), 5, 1, 2
    )
    return clean + 0.3 * torch.randn(clean.shape, generator=g), clean


def my_share(x: torch.Tensor, mesh: Mesh, device: torch.device) -> torch.Tensor:
    """This rank's part of a global batch: its copy's images, its shard of each.

    In a real pipeline a sampler keyed by ``mesh["dp"].rank()`` picks the images and
    the Dataset reads only the shard (see ``dataloader.py``).
    """
    x = shard(x, 0, mesh["dp"])  # the batch over the copies ...
    return shard(shard(x, 1, mesh["col"]), -1, mesh["row"]).to(device)  # ... C, W


def train_step(
    ddp: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mesh: Mesh,
) -> float:
    """One optimizer step on this copy's share ``x, y``; returns the global batch loss."""
    pairs = list(zip(x.split(MICRO_BATCH), y.split(MICRO_BATCH)))
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=x.device)
    for i, (xm, ym) in enumerate(pairs):
        # Accumulate locally; DDP all-reduces once, in the backward of the last one.
        last = i == len(pairs) - 1
        with contextlib.nullcontext() if last else ddp.no_sync():
            loss = loss_fn(ddp(xm), ym) / len(pairs)  # the mean over the share
            loss.backward()
        total += loss.detach()
    optimizer.step()
    # Each copy saw different images: average their losses (gloo has no AVG).
    dist.all_reduce(total, group=mesh["dp"])
    return total.item() / mesh["dp"].size()


def main(steps: int = 100, joint_hook: bool = True) -> list[float]:
    """Train for ``steps`` steps and return the global batch loss of every step."""
    device, rank, world_size = setup()

    # "row" and "col" split one copy of the model; "dp" (world size / 4) counts them.
    mesh = jigsaw.make_process_mesh(
        {"dp": -1, "row": 2, "col": 2}, model_parallel_axes=("row", "col")
    )
    if GLOBAL_BATCH % (mesh["dp"].size() * MICRO_BATCH):
        raise ValueError(
            f"{mesh['dp'].size()} copies cannot split a global batch of "
            f"{GLOBAL_BATCH} into microbatches of {MICRO_BATCH}"
        )

    set_all_seeds(mesh["model"].rank())  # as in vit.py
    # wrap_ddp: DDP over "dp" only, with the setup checked before the first step.
    ddp = jigsaw.parallel.wrap_ddp(ViT(mesh, device), mesh, joint_hook=joint_hook)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
    loss_fn = jigsaw.nn.DistributedMSELoss(reduce_groups=(mesh["row"], mesh["col"]))

    losses = []
    for step in range(steps):
        x, y = (my_share(t, mesh, device) for t in global_batch(step))
        losses.append(train_step(ddp, optimizer, loss_fn, x, y, mesh))
        if step % CHECK_EVERY == 0 or step == steps - 1:
            # Collective over the whole mesh: every rank must call it.
            jigsaw.debug.assert_replica_consistency(ddp, mesh)
        if step % 20 == 0:
            log.info("step %3d  loss %.4f", step, losses[-1])
    return losses


if __name__ == "__main__":
    main(joint_hook="--plain" not in sys.argv)
