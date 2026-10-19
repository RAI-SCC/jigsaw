"""Domain parallelism only: split the image over the ranks, replicate the weights.

A U-Net-style denoiser is trained on images that are cut into a 2x2 grid of tiles,
one tile per rank. No parameter is sharded -- every rank holds the full model -- so
memory per rank only shrinks with the *activations*, which is what dominates for
large images. This is the simplest way to use jigsaw.

What jigsaw does for you:
  * ``DistributedConv2d`` swaps a halo with its neighbours, so a 3x3 convolution
    gives the same result on a tile border as on the undivided image.
  * The weights are broadcast from rank 0 at construction and their gradients are
    summed over the spatial groups inside the backward pass, so a plain
    ``torch.optim`` optimizer keeps the replicas identical.
  * ``DistributedMSELoss`` reduces over the tiles: every rank sees the loss of the
    whole image.

Run with one process per GPU (or 4 CPU processes):

    torchrun --nproc-per-node 4 examples/domain_parallel.py

Add data parallelism on top with ``--ddp``: every group of 4 ranks then holds one
copy of the model and trains on its own images, and ``DistributedDataParallel``
averages the gradients over the copies:

    torchrun --nproc-per-node 8 examples/domain_parallel.py --ddp

``--joint-hook`` reduces the gradients of the replicated weights with one DDP comm
hook instead (one all-reduce over ``dp`` x tiles, rather than a sum over the tiles in
the backward pass and an average over ``dp`` afterwards); the result is the same.
See ``docs/ddp.md``:

    torchrun --nproc-per-node 8 examples/domain_parallel.py --joint-hook
"""

import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from _common import data_seed, ddp_flags, setup, shard, steps_flag

import jigsaw
from jigsaw import log
from jigsaw.gather import gather_results


def build_model(mesh: dict[str, dist.ProcessGroup], device: torch.device) -> nn.Module:
    """A small U-Net-style denoiser whose convolutions split ``(H, W)`` over the mesh."""
    spatial = (mesh["sh"], mesh["sw"])  # groups splitting (H, W)

    def conv(cin: int, cout: int, stride: int = 1) -> nn.Module:
        """A 3x3 distributed convolution with 'same' padding."""
        return jigsaw.modules.DistributedConv2d(
            cin,
            cout,
            kernel_size=3,
            stride=stride,
            padding=1,
            device=device,
            spatial_group=spatial,
            channel_group=mesh["col"],
        )

    def up(cin: int, cout: int) -> nn.Module:
        """A stride-2 transposed convolution that doubles the resolution."""
        return jigsaw.modules.DistributedConvTranspose2d(
            cin,
            cout,
            kernel_size=2,
            stride=2,
            device=device,
            spatial_group=spatial,
            channel_group=mesh["col"],
        )

    return nn.Sequential(
        conv(1, 16),
        nn.GELU(),
        conv(16, 32, stride=2),  # halves the resolution
        nn.GELU(),
        up(32, 16),
        nn.GELU(),
        conv(16, 1),
    )


def make_batch(
    step: int, mesh: dict[str, dist.ProcessGroup], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return this rank's tile of a noisy input and its clean target.

    Every rank of a model copy draws the *same* batch (same seed) and keeps only its
    tile; with data parallelism every copy draws a different one. In a real pipeline
    the Dataset reads just that tile from disk instead.
    """
    g = torch.Generator().manual_seed(data_seed(step, mesh))
    clean = torch.nn.functional.avg_pool2d(  # smooth random fields
        torch.randn(4, 1, 64, 64, generator=g), kernel_size=5, stride=1, padding=2
    )
    noisy = clean + 0.3 * torch.randn(clean.shape, generator=g)

    def tile(x: torch.Tensor) -> torch.Tensor:
        return shard(shard(x, -2, mesh["sh"]), -1, mesh["sw"]).to(device)

    return tile(noisy), tile(clean)


def main(
    steps: int = 100, data_parallel: bool = False, joint_hook: bool = False
) -> list[float]:
    """Train the denoiser for ``steps`` steps and return the loss of every step."""
    device, rank, world_size = setup()

    # One axis per split dimension: "sh" splits the height, "sw" the width. The channel
    # axis "col" has size 1, i.e. no tensor parallelism. The mesh also gets a "model"
    # group spanning the four ranks of one model copy.
    if data_parallel:
        # A leading "dp" axis holds the copies: world size / 4 of them.
        mesh = jigsaw.make_process_mesh(
            {"dp": -1, "sh": 2, "sw": 2, "col": 1},
            model_parallel_axes=("sh", "sw", "col"),
        )
    else:
        mesh = jigsaw.make_process_mesh({"sh": 2, "sw": 2, "col": 1})

    model = build_model(mesh, device)
    if data_parallel:
        # Plain DistributedDataParallel over "dp" only. The default group would also
        # span the tiles of a copy, which jigsaw rejects in the first forward: with
        # sharded weights it would average different shards together.
        model = DistributedDataParallel(model, process_group=mesh["dp"])
        if joint_hook:
            # Collective: every rank, before the first forward.
            jigsaw.parallel.register_joint_comm_hook(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Every rank averages over the *global* image, so all ranks log the same number.
    loss_fn = jigsaw.modules.DistributedMSELoss(reduce_groups=(mesh["sh"], mesh["sw"]))

    losses = []
    for step in range(steps):
        x, y = make_batch(step, mesh, device)  # each (4, 1, 32, 32): a quarter of 64x64
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 20 == 0:
            log.info("step %3d  loss %.4f", step, losses[-1])

    # To look at a full prediction, gather the tiles: `gather_results` stitches the last
    # two dimensions back together (the "row" axis is the height, "col" the width).
    with torch.no_grad():
        pred = gather_results(model(x), mesh, pm_key_row="sh", pm_key_col="sw")
    log.info(
        "local tile %s -> gathered prediction %s", tuple(x.shape), tuple(pred.shape)
    )
    return losses


if __name__ == "__main__":
    main(**ddp_flags(sys.argv), **steps_flag(sys.argv))
