"""A dataloader for domain parallelism: every rank reads only its tile of a sample.

With a domain-parallel model the ranks of one model replica work on *the same sample*,
each on a different piece of it. That changes what a dataloader has to do:

  1. The Dataset returns a tile, not the whole sample. Read just that tile from disk
     (here a memory-mapped ``.npy`` file; with zarr or HDF5 you slice the array the same
     way): the I/O per rank shrinks with the number of tiles, and no rank ever holds
     the full sample.
  2. All ranks of a model replica must draw the same sample indices in the same order,
     or the tiles they hold do not belong together. Ranks of *different* replicas must
     draw different indices. So the sampler is keyed by the **data-parallel index**
     (``mesh["dp"].rank()``), not by the global rank -- the usual
     ``DistributedSampler(dataset)`` would hand every rank its own samples.
  3. Every replica has to run the same number of steps, because every step contains
     collectives: use ``drop_last=True``.
  4. Anything random that belongs to a sample (a flip, a crop, noise) has to be seeded
     by the sample, not by the rank, or the tiles of one sample get different draws.

Below, the mesh is 2 data-parallel replicas of a model that is split in two along the
height. The script draws a few batches and checks that the tiles of each replica fit
together into one sample, and that the two replicas never see the same sample.

Averaging the gradients of the data-parallel replicas is not part of this example.

Run with one process per GPU (or 4 CPU processes):

    torchrun --nproc-per-node 4 examples/dataloader.py
"""

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from _common import setup
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import jigsaw
from jigsaw import log
from jigsaw.utils_aggregate import gather_results

N, C, H, W = 16, 2, 16, 16  # samples, channels, global image size


class TileDataset(Dataset):
    """Samples of shape ``(C, H, W)`` stored in one ``.npy`` file; returns one tile."""

    def __init__(self, path: Path, splits: dict[int, dist.ProcessGroup]) -> None:
        """``splits`` maps a dimension of a sample to the group that splits it."""
        self.path = path
        data = np.load(path, mmap_mode="r")
        self.length = len(data)

        # This rank's slice along every dimension. Only plain slices are stored so that
        # the dataset can be sent to worker processes.
        index = [slice(None)] * (data.ndim - 1)
        for dim, group in splits.items():
            size = data.shape[dim + 1] // group.size()
            index[dim] = slice(group.rank() * size, (group.rank() + 1) * size)
        self.index = tuple(index)
        self.data: np.ndarray | None = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        if self.data is None:  # open the file inside the worker that reads it
            self.data = np.load(self.path, mmap_mode="r")
        tile = self.data[(i, *self.index)]  # touches only this tile's bytes
        return torch.from_numpy(np.ascontiguousarray(tile)), i


def main(num_workers: int = 2) -> None:
    """Draw batches on 2 replicas x 2 tiles and check that they fit together."""
    device, rank, world_size = setup()

    # "dp" enumerates the replicas; the other axes split one replica over 2 ranks.
    mesh = jigsaw.make_process_mesh(
        {"dp": 2, "sh": 2, "sw": 1, "col": 1}, model_parallel_axes=("sh", "sw", "col")
    )
    dp = mesh["dp"]

    # --- a small dataset on disk, written by rank 0 ---------------------------------------
    # (/tmp is local to a node: with several nodes, use a shared file system instead.)
    if rank == 0:
        root = Path(tempfile.mkdtemp())
        samples = np.lib.format.open_memmap(
            root / "samples.npy", mode="w+", dtype=np.float32, shape=(N, C, H, W)
        )
        samples[:] = np.random.default_rng(0).standard_normal((N, C, H, W))
        samples.flush()
    paths = [root if rank == 0 else None]
    dist.broadcast_object_list(paths, src=0)  # every rank needs to know where it is
    root = paths[0]

    # --- the dataloader ---------------------------------------------------------------------
    # Dimensions of a sample are (C, H, W) = (0, 1, 2): split the height over "sh" and
    # the width over "sw" (a size-1 axis is a no-op). To also split the channels over
    # "col", add ``0: mesh["col"]``.
    dataset = TileDataset(root / "samples.npy", splits={1: mesh["sh"], 2: mesh["sw"]})

    sampler = DistributedSampler(
        dataset,
        num_replicas=dp.size(),  # not world_size ...
        rank=dp.rank(),  # ... and not the global rank
        shuffle=True,
        seed=0,  # the same on every rank: the replicas then draw disjoint samples
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=num_workers,  # each worker reads tiles, never whole samples
        drop_last=True,  # every replica runs the same number of steps
        pin_memory=device.type == "cuda",
    )

    # --- draw batches and check them --------------------------------------------------------
    full = np.load(root / "samples.npy", mmap_mode="r")  # only used for the check

    for epoch in range(2):
        sampler.set_epoch(epoch)  # reshuffle; the same order on the ranks of a replica
        for step, (x, idx) in enumerate(loader):
            x = x.to(device)  # (B, C, H / sh, W / sw): what a domain-parallel model takes

            # 1. The ranks of a replica hold tiles of the same samples ...
            #    (The gather buffers live on ``device``: NCCL only moves GPU tensors.)
            same = [torch.empty_like(idx, device=device) for _ in range(mesh["model"].size())]
            dist.all_gather(same, idx.to(device), group=mesh["model"])
            assert all((i.cpu() == idx).all() for i in same)

            # ... and the tiles fit together into those samples.
            whole = gather_results(x, mesh, pm_key_row="sh", pm_key_col="sw")
            assert np.array_equal(whole.cpu().numpy(), full[idx.numpy()])

            # 2. The other replica draws different samples.
            seen = [torch.empty_like(idx, device=device) for _ in range(dp.size())]
            dist.all_gather(seen, idx.to(device), group=dp)
            flat = torch.cat([i.cpu() for i in seen])
            assert len(flat.unique()) == len(flat)

            if epoch == 0 and step == 0:
                log.log_ranks(
                    "replica %d, tile %d of %d: samples %s, batch %s",
                    dp.rank(),
                    mesh["sh"].rank(),
                    mesh["sh"].size(),
                    idx.tolist(),
                    tuple(x.shape),
                    level=logging.INFO,
                )

    log.info("%d steps per epoch: all checks passed", step + 1)

    dist.barrier()
    if rank == 0:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
