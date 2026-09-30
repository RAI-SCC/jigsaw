"""Boilerplate shared by the tutorials: process-group bring-up and data sharding."""

import torch
import torch.distributed as dist

from jigsaw import log
from jigsaw.utils import init_cuda, resolve_launch_env


def setup() -> tuple[torch.device, int, int]:
    """Initialise ``torch.distributed`` and return ``(device, rank, world_size)``.

    With a GPU this is :func:`jigsaw.utils.init_cuda` (NCCL, one process per GPU,
    launched by ``srun`` or ``torchrun``). Without one it falls back to gloo on CPU,
    so every tutorial also runs on a laptop with ``torchrun --nproc-per-node 4``.
    """
    if torch.cuda.is_available():
        device, _, rank, world_size = init_cuda()
        return device, rank, world_size

    env = resolve_launch_env()
    log.configure_logging()
    dist.init_process_group(
        "gloo", rank=env.rank, world_size=env.world_size, init_method="env://"
    )
    return torch.device("cpu"), env.rank, env.world_size


def shard(x: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
    """Return this rank's slice of ``x`` along ``dim`` when split over ``group``."""
    size = x.shape[dim] // group.size()
    return x.narrow(dim, group.rank() * size, size).contiguous()
