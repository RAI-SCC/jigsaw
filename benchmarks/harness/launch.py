"""Distributed process-group setup for benchmark cases.

Reads the launch contract via ``jigsaw.utils.resolve_launch_env`` (SLURM /
torchrun / single process), picks NCCL on CUDA and gloo on CPU, and brings up the
default process group. ``world_size == 1`` is fine -- the distributed cases still
build a 1x1 mesh and exercise the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from jigsaw.utils import resolve_launch_env

__all__ = ["LaunchCtx", "init", "cleanup"]


@dataclass
class LaunchCtx:
    """Resolved device + rank info for the current process."""

    device: torch.device
    rank: int
    world_size: int
    local_rank: int
    backend: str


def init(device_pref: str = "auto") -> LaunchCtx:
    """Resolve the launch env and initialise the default process group.

    Parameters
    ----------
    device_pref
        ``"auto"`` (CUDA when available), ``"cuda"`` or ``"cpu"``.
    """
    env = resolve_launch_env()  # also writes MASTER_ADDR / MASTER_PORT
    use_cuda = device_pref != "cpu" and torch.cuda.is_available()
    if use_cuda:
        local = env.local_rank % torch.cuda.device_count()
        device = torch.device("cuda", local)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=env.rank,
            world_size=env.world_size,
            device_id=device.index if device.type == "cuda" else None,
        )
    return LaunchCtx(device, env.rank, env.world_size, env.local_rank, backend)


def cleanup() -> None:
    """Tear down the default process group if it is up."""
    if dist.is_initialized():
        dist.destroy_process_group()
