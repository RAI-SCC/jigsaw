"""Boilerplate shared by the tutorials: process-group bring-up and data sharding."""

import os

import torch
import torch.distributed as dist

from jigsaw import log
from jigsaw.utils import init_cuda, resolve_launch_env


def _processes_on_this_node(world_size: int) -> int:
    """Number of launched processes that share this node's GPUs."""
    for var in ("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"):  # torchrun, srun
        if os.getenv(var, "").isdigit():
            return int(os.environ[var])
    return world_size


def _gpus_on_this_node() -> int:
    """Number of GPUs the launched processes on this node have between them.

    Usually every process sees all of them. Some clusters make ``srun`` bind one GPU
    to each task instead: a process then sees a single device, and ``SLURM_STEP_GPUS``
    lists the ones the tasks of this node were given.
    """
    n_gpus = torch.cuda.device_count()
    step_gpus = os.getenv("SLURM_STEP_GPUS", "")
    if n_gpus == 1 and step_gpus and os.getenv("SLURM_PROCID") is not None:
        n_gpus = len(step_gpus.split(","))
    return n_gpus


def setup() -> tuple[torch.device, int, int]:
    """Initialise ``torch.distributed`` and return ``(device, rank, world_size)``.

    With one GPU per process this is :func:`jigsaw.utils.init_cuda` (NCCL, launched by
    ``srun`` or ``torchrun``). Otherwise -- no GPU, or fewer GPUs than processes on the
    node, as on a laptop -- it falls back to gloo on CPU, so every tutorial also runs
    with ``torchrun --nproc-per-node 4``.
    """
    env = resolve_launch_env()
    n_gpus = _gpus_on_this_node()
    if n_gpus and n_gpus >= _processes_on_this_node(env.world_size):
        device, _, rank, world_size = init_cuda()
        return device, rank, world_size

    log.configure_logging()
    if n_gpus:
        log.info("%d GPU(s) for more processes than that: running on CPU", n_gpus)
    dist.init_process_group(
        "gloo", rank=env.rank, world_size=env.world_size, init_method="env://"
    )
    return torch.device("cpu"), env.rank, env.world_size


def shard(x: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
    """Return this rank's slice of ``x`` along ``dim`` when split over ``group``."""
    size = x.shape[dim] // group.size()
    return x.narrow(dim, group.rank() * size, size).contiguous()


def ddp_flags(argv: list[str]) -> dict[str, bool]:
    """``main()`` keyword arguments for ``--ddp`` and ``--joint-hook``.

    ``--joint-hook`` implies ``--ddp``: it is a different way of reducing the
    gradients under ``DistributedDataParallel`` (see ``docs/ddp.md``).
    """
    joint = "--joint-hook" in argv
    return {"data_parallel": joint or "--ddp" in argv, "joint_hook": joint}


def data_seed(step: int, mesh: dict[str, dist.ProcessGroup]) -> int:
    """Seed of the batch of ``step``: the same within a model copy, distinct across copies.

    Every rank of one copy must see the same samples (each keeps only its own tile),
    and every data-parallel copy different ones, or data parallelism only repeats the
    same batch. Without a ``"dp"`` axis this is just ``step``.
    """
    dp = mesh.get("dp")
    return step * dp.size() + dp.rank() if dp is not None else step
