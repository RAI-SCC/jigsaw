"""Process launch and runtime setup."""

import os
import subprocess
from dataclasses import dataclass

import torch
import torch.distributed as dist

from jigsaw import log

__all__ = [
    "LaunchEnv",
    "init_cuda",
    "resolve_launch_env",
    "set_all_seeds",
    "setup_precision",
]


def set_all_seeds(seed: int = 0) -> None:
    """Seed the PyTorch CPU and CUDA random number generators.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class LaunchEnv:
    """Distributed launch parameters.

    Attributes
    ----------
    rank : int
        Global process rank.
    world_size : int
        Total number of processes.
    local_rank : int
        Rank within the node (selects the local GPU).
    master_addr : str
        Rendezvous host.
    master_port : str
        Rendezvous port.
    source : str
        Detected launcher: ``"slurm"``, ``"torchrun"`` or ``"single"``.
    """

    rank: int
    world_size: int
    local_rank: int
    master_addr: str
    master_port: str
    source: str


def resolve_launch_env(strict: bool = False) -> LaunchEnv:
    """Read the launch parameters from the environment.

    Supports ``srun`` and ``torchrun``. ``MASTER_ADDR`` and ``MASTER_PORT`` are
    written back to ``os.environ``.

    Parameters
    ----------
    strict : bool
        Raise ``RuntimeError`` if no launcher is detected instead of assuming a
        single process.

    Returns
    -------
    LaunchEnv
        Resolved launch parameters.
    """
    if os.getenv("SLURM_PROCID") is not None:
        source = "slurm"
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.getenv("SLURM_NTASKS") or 1)
        local_rank = int(os.getenv("SLURM_LOCALID") or 0)
    elif os.getenv("RANK") is not None:
        source = "torchrun"
        rank = int(os.environ["RANK"])
        world_size = int(os.getenv("WORLD_SIZE") or 1)
        local_rank = int(os.getenv("LOCAL_RANK") or 0)
    else:
        if strict:
            raise RuntimeError(
                "No distributed launcher detected: expected SLURM_PROCID (srun) "
                "or RANK (torchrun) in the environment. Launch via "
                "`srun scripts/run.sh ...` or `torchrun ...`, or pass strict=False "
                "for a single-process run."
            )
        source = "single"
        rank, world_size, local_rank = 0, 1, 0

    master_addr = os.getenv("MASTER_ADDR", "")
    if not master_addr:
        nodelist = os.getenv("SLURM_JOB_NODELIST", "")
        if nodelist:
            master_addr = (
                subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
                .decode()
                .splitlines()[0]
                .strip()
            )
        else:
            master_addr = "localhost"
    master_port = os.getenv("MASTER_PORT", "29500")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    return LaunchEnv(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
        source=source,
    )


def init_cuda() -> tuple[torch.device, int, int, int]:
    """Select the local CUDA device and initialize the NCCL process group.

    Returns
    -------
    device : torch.device
        Local CUDA device.
    local_rank : int
        Rank within the node.
    rank : int
        Global rank.
    world_size : int
        Total number of processes.
    """
    env = resolve_launch_env()
    rank, world_size, slurm_localid = env.rank, env.world_size, env.local_rank

    log.configure_logging()
    log.info_all(
        "launch: rank %d/%d, local_rank %d, via %s",
        rank,
        world_size,
        slurm_localid,
        env.source,
    )
    # Select the local GPU. A scheduler that binds one GPU to each task
    # (`srun` on some clusters) leaves a single visible device, which is cuda:0
    # whatever the local rank.
    device_id = slurm_localid if torch.cuda.device_count() != 1 else 0
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)

    # Initialize the NCCL process group.
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method="env://"
    )

    if dist.is_initialized():
        log.info(
            "process group up: torch rank %d, world size %d, backend %s",
            dist.get_rank(),
            dist.get_world_size(),
            dist.get_backend(),
        )
    else:
        log.error_all("process group failed to initialize on rank %d", rank)

    return device, slurm_localid, rank, world_size


def setup_precision(tf32: bool) -> None:
    """Enable or disable TF32 for matmul and cuDNN.

    Parameters
    ----------
    tf32 : bool
        Whether to allow TF32.
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    return
