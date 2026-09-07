"""Utility functions to setup distributed environment."""

import os
import subprocess
from dataclasses import dataclass

import torch
import torch.distributed as dist

from jigsaw import log


def set_all_seeds(seed: int = 0) -> None:
    """Initialize all random seeds."""
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class LaunchEnv:
    """Resolved distributed-launch parameters.

    Attributes
    ----------
    rank : int
        Global process rank.
    world_size : int
        Total number of processes.
    local_rank : int
        Rank within the node (selects the local GPU).
    master_addr : str
        Rendezvous host for ``init_method="env://"``.
    master_port : str
        Rendezvous port.
    source : str
        Which launcher was detected: ``"slurm"``, ``"torchrun"`` or ``"single"``.
    """

    rank: int
    world_size: int
    local_rank: int
    master_addr: str
    master_port: str
    source: str


def resolve_launch_env(strict: bool = False) -> LaunchEnv:
    """Read the distributed-launch contract from the environment.

    Supports both ``srun`` (``SLURM_PROCID`` / ``SLURM_NTASKS`` /
    ``SLURM_LOCALID``) and ``torchrun`` (``RANK`` / ``WORLD_SIZE`` /
    ``LOCAL_RANK``). ``MASTER_ADDR`` is taken from the environment, else derived
    from ``SLURM_JOB_NODELIST``, else falls back to ``localhost``;
    ``MASTER_PORT`` defaults to ``29500``. Both are written back to
    ``os.environ`` so a subsequent ``init_method="env://"`` sees them.

    Parameters
    ----------
    strict : bool
        If ``True``, raise ``RuntimeError`` when neither a SLURM nor a torchrun
        launcher is detected instead of assuming a single-process run.

    Returns
    -------
    LaunchEnv
        The resolved parameters.
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
    """
    Initialize DistributedDataParallel.

    Returns
    -------
    device: String
    slurm_localid: int
    rank: int
    world_size: int
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
    # Initialize GPUs and dataloaders
    device = torch.device(f"cuda:{slurm_localid}")
    torch.cuda.set_device(slurm_localid)

    # Initialize DistributedDataParallel.
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
        log.error_all("process group failed to initialise on rank %d", rank)

    return device, slurm_localid, rank, world_size


def count_parameters(model: torch.nn.Module) -> int:
    """Return number of parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# save model


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    file_path: str,
    rank: int,
    epoch: int,
    model_name: str = "model",
) -> None:  # noqa: D103
    """Save model weights, optimizer state, epoch."""
    file_path = f"{file_path}/{model_name}_{rank}.pt"
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, file_path)


def load_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    file_path: str,
    rank: int = 0,
    model_parallel: bool = False,
    parallelism: int = 4,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    """Load model from state dict."""
    if model_parallel:
        load_rank = rank % parallelism
        file_path = f"{file_path}/model_{load_rank}.pt"
    try:
        state = torch.load(f"{file_path}", map_location="cpu")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        epoch = state["epoch"] + 1
        # epoch = 0
    except Exception:
        log.warning(
            "could not load model from %s; initialising from scratch", file_path
        )
        epoch = 0
    return model, optimizer, epoch


def setup_precision(tf32: bool) -> None:
    """Configure matmul and cudnn TF32."""
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    return
