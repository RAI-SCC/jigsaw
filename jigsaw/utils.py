"""Utilities for distributed setup and checkpointing."""

import os
import subprocess
from dataclasses import dataclass

import torch
import torch.distributed as dist

from jigsaw import log


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
    # Initialize GPUs and dataloaders. A scheduler that binds one GPU to each task
    # (`srun` on some clusters) leaves a single visible device, which is cuda:0
    # whatever the local rank.
    device_id = slurm_localid if torch.cuda.device_count() != 1 else 0
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)

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
    """Count the trainable parameters of a model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to inspect.

    Returns
    -------
    int
        Number of parameters with ``requires_grad``.
    """
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
    """Save the model and optimizer state and the epoch to ``<file_path>/<model_name>_<rank>.pt``.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    optimizer : torch.optim.Optimizer
        Optimizer to save.
    scheduler : object
        Unused.
    file_path : str
        Output directory.
    rank : int
        Rank, used in the file name.
    epoch : int
        Current epoch.
    model_name : str
        File name prefix.
    """
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
    """Load model and optimizer state from a checkpoint.

    Starts from epoch 0 if the checkpoint cannot be loaded.

    Parameters
    ----------
    model : torch.nn.Module
        Model to load into.
    optimizer : torch.optim.Optimizer
        Optimizer to load into.
    file_path : str
        Checkpoint file, or the directory containing ``model_<rank>.pt`` if
        ``model_parallel`` is set.
    rank : int
        Global rank.
    model_parallel : bool
        Select the checkpoint file by ``rank % parallelism``.
    parallelism : int
        Model-parallel size.

    Returns
    -------
    model : torch.nn.Module
        Loaded model.
    optimizer : torch.optim.Optimizer
        Loaded optimizer.
    epoch : int
        Epoch to resume from.
    """
    if model_parallel:
        load_rank = rank % parallelism
        file_path = f"{file_path}/model_{load_rank}.pt"
    try:
        state = torch.load(f"{file_path}", map_location="cpu")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        epoch = state["epoch"] + 1
    except Exception:
        log.warning(
            "could not load model from %s; initialising from scratch", file_path
        )
        epoch = 0
    return model, optimizer, epoch


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
