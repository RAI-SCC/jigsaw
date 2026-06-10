"""Utility functions to setup distributed environment."""

import os

import torch
import torch.distributed as dist


def set_all_seeds(seed: int = 0) -> None:
    """Initialize all random seeds."""
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_cuda() -> tuple[str, int, int, int]:
    """
    Initialize DistributedDataParallel.

    Returns
    -------
    device: String
    slurm_localid: int
    rank: int
    world_size: int
    """
    rank = int(os.getenv("SLURM_PROCID") or 0)  # Get individual process ID.
    # Get overall number of processes.
    world_size = int(os.getenv("SLURM_NTASKS") or 1)
    slurm_localid = int(os.getenv("SLURM_LOCALID") or 0)

    print(f"Rank {rank}: world_size {world_size}, slurm_localid {slurm_localid}")
    # Initialize GPUs and dataloaders
    device = f"cuda:{slurm_localid}"
    torch.cuda.set_device(slurm_localid)

    # Initialize DistributedDataParallel.
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method="env://"
    )

    if dist.is_initialized():
        print(
            f"Rank {rank}/{world_size}: Process group initialized with torch rank {dist.get_rank()} and torch world size {dist.get_world_size()}."
        )
    else:
        print(f"Rank {rank} not initialized")

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
        if rank == 0:
            print(f"Could not load model from {file_path}. Initializing from scratch.")
        epoch = 0
    return model, optimizer, epoch


def setup_precision(tf32: bool) -> None:
    """Configure matmul and cudnn TF32."""
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    return
