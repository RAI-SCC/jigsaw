import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP

import Dataloaders
import Model


def get_ddp_groups(rank: int, n_parallel: int) -> dist.ProcessGroup:
    """Return DDP groups."""
    n_nodes = torch.distributed.get_world_size() // n_parallel
    global_group_list = []
    global_groups     = []

    for node in np.arange(n_parallel):
        global_group_list.append(((np.arange(n_nodes)*n_parallel+node)).tolist())
               
    for global_group in global_group_list:
        global_groups.append(torch.distributed.new_group(ranks=global_group))
    
    return global_groups[rank%n_parallel]

def get_model_group(rank, n_parallel):
    """Return model groups."""
    local_groups = []
    for i in range(torch.distributed.get_world_size() // n_parallel):
        local_start_rank = i * n_parallel
        local_groups.append(
                torch.distributed.new_group(
                    ranks=list(range(local_start_rank, local_start_rank+n_parallel))
                )
        )
    
    return local_groups[rank//n_parallel]

def get_model_group_rows(rank, n_parallel, n_rows):
    """
    Return mpi comm subgroups within model groups. 

    rank: int
        global rank of process
    n_parallel: int
        number of processes in model-parallel group
    n_rows: int
        number of rows that the model-parallel group is subdivided into.
        default for 4: 2; default for 8: 4
    """
    groups = []
    n_cols = n_parallel // n_rows
    n_nodes = torch.distributed.get_world_size() // n_parallel

    for node in range(n_nodes):
        base = node * n_parallel
        for row in range(n_rows):
            ranks = [base + row * n_cols + c for c in range(n_cols)]
            groups.append(dist.new_group(ranks=ranks))

    node = rank // n_parallel
    local = rank % n_parallel
    row = local // n_cols
    idx = node * n_rows + row
    return groups[idx]

def get_model_group_columns(rank, n_parallel, n_cols):
    """
    Return mpi comm subgroups within model groups. 

    rank: int
        global rank of process
    n_parallel: int
        number of processes in model-parallel group
    n_cols: int
        number of rows that the model-parallel group is subdivided into.
        default for 4: 2; default for 8: 2
    """
    groups = []
    n_rows = n_parallel // n_cols
    n_nodes = torch.distributed.get_world_size() // n_parallel

    for node in range(n_nodes): # [0, 2], [1, 3], [4, 6], [5, 7]
        base = node * n_parallel
        for col in range(n_cols):
            ranks = [base + r * n_cols + col for r in range(n_rows)]
            groups.append(torch.distributed.new_group(ranks=ranks)
        )
            
    node = rank // n_parallel
    local = rank % n_parallel
    col = local % n_cols
    idx = node * n_cols + col

    return groups[idx]

def set_all_seeds(seed: int = 0, rank: Optional[int] = None) -> None:
    """Initialize all random seeds"""
    if rank is None:
        rank = 0
    os.environ["PL_GLOBAL_SEED"] = str(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

def set_all_seeds_local_rank(seed: int = 0, rank: Optional[int] = None, n_parallel: int = None) -> None:
    """Set all seeds on all model groups to the same value."""
    if rank is None:
        rank = 0
    set_all_seeds(seed + rank // n_parallel, rank=0)



def init_CUDA():
    """
    Initialize DistributedDataParallel. 
    
    Returns
    -------
    device: String
    slurm_localid: int
    rank: int
    world_size: int
    """
    rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
    world_size = int(os.getenv("SLURM_NTASKS")) # Get overall number of processes.
    slurm_localid = int(os.getenv("SLURM_LOCALID"))

    print(f"Rank {rank}: world_size {world_size}, slurm_localid {slurm_localid}")
    # Initialize GPUs and dataloaders
    device = f"cuda:{slurm_localid}"
    torch.cuda.set_device(slurm_localid)
    
    # Initialize DistributedDataParallel.
    dist.init_process_group(backend="nccl", 
                            rank=rank, 
                            world_size=world_size, 
                            init_method="env://")
    
    if dist.is_initialized(): 
        print(f"Rank {rank}/{world_size}: Process group initialized with torch rank {torch.distributed.get_rank()} and torch world size {torch.distributed.get_world_size()}.")
    else:
        print(f"Rank {rank} not initialized")

    return device, slurm_localid, rank, world_size

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# save model
def save_model(model, optimizer, file_path, rank, epoch, distributed=True):
    file_path = f"{file_path}/model_{rank}.pt"
    state = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, file_path)

def load_model(model, optimizer, file_path, rank, model_parallel=False, n_parallel=4):
    if not model_parallel:
        rank = 0
    if model_parallel:
        load_rank = rank % n_parallel
        file_path = f'{file_path}/model_{load_rank}.pt'
    try:
        state = torch.load(f"{file_path}", map_location='cpu')
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])

        epoch = state['epoch']
    except Exception:
        print(f'Could not load model from {file_path}. Initializing from scratch.')
        epoch = 0
    return model, optimizer, epoch

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    P = torch.tensor(P).to(torch.float32)
    return P

def setup_precision(tf32: bool) -> None:
    """Configure matmul and cudnn TF32."""
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)

def setup_groups(rank: int, n_parallel: int, nrows: int, ncols: int):
    """Return communication groups."""
    model_group = get_model_group(rank, n_parallel)
    ddp_group = get_ddp_groups(rank, n_parallel)
    model_subgroup_row = get_model_group_rows(rank, n_parallel, nrows)
    model_subgroup_col = get_model_group_columns(rank, n_parallel, ncols)
    return model_group, ddp_group, model_subgroup_row, model_subgroup_col

def build_model(
    config_dict: Dict,
    device: torch.device,
    localid: int,
    rank: int,
    n_parallel: int,
    model_group,
    ddp_group,
) -> torch.nn.Module:
    """Instantiate model and wrap with DDP."""
    mlp = Model.Model_twoway(config_dict, rank, device, n_parallel, model_group).to(device)
    mlp = DDP(mlp, device_ids=[localid], output_device=localid, process_group=ddp_group)
    return mlp


def build_optimizer(config_dict: Dict, mlp: torch.nn.Module) -> optim.Optimizer:
    """Build Adam optimizer with per-block lrs."""
    lr = config_dict["training"]["lr"]
    opt = optim.Adam(
        [
            {"params": mlp.module.params.embedding.parameters(), "lr": config_dict["training"]["lr_embedding"]},
            {"params": mlp.module.params.recovery.parameters(), "lr": config_dict["training"]["lr_recovery"]},
            {"params": mlp.module.params.common.parameters()},
        ],
        lr=lr,
        weight_decay=3e-6,
    )
    return opt

def build_scheduler(config_dict: Dict, optimizer: optim.Optimizer, start_epoch: int = 0):
    """Build LR scheduler and step it to start_epoch."""
    if config_dict["training"]["lr_constant"]:
        scheduler = StepLRScheduler(optimizer, decay_rate=1.0, decay_t=100)
    else:
        scheduler = CosineLRScheduler(
            optimizer, t_initial=30, lr_min=5e-6, cycle_decay=1, cycle_mul=1, cycle_limit=10
        )
    scheduler.step(start_epoch)
    return scheduler

def build_dataloaders(
    config_dict: Dict, model_parallel: bool, rank: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train and validation dataloaders."""
    train_subset = config_dict["data"]["train_subset"]
    valid_subset = config_dict["data"]["valid_subset"]
    train_subset = None if train_subset < 0 else train_subset
    valid_subset = None if valid_subset < 0 else valid_subset

    train_dataloader = Dataloaders.get_dataloader(
        config_dict,
        config_dict["data"]["train_data_path"],
        mode="train",
        distributed=(not model_parallel),
        batch_size=config_dict["training"]["train_batch"],
        subset_size=train_subset,
        rank=rank,
    )
    valid_dataloader = Dataloaders.get_dataloader(
        config_dict,
        config_dict["data"]["valid_data_path"],
        mode="validation",
        distributed=(not model_parallel),
        batch_size=config_dict["training"]["valid_batch"],
        subset_size=valid_subset,
        rank=rank,
    )
    return train_dataloader, valid_dataloader

@torch.no_grad()
def validate(
    config_dict: Dict,
    mlp: torch.nn.Module,
    valid_dataloader,
    rank: int,
    device: torch.device,
    epoch: int,
    save_path: str,
    model_parallel: bool,
    world_size: int,
    model_subgroup_col,
    model_group
) -> float:
    """
    Run validation and return scalar loss.
    """
    val_loss = eval.get_validation_mse(
        config_dict,
        mlp,
        valid_dataloader,
        rank,
        device,
        epoch,
        file_path=save_path,
        model_parallel=model_parallel,
        world_size=world_size,
        model_subgroup_col=model_subgroup_col,
        model_group=model_group
    )
    return float(val_loss)


def train_one_epoch(
    config_dict: Dict,
    epoch: int,
    model: torch.nn.Module,
    train_dataloader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    area_weights: torch.Tensor,       # [1, xlat, 1, 1]
    variable_weights: torch.Tensor,   # [36]
    xlat: int,
    xlon: int,
    max_rollout: int,
    rank: int,
    lr: float,
    start_lr: float,
) -> float:
    """
    Train for one epoch.

    Tensor shapes:
        input_data: [B, T, H, W, C] or as produced by Dataloaders
        output:     [B, 720, 720, 36] -> reshaped to [B, xlat, xlon, 36]
        target_data: list-like per rollout; target_data = data[1][rollout]
    """
    model.train()
    epoch_loss = 0.0

    if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
        train_dataloader.sampler.set_epoch(epoch)

    start_time = time.perf_counter()

    for i, data in enumerate(train_dataloader):
        rollout = i % max_rollout

        # warmup lr only in epoch 0
        if epoch == 0:
            warm = i * (lr - start_lr) / max(1, len(train_dataloader)) + start_lr
            for g in optimizer.param_groups:
                g["lr"] = warm

        input_data = data[0].to(device, non_blocking=True)
        target_data = data[1][rollout].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        output = model(input_data, rollout_=rollout + 1)  # [B, 720, 720, 36]
        output = output.reshape(output.shape[0], xlat, xlon, 36)

        # broadcast weights
        vw = variable_weights.view(1, 1, 1, 36)
        ow = output * area_weights * vw
        tw = target_data * area_weights * vw

        loss = torch.nn.functional.mse_loss(ow, tw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config_dict["training"]["grad_clip"])
        optimizer.step()

        epoch_loss += float(loss.detach())
        if rank == 0:
            print(f"{epoch}, after Step {i}/{len(train_dataloader)}. Rollout {rollout}. loss: {float(loss.detach())}")

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg = epoch_loss / max(1, len(train_dataloader))
    if rank == 0:
        print(f"Epoch {epoch} time [min]: {(end_time - start_time) / 60.0}, average loss {avg}")

    scheduler.step(epoch)

    if rank == 0:
        epoch_average_loss = epoch_average_loss / len(train_dataloader)
        print(f"Epoch {epoch} time [min]: {((end_time - start_time) / 60 )}, average loss {epoch_average_loss[0]}")

    return avg