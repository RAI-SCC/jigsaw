import argparse
import time
from typing import Dict, Tuple

import torch
import torch.distributed
import torch.optim as optim
import yaml
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP

import Dataloaders
import eval
import Model
import utils

# ----------------------------
# Setup helpers
# ----------------------------

def setup_seeds(config_dict: Dict, rank: int, n_parallel: int) -> None:
    """
    Set all random seeds. Ensures ranks 0, 4, ... share initial params when using model parallel.
    """
    if n_parallel > 1:
        utils.set_all_seeds_local_rank(1, rank=rank, n_parallel=n_parallel)
    else:
        utils.set_all_seeds(1, rank=rank)


def build_variable_weights(
    config_dict: Dict, device: torch.device, rank: int, n_parallel: int
) -> torch.Tensor:
    """
    Build variable weights and split by model-parallel shard.

    Returns:
        variable_weights: [36] on each rank's shard (float32 on device)
    """
    # surface: [4], pressure: [5 vars x 13 levels] -> 65, total 69 then later padded to 72
    surface_weights = torch.tensor([1.5, 0.77, 0.77, 3.0]) * 4
    pressure_weights = (
        torch.tensor([3.00, 0.6, 1.5, 0.9, 0.9])
        .unsqueeze(1)
        .expand(5, 13)
        .reshape(-1)
    )
    surface_upper_pressure_weights = torch.cat(
        [
            torch.tensor([1, 1, 1, 1]),  # surface variables
            torch.tensor([1, 1, 1, 1, 1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]).repeat(5),
        ]
    ).view(1, 69)

    variable_weights = (
        torch.cat([surface_weights, pressure_weights])
        .view(1, 1, 1, 69)
        * surface_upper_pressure_weights
    ).to(device)

    # align to 72 then split first-half/second-half 36 based on model-parallel shard
    variable_weights = torch.nn.functional.pad(variable_weights, (0, 3))
    shard = variable_weights[0, 0, 0, :36] if (rank % n_parallel) == 0 else variable_weights[0, 0, 0, 36:]
    return shard.to(torch.float32).contiguous()

def build_area_weights(config_dict: Dict, device: torch.device) -> torch.Tensor:
    """
    Build area weights.

    Returns:
        area_weights: [1, xlat, 1, 1] float32 on device
    """
    xlat = config_dict["data"]["xlat"]
    area_weights = torch.tensor(eval.calc_weight(xlat)).view(1, xlat, 1, 1).to(device).to(torch.float32)
    return area_weights

def build_scheduler(config_dict: Dict, optimizer: optim.Optimizer, start_epoch: int):
    """
    Build LR scheduler and step it to start_epoch.
    """
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
    """
    Build train and validation dataloaders.
    """
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


def load_state_dict(
    config_dict: Dict, mlp:  torch.nn.Module, optimizer: optim.Optimizer, rank: int, model_parallel: bool, n_parallel: int
) -> Tuple[torch.nn.Module, optim.Optimizer, int]:
    """
    Try to load model state. Return (mlp, optimizer, start_epoch).
    """
    try:
        file_path = config_dict["training"]["load_path"]
        mlp, optimizer, epoch = utils.load_model(
            mlp, optimizer=optimizer, file_path=file_path, rank=rank, model_parallel=model_parallel, parallelism=n_parallel
        )
        return mlp, optimizer, epoch
    except Exception as err:
        if rank == 0:
            print(err)
            print("Initializing model from scratch")
        return mlp, optimizer, 0


# ----------------------------
# Training and validation
# ----------------------------

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
    local_groups,
    node_groups,
    global_groups,
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
        local_groups=local_groups,
        node_groups=node_groups,
        global_groups=global_groups,
    )
    return float(val_loss)


def train_one_epoch(
    epoch: int,
    mlp: torch.nn.Module,
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
    mlp.train()
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

        output = mlp(input_data, rollout_=rollout + 1)  # [B, 720, 720, 36]
        output = output.reshape(output.shape[0], xlat, xlon, 36)

        # broadcast weights
        vw = variable_weights.view(1, 1, 1, 36)
        ow = output * area_weights * vw
        tw = target_data * area_weights * vw

        loss = torch.nn.functional.mse_loss(ow, tw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), config_dict["training"]["grad_clip"])
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
    return avg


# ----------------------------
# Orchestrator
# ----------------------------

def training_loop(config_dict: Dict, device, localid: int, rank: int, world_size: int) -> None:
    """
    Main training loop.
    """
    model_parallel = config_dict["model"]["model_parallel"]
    n_parallel = config_dict["model_parallel"]["parallelism"] if model_parallel else 1

    setup_seeds(config_dict, rank, n_parallel)
    setup_precision(config_dict["training"]["tf32"])

    if rank == 0:
        ts = config_dict["data"]["train_subset"]
        vs = config_dict["data"]["valid_subset"]
        print(f"Training subset size {None if ts < 0 else ts}")
        print(f"Reading from {config_dict['data']['train_data_path']}")

    xlat, xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
    variable_weights = build_variable_weights(config_dict, device, rank, n_parallel)  # [36]
    area_weights = build_area_weights(config_dict, device)  # [1, xlat, 1, 1]

    local_group, local_groups, node_groups, ddp_group, global_groups = setup_groups(rank, n_parallel)

    mlp = build_model(config_dict, device, localid, rank, n_parallel, local_group, ddp_group)
    if rank == 0:
        print(f"model has {utils.count_parameters(mlp) * n_parallel / 1e6} million parameters")

    optimizer = build_optimizer(config_dict, mlp)
    mlp, optimizer, epoch = load_state_dict(config_dict, mlp, optimizer, rank, model_parallel, n_parallel)
    mlp = mlp.to(device)

    scheduler = build_scheduler(config_dict, optimizer, epoch)
    
    train_dataloader, valid_dataloader = build_dataloaders(config_dict, model_parallel, rank)

    best_val_loss = 1e5
    max_rollout = config_dict["model"]["latent_rollout"]
    lr = config_dict["training"]["lr"]
    start_lr = 1e-6
    save_path = config_dict["training"]["save_dir"]

    for epoch_idx in range(epoch, 50):
        avg_loss = train_one_epoch(
            epoch=epoch_idx,
            mlp=mlp,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            area_weights=area_weights,
            variable_weights=variable_weights,
            xlat=xlat,
            xlon=xlon,
            max_rollout=max_rollout,
            rank=rank,
            lr=lr,
            start_lr=start_lr,
        )

        mlp.eval()
        val_loss = validate(
            config_dict,
            mlp,
            valid_dataloader,
            rank,
            device,
            epoch_idx,
            save_path,
            model_parallel,
            world_size,
            local_groups,
            node_groups,
            global_groups,
        )

        if rank < n_parallel and val_loss < best_val_loss:
            best_val_loss = val_loss
            utils.save_model(mlp, optimizer, save_path, rank, epoch_idx)
        if rank == 0:
            print(f"Epoch {epoch_idx} val loss: {val_loss}")

        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    return parser.parse_args()


def load_config(path: str) -> Dict:
    """
    Load YAML config.
    """
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


if __name__ == "__main__":
    args = parse_args()
    config_dict = load_config(args.config_file)

    device, localid, rank, world_size = utils.init_CUDA()
    training_loop(config_dict, device, localid, rank, world_size)
    torch.distributed.destroy_process_group()
