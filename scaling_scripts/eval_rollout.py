import argparse
import time

import torch
import torch.distributed as dist
import torch.optim as optim
import timm
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import vector_to_parameters

import eval
import utils


def training_loop(
    config_dict: dict, device: torch.device, localid: int, rank: int
) -> None:
    """Training loop for MLP Mixer model."""
    n_parallel = config_dict["model_parallel"]["parallelism"]
    model_parallel = n_parallel > 1
    # Set all seeds in the same model group to be different
    utils.set_all_seeds(1 + rank % n_parallel)

    best_val_loss = 1e5

    xlat, xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
    variable_weights, area_weights = utils.get_weights(
        config_dict, device, n_parallel, rank
    )

    # Set up model  groups
    match n_parallel:
        case 1:
            model_group, ddp_group, _, model_subgroup_col = utils.setup_groups(
                rank, n_parallel, 1, 1
            )
            n_vars_local = 69
        case 2:
            model_group, ddp_group, _, model_subgroup_col = utils.setup_groups(
                rank, n_parallel, 1, 2
            )
            n_vars_local = 36
        case 4:
            model_group, ddp_group, _, model_subgroup_col = utils.setup_groups(
                rank, n_parallel, 2, 2
            )
            n_vars_local = 36

    # Define model and load model
    model = utils.build_model(
        config_dict,
        device,
        localid,
        rank,
        n_parallel,
        model_group,
        ddp_group,
        model_subgroup_col,
    )
    optimizer = utils.build_optimizer(config_dict, model)
    model, optimizer, epoch = utils.load_model(
        model,
        optimizer=optimizer,
        file_path=config_dict["training"]["load_path"],
        rank=rank,
        model_parallel=model_parallel,
        n_parallel=n_parallel,
    )
    scheduler = utils.build_scheduler(
        config_dict,
        optimizer,
        epoch,
        n_epochs=config_dict["training"]["n_epochs"],
        lr_min=config_dict["training"]["lr_min"],
    )
    loss_fn = utils.build_loss_fn(config_dict)

    # Set random seed and ensure the same model groups same seeds
    # Set all GPUs to the same seed so that they read the same data
    utils.set_all_seeds_model_group(1, rank=rank, n_parallel=n_parallel)
    _, valid_dataloader = utils.build_dataloaders(
        config_dict, model_parallel, rank
    )

    utils.setup_precision(config_dict["training"]["tf32"])

    max_rollout = config_dict["model"]["latent_rollout"]
    epoch_end = epoch + 11 
    epoch_end = config_dict['training']['n_epochs'] if epoch_end > config_dict['training']['n_epochs'] else epoch_end
    
    model.eval()
    val_loss = eval.validate(
        config_dict,
        model,
        valid_dataloader,
        rank,
        device,
        epoch,
        config_dict["training"]["save_dir"],
        mse_file_path="rmse_rollout",
        acc_file_path="acc_rollout",
        model_parallel=True,
        world_size=world_size,
        model_subgroup_col=model_subgroup_col,
        ddp_group=ddp_group,
        model_group=model_group,
    )

    if rank == 0:
        print(
            f"Epoch {epoch}. Average validation loss: {val_loss}"
        )

        torch.cuda.synchronize()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    device, localid, rank, world_size = utils.init_cuda()

    training_loop(config_dict, device, localid, rank)
    dist.destroy_process_group()
