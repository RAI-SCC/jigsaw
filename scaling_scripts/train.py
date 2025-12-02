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
    
    for name, param in model.named_parameters():
        if param is not None:
            if "bias" in name:
                dist.all_reduce(param.data, op=dist.ReduceOp.AVG, group=model_subgroup_col)
                
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
    train_dataloader, valid_dataloader = utils.build_dataloaders(
        config_dict, model_parallel, rank
    )

    utils.setup_precision(config_dict["training"]["tf32"])

    max_rollout = config_dict["model"]["latent_rollout"]
    epoch_end = epoch + 11 
    epoch_end = config_dict['training']['n_epochs'] if epoch_end > config_dict['training']['n_epochs'] else epoch_end
    
    for epoch in range(epoch, epoch_end):
        avg_loss = train_one_epoch(
            config_dict,
            epoch=epoch,
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            area_weights=area_weights,
            variable_weights=variable_weights,
            xlat=xlat,
            xlon=xlon,
            n_vars_local=n_vars_local,
            max_rollout=max_rollout,
            rank=rank,
            start_lr=1e-6,
            col_group=model_subgroup_col
        )

        torch.cuda.synchronize()

        model.eval()
        val_loss = eval.validate(
            config_dict,
            model,
            valid_dataloader,
            rank,
            device,
            epoch,
            config_dict["training"]["save_dir"],
            mse_file_path="rmse.csv",
            acc_file_path="acc.csv",
            model_parallel=True,
            world_size=world_size,
            model_subgroup_col=model_subgroup_col,
            ddp_group=ddp_group,
            model_group=model_group,
        )

        if rank < n_parallel:
            utils.save_model(
                model,
                optimizer,
                scheduler,
                config_dict["training"]["save_dir"],
                rank,
                epoch,
            )
        if rank == 0:
            print(
                f"Epoch {epoch}. Average training loss {avg_loss}. Average validation loss: {val_loss}"
            )

        torch.cuda.synchronize()
        scheduler.step(epoch)


def train_one_epoch(
    config_dict: dict,
    epoch: int,
    model: torch.nn.Module,
    train_dataloader,
    optimizer: optim.Optimizer,
    scheduler: timm.scheduler,
    loss_fn: torch.nn.modules.loss,
    device: torch.device,
    area_weights: torch.Tensor,  # [1, xlat, 1, 1]
    variable_weights: torch.Tensor,  # [36]
    xlat: int,
    xlon: int,
    n_vars_local: int,
    max_rollout: int,
    rank: int,
    start_lr: float,
    col_group: dist.ProcessGroup
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

    if hasattr(train_dataloader, "sampler") and hasattr(
        train_dataloader.sampler, "set_epoch"
    ):
        train_dataloader.sampler.set_epoch(epoch)

    start_time = time.perf_counter()
    if config_dict["model_parallel"]["parallelism"] == 4:
        xlon = xlon // 2

    if rank == 0:
        print(f"Epoch {epoch}, learning rate {scheduler._get_lr(epoch)}")

    for i, data in enumerate(train_dataloader):
        rollout = i % max_rollout

        # warmup lr only in epoch 0
        if epoch == 0 and i < 5000:
            max_values = scheduler._get_lr(epoch)
            for index, g in enumerate(optimizer.param_groups):
                warm = i * (max_values[index] - start_lr) / 5000 + start_lr
                g["lr"] = warm
        
        input_data = data[0].to(device, non_blocking=True)
        target_data = data[1][rollout].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        output = model(input_data, rollout_=1)  # [B, 720, 720, 36]
        output = output.reshape(-1, xlat, xlon, n_vars_local)

        # broadcast weights
        vw = variable_weights.view(1, 1, 1, n_vars_local)
        ow = output * area_weights * vw
        tw = target_data * area_weights * vw
        
        match config_dict["model_parallel"]["parallelism"]:
            case 4:
                if rank in [1,3]:
                    loss = loss_fn(ow[...,:33], tw[...,:33])
                    loss = loss * 33 / 69 * 0.5 * 4
                else:
                    loss = loss_fn(ow, tw)
                    loss = loss * 39 / 69 * 0.5 * 4
            case _:
                loss = loss_fn(ow, tw)
            
        loss.backward()
        
        dist.all_reduce(loss)
        
        loss = loss / config_dict["model_parallel"]["parallelism"]
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config_dict["training"]["grad_clip"]
        )
        
        if config_dict['model_parallel']['parallelism'] == 4:
            handles, flat_grad, gradients = utils.reduce_ddp_layernorms(model, col_group)
            handles.wait()
            vector_to_parameters(flat_grad, gradients)
        
        optimizer.step()
            
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        epoch_loss = epoch_loss + float(loss.detach())
        if rank == 0:
            print(
                f"{epoch}, after Step {i}/{len(train_dataloader)}. Rollout {rollout}. loss: {float(loss.detach())}"
            )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    epoch_average_loss = epoch_loss / max(1, len(train_dataloader))
    if rank == 0:
        print(
            f"Epoch {epoch} time [min]: {(end_time - start_time) / 60.0}, average loss {epoch_average_loss}"
        )

    scheduler.step(epoch)

    return epoch_average_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    device, localid, rank, world_size = utils.init_cuda()

    training_loop(config_dict, device, localid, rank)
    dist.destroy_process_group()
