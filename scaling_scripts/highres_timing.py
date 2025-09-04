
import numpy as np
import torch
import torch.distributed
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
import utils
import eval
import Model
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import yaml


def training_loop(config_dict, device, localid, rank):
    """Training loop for scaling experiments."""
    model_parallel = config_dict['model']['model_parallel']
    n_parallel = config_dict['model_parallel']['parallelism'] if model_parallel else 1

    utils.set_all_seeds(1 + rank % n_parallel)

    xlat, xlon = config_dict['data']['xlat'], config_dict['data']['xlon']

    # Set up model  groups
    match n_parallel:
        case 1:
            model_group, ddp_group, _, model_subgroup_col = utils.setup_groups(
                rank, n_parallel, 1, 1)
            n_vars_out = 69
        case 2:
            model_group, ddp_group, _, model_subgroup_col = utils.setup_groups(
                rank, n_parallel, 1, 2)
            n_vars_out = 36
        case 4:
            model_group, ddp_group, _, model_subgroup_col = utils.setup_groups(
                rank, n_parallel, 2, 2)
            xlon = xlon//2
            n_vars_out = 36

    # Set up model, optimizer, loss function
    model = utils.build_model(config_dict, device, localid, rank,
                              n_parallel, model_group, ddp_group, model_subgroup_col)
    optimizer = utils.build_optimizer(config_dict, model)
    loss_fn = utils.build_loss_fn(config_dict)

    # Set all GPUs to the same seed so that they read the same data
    utils.set_all_seeds_model_group(1, rank=rank, n_parallel=n_parallel)

    config_dict['data']['xlat'] = config_dict['data']['xlat'] // 2
    config_dict['data']['xlon'] = config_dict['data']['xlon'] // 2
    train_dataloader, _ = utils.build_dataloaders(
        config_dict, model_parallel, rank)
    config_dict['data']['xlat'] = config_dict['data']['xlat'] * 2
    config_dict['data']['xlon'] = config_dict['data']['xlon'] * 2

    start_lr = 1e-6

    utils.setup_precision(config_dict['training']['tf32'])
    epoch_times = np.zeros(11)
    torch.cuda.synchronize()

    for epoch in range(11):
        train_dataloader.sampler.set_epoch(epoch)

        model.train()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        train_one_epoch(model, train_dataloader, optimizer,
                        loss_fn, xlat, xlon, n_vars_out)
        end.record()
        torch.cuda.synchronize()

        if rank == 0:
            epoch_times[epoch] = start.elapsed_time(end) / 1000 / 60
            print(f"Epoch {epoch} time [min]: {(epoch_times[epoch])}")

    if rank == 0:
        print(f'Average epoch time is {np.average(epoch_times[1:])}')
        print(f'Std epoch time is {np.std(epoch_times[1:])}')


def train_one_epoch(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim,
    loss_fn: torch.nn.modules.loss,
    xlat: int,
    xlon: int,
    n_vars_out: int
):
    """Time and train one epoch."""
    for data in train_dataloader:
        # Loading data: Always with async ops and handles
        input_data = data[0].to(device, non_blocking=True)
        target_data = data[1][0].to(device, non_blocking=True)
        target_data.unsqueeze(dim=0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # output shape [n_batch, 720, 720, 36]
        output = model(input_data, rollout_=0+1)

        # Reshape output
        output = (output.reshape(output.shape[0], xlat, xlon, n_vars_out))
        loss = loss_fn(output, target_data)
        loss.backward()


if __name__ == '__main__':
    model_parallel = False
    train_subset = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    device, localid, rank, world_size = utils.init_cuda()

    training_loop(config_dict, device, localid, rank)
    torch.distributed.destroy_process_group()
