
import argparse

import torch
import torch.distributed
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
import time

import eval
import utils


def training_loop(config_dict, device, localid, rank):
    model_parallel = config_dict['model']['model_parallel']
    n_parallel = config_dict['model_parallel']['parallelism'] if model_parallel else 1
    
    utils.set_all_seeds(1 + rank%n_parallel)
    
    train_subset, valid_subset = utils.define_subset_sizes(config_dict, rank)
    
    best_val_loss = 1e5
    
    xlat, xlon = config_dict['data']['xlat'], config_dict['data']['xlon']
    
    model_group, ddp_group, model_subgroup_row, model_subgroup_col = utils.setup_groups(rank, n_parallel, 2, 2)

    variable_weights, area_weights = utils.get_weights(config_dict, device, n_parallel, rank)

    model = utils.build_model(config_dict, device, localid, rank, n_parallel, model_group, ddp_group, model_subgroup_col)
    optimizer = utils.build_optimizer(config_dict, model)
        
    lr = config_dict['training']['lr']
    save_path = config_dict['training']['save_dir']

    utils.set_all_seeds_model_group(1, rank=rank, n_parallel=n_parallel) # Set all GPUs to the same seed so that they read the same data
    train_dataloader, valid_dataloader = utils.build_dataloaders(config_dict, model_parallel, rank)
    model, optimizer, epoch = utils.load_model(model, optimizer=optimizer, file_path=config_dict['training']['load_path'], rank=rank, model_parallel=model_parallel, n_parallel=n_parallel)
    loss_fn = utils.build_loss_fn(config_dict)
    
    scheduler = utils.build_scheduler(config_dict, optimizer)
    scheduler.step(epoch)
    
    start_lr = 1e-6
    
    utils.setup_precision(config_dict['training']['tf32'])
    epoch_average_loss = torch.tensor([0]).to(device)

    for epoch in range(epoch, 50):
        start_time = time.time()
        avg_loss = utils.train_one_epoch(
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
            n_vars_local=36,
            max_rollout=1,
            rank=rank,
            lr=lr,
            start_lr=start_lr,
        )

        model.eval()
        # average loss
        epoch_average_loss = avg_loss / world_size
        torch.distributed.all_reduce(epoch_average_loss)
        end_time = time.time()
        if rank == 0:
            epoch_average_loss = epoch_average_loss / len(train_dataloader)
            print(f"Epoch {epoch} time [min]: {((end_time - start_time) / 60 )}, average loss {epoch_average_loss[0]}")

        torch.cuda.synchronize()
        
        val_loss = eval.get_validation_mse(config_dict, model, valid_dataloader, device, rank, epoch, file_path=save_path, model_parallel=model_parallel, world_size=world_size, col_group=col_group, model_group=model_group, ddp_group=ddp_group)

        #if rank < n_parallel and val_loss < best_val_loss:
        #    best_val_loss = val_loss
        #    utils.save_model(model, optimizer, save_path,  rank, epoch)
        #if rank == 0:
        #    print(f"Epoch {epoch} val loss [min]: {val_loss}")

        torch.cuda.synchronize()
        
        scheduler.step(epoch)    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    device, localid, rank, world_size = utils.init_cuda()
    
    training_loop(config_dict, device, localid, rank)
    torch.distributed.destroy_process_group()