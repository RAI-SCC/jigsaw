import argparse
import time

import torch
import torch.distributed
import yaml

from torch.nn.parallel import DistributedDataParallel as DDP

import eval
import utils


def training_loop(config_dict, device, localid, rank):
    n_parallel = config_dict['model_parallel']['parallelism'] 
    
    # Set all seeds in the same model group to be different 
    utils.set_all_seeds(1, rank=(rank%n_parallel))
    
    best_val_loss = 1e5
    
    xlat, xlon = config_dict['data']['xlat'], config_dict['data']['xlon']
    variable_weights, area_weights = utils.get_weights(config_dict, device, n_parallel, rank)

    # Set up model  groups
    model_group, ddp_group, _, model_subgroup_col = utils.setup_groups(rank, n_parallel, 1, 2)
    
    # Define model and load model 
    model = utils.build_model(config_dict, device, localid, rank, n_parallel, model_group, ddp_group, model_subgroup_col)
    optimizer = utils.build_optimizer(config_dict, model)
    model, optimizer, epoch = utils.load_model(model, optimizer=optimizer, file_path=config_dict['training']['load_path'], rank=rank, model_parallel=model_parallel, n_parallel=n_parallel)
    scheduler = utils.build_scheduler(config_dict, optimizer, epoch)
        
    # Set random seed and 
    utils.set_all_seeds_local_rank(1, rank=rank, n_parallel=n_parallel) # Set all GPUs to the same seed so that they read the same data           
    train_dataloader, valid_dataloader = utils.build_dataloaders(config_dict, model_parallel, rank)

    utils.setup_precision(config_dict['training']['tf32'])

    max_rollout = config_dict['model']['latent_rollout']

    for epoch in range(epoch, 50):
        avg_loss = utils.train_one_epoch(
            config_dict,
            epoch=epoch,
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            area_weights=area_weights,
            variable_weights=variable_weights,
            xlat=xlat,
            xlon=xlon,
            max_rollout=max_rollout,
            rank=rank
        )

        torch.cuda.synchronize()
        
        model.eval()
        val_loss = utils.validate(
            config_dict,
            model,
            valid_dataloader,
            rank,
            device,
            epoch,
            config_dict['training']['save_dir'],
            model_parallel,
            world_size,
            model_subgroup_col,
            model_group
        )

        if rank < n_parallel and val_loss < best_val_loss:
            best_val_loss = val_loss
            utils.save_model(model, optimizer, config_dict['training']['save_dir'],  rank, epoch)
        if rank == 0:
            print(f"Epoch {epoch} val loss: {val_loss}")

        torch.cuda.synchronize()
        scheduler.step(epoch)    
        

if __name__ == '__main__':
    model_parallel = False
    train_subset = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    device, localid, rank, world_size = utils.init_CUDA()
    
    training_loop(config_dict, device, localid, rank)
    torch.distributed.destroy_process_group()