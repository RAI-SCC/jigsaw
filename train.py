
import numpy as np
import torch
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
import sys
sys.path.insert(1, '/hkfs/work/workspace/scratch/ke4365-summa/')
sys.path.insert(1, '/hkfs/work/workspace/scratch/ke4365-summa/src')
import utils, eval
import TestModel
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import yaml


def training_loop(config_dict, device, slurm_localid, rank):
    comm = None # TODO

    model_parallel = config_dict['model']['model_parallel']
    train_subset, valid_subset = config_dict['data']['train_subset'], config_dict['data']['valid_subset']
    if train_subset < 0:
        train_subset = None
    if valid_subset < 0:
        valid_subset = None
    train_dataloader = utils.get_dataloader(config_dict, config_dict['data']['train_data_path'], mode='train', distributed=(not model_parallel), batch_size=1, subset_size=train_subset)
    valid_dataloader = utils.get_dataloader(config_dict, config_dict['data']['valid_data_path'], mode='validation', distributed=(not model_parallel), batch_size=8, subset_size=valid_subset)


    if not model_parallel:
        mlp = TestModel.ModelSequential(config_dict, device, rank).to(torch.float32).to(device)

        mlp_ddp = DDP(
            mlp,
            device_ids=[slurm_localid],
            output_device=slurm_localid
        )
    else:
        mlp = TestModel.Model(device, comm, rank).to(torch.float32).to(device)

    lr = config_dict['training']['lr']
    save_path = config_dict['training']['save_dir']
    
    # optimizer = optim.Adam(mlp_ddp.parameters(), lr=lr, weight_decay=3e-6)
    if config_dict['training']['preload']:
        optimizer = optim.Adam([
            {'params': mlp_ddp.module.params.embedding.parameters(), 'lr': config_dict['training']['lr_embedding']},
            {'params': mlp_ddp.module.params.recovery.parameters(), 'lr': config_dict['training']['lr_recovery']},
            {'params': mlp_ddp.module.params.common.parameters()}
            ], 
            lr=lr, weight_decay=3e-6
        )
    else:
        optimizer = optim.Adam(mlp_ddp.parameters(), lr=lr, weight_decay=3e-6)

    
    if slurm_localid == 0:
        print(f"model has {utils.count_parameters(mlp_ddp)/1e6} million parameters")

    # load
    try:
        mlp_ddp, optimizer, epoch = utils.load_model(mlp_ddp, optimizer=optimizer, file_path=save_path, rank=rank, model_parallel=model_parallel)
        epoch = epoch + 1
        
    except Exception as err:
        print(err)
        epoch = 0
        if slurm_localid == 0:
            print("Initializing model from scratch")
        pass
    
    try:
        if slurm_localid == 0 and config_dict['training']['preload']:
            print("loading pretrained convolutional weights")

        if config_dict['training']['preload'] and epoch == 0:
            if config_dict['training']['preload_embedding']:
                mlp_ddp = utils.load_pretrained_embedding(mlp_ddp, config_dict)
            else: # Loading from pangu model
                mlp_ddp = utils.load_pretrained_convolution(mlp_ddp, config_dict)
            
    except Exception as err:
        if slurm_localid == 0:
            print(f"Unable to load pretrained convolutional weights {err}")
    mlp_ddp = mlp_ddp.to(device)
    # Completely DDP model, no model parallelism

    loss_fn = torch.nn.MSELoss()

    surface_weights  = torch.tensor([1.5, 0.77, 0.66, 3.0])*.25
    pressure_weights = torch.tensor([3.00, 0.6, 1.5, 0.77, 0.54]).unsqueeze(1).expand(5, 13).flatten()
    weights = torch.concat([surface_weights, pressure_weights]).view(1, 69).to(device)
    xlat = config_dict['data']['xlat']
       
    scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-5, cycle_decay=0.5, cycle_mul=1, cycle_limit=10)
    scheduler.step(epoch)
    start_lr = 1e-6

    for epoch in range(epoch, 100):
        if rank == 0:
            print(f"Epoch {epoch}, learning rate {scheduler._get_lr(epoch)}")
        start_time = time.perf_counter()
        mlp.train()
        for i, data in enumerate(train_dataloader):
        ## Get the batch data on rank 0
            if epoch == 0:
                for g in optimizer.param_groups:
                    g['lr'] = i * (lr-start_lr) / len(train_dataloader) + start_lr
            input_data  = data[0].to(torch.float32)#.to(device)
            target_data = data[1].to(torch.float32).to(device)
            
            optimizer.zero_grad()
            output = mlp_ddp(input_data)

            # output: n_batch, 720, 1440, 72//4
            # target data shape: (n_batch, 69, 720, 1440)        
            target_data = target_data.permute(0, 2, 3, 1)
            # target data: [n_batch, 720, 1440, 69]
            
            if model_parallel:
                fraction_1 = 720*1440//2
                fraction_2 = 72 // 2
                target_data = target_data.reshape(target_data.shape[0], -1, target_data.shape[3])
                target_data = torch.nn.functional.pad(target_data, (0,3)) 
                if rank == 0:
                    loss = loss_fn(output, target_data[:, :fraction_1,  :fraction_2])
                elif rank == 1:
                    loss = loss_fn(output, target_data[:, :fraction_1,  fraction_2:])
                elif rank == 2:
                    loss = loss_fn(output, target_data[:, fraction_1:,  fraction_2:])
                elif rank == 3:
                    loss = loss_fn(output, target_data[:, :fraction_1,  :fraction_2])

            else:
                loss = loss_fn(output*weights, target_data*weights)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()

            torch.distributed.all_reduce(loss)
            loss = loss / world_size
            if rank == 0:
                print(f"{rank}:  {epoch}, after Step {i}/{len(train_dataloader)}. loss: {loss.detach()}")

        end_time = time.perf_counter()
            
        mlp.eval()
        if rank == 0:
            print(f"Epoch {epoch} time [min]: {(end_time - start_time)/60.}")
        eval.get_validation_mse(config_dict, mlp_ddp, valid_dataloader, device, rank, epoch, file_path=save_path, model_parallel=model_parallel)
    
        if not model_parallel:
            if rank == 0:
                utils.save_model(mlp_ddp, optimizer, save_path, rank, epoch)
            else:
                pass
        else:
            pass
            #utils.save_model(mlp_ddp, optimizer, save_path,  rank, epoch)

        scheduler.step(epoch)    
        torch.distributed.barrier()
                

if __name__ == '__main__':
    model_parallel = False
    train_subset = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    utils.set_all_seeds(1)

    device, slurm_localid, rank, world_size = utils.init_CUDA()

    training_loop(config_dict, device, slurm_localid, rank)
    torch.distributed.destroy_process_group()