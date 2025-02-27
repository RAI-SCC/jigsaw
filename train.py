
import numpy as np
import torch
import torch.distributed
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
import torch.profiler

import sys
sys.path.insert(1, '/hkfs/work/workspace/scratch/ke4365-summa/')
sys.path.insert(1, '/hkfs/work/workspace/scratch/ke4365-summa/src')
import utils, eval
import Model
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import yaml
import matplotlib.pyplot as plt
import os, psutil
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def training_loop(config_dict, device, localid, rank):

    
    model_parallel = config_dict['model']['model_parallel']
    parallelism = config_dict['model_parallel']['parallelism'] if model_parallel else 1
    
    utils.set_all_seeds(1, rank=(rank%parallelism)) if parallelism > 1 else utils.set_all_seeds(1, rank=None) # ensure ranks 0, 4, ...8... have the same initial parameters
    
    train_subset, valid_subset = config_dict['data']['train_subset'], config_dict['data']['valid_subset']
    train_subset = None if train_subset < 0 else train_subset
    valid_subset    = None if valid_subset < 0 else valid_subset
    
    best_val_loss = 1e5
    
    xlat, xlon = config_dict['data']['xlat'], config_dict['data']['xlon']
    #surface_weights  = torch.tensor([1.5, 0.77, 0.77, 3.0]) * 4
    #pressure_weights = torch.tensor([3.00, 0.6, 1.5, 0.9, 0.9]).unsqueeze(1).expand(5, 13).flatten()
    surface_weights = torch.tensor([0.6, 0.8, 0.8, 3]) * 4
    pressure_weights = torch.tensor([0.6, 0.5, 1.5, 0.8, 0.8]).unsqueeze(1).expand(5, 13).flatten()
    surface_upper_pressure_weights = torch.concat(
        [torch.tensor([1, 1, 1, 1]), # surface variables
        torch.tensor([1, 1, 1, 1, 1, 1, 0.8, 0.1, .7, .1, .6, .1, .5]).repeat(5)]
    ).view(1, 69)

    variable_weights = ((torch.concat([surface_weights, pressure_weights]).view(1, 1, 1, 69))*surface_upper_pressure_weights).to(device)
    
    # underground mask
    weights = torch.load('/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/kangu_finetuning/finetuning_weights/feature_weights.pt', map_location='cpu')
    underground_mask = weights['pressure_mask_lower'] # shape [5, 724, 1440]
    underground_mask = underground_mask.permute(1, 2, 0)[1:-2][:xlat].unsqueeze(-1).expand(-1, -1, -1, 13).reshape(1, xlat, xlon, 65) # shape is [1, xlat, xlon, 65]

    area_weights = torch.tensor(eval.calc_weight(xlat)).view(1, xlat, 1, 1).to(device)

    if parallelism == 1:
        local_groups = None
        node_groups = None
        global_groups = None
        underground_mask = underground_mask.to(device)
    elif parallelism == 2:
        local_group = utils.get_local_groups_twoway(rank)
        local_groups = None
        node_groups = local_group
        global_groups = utils.get_ddp_groups(parallelism)
        variable_weights = torch.nn.functional.pad(variable_weights, (0,3))
        variable_weights = variable_weights[0,0,0,:36] if rank % parallelism == 0 else variable_weights[0,0,0,36:]
        underground_mask = torch.nn.functional.pad(underground_mask, (0,3)) # [1, xlat, xlon, 68]
        underground_mask = underground_mask[:,:,:,:32] if rank % parallelism == 0 else underground_mask[:,:,:,32:]
        underground_mask = underground_mask.to(device)
        node_rank     = rank // parallelism
        
    elif  parallelism == 4:
        local_groups  = utils.get_local_groups_fourway(rank) # groups are [[0, 2], [1, 3], [4, 6], [5, 7]]
                                                         # group [0, 2]: index 0. group [1, 3]: index 1. group [4, 6]: index 3. 
                                                         # index = rank % 2 + node_rank    
        global_groups = utils.get_ddp_groups(parallelism)
        node_groups   = utils.get_local_group_per_node(rank)
        node_rank     = rank // parallelism  

        # weights
        mod = rank % parallelism
        variable_weights = torch.nn.functional.pad(variable_weights, (0,3))
        variable_weights = variable_weights[0,0,0,:36] if rank % parallelism in [0, 2] else variable_weights[0,0,0,36:]
        underground_mask = torch.nn.functional.pad(underground_mask, (0,3)) # [1, xlat, xlon, 68]
        if mod == 0:             
            underground_mask = underground_mask[:,:,:720,:32].to(device)
        elif mod == 1:
            underground_mask = underground_mask[:,:,:720,32:].to(device)
        elif mod == 2:
            underground_mask = underground_mask[:,:,720:,:32].to(device)
        else:
            underground_mask = underground_mask[:,:,720:,32:].to(device)
        

    # Define model
    if parallelism == 1:
        mlp = Model.ModelSequential(config_dict, device, rank).to(device)

        mlp = DDP(
            mlp,
            device_ids=[localid],
            output_device=localid
        )
    elif parallelism == 2:
        mlp = Model.Model_twoway(config_dict, device, rank, local_group).to(device)

        mlp = DDP(
            mlp,
            device_ids=[localid],
            output_device=localid,
            process_group=global_groups[rank % parallelism]
        )

    elif parallelism == 4:
        mlp = Model.Model(config_dict, device, rank, node_groups).to(device)
        mlp = DDP(
            mlp,
            device_ids=[localid],
            output_device=localid,
            process_group=global_groups[rank % parallelism]
        )

    
        
    lr = config_dict['training']['lr']
    save_path = config_dict['training']['save_dir']
    print_on = config_dict['training']['print']
    
    optimizer = optim.Adam([
            {'params': mlp.module.params.embedding.parameters(), 'lr': config_dict['training']['lr_embedding']},
            {'params': mlp.module.params.recovery.parameters(), 'lr': config_dict['training']['lr_recovery']},
            {'params': mlp.module.params.common.parameters()}
            ], 
            lr=lr, 
            weight_decay=3e-6
    )

    if parallelism > 1: # TODO set proper seeds for model parallelism
        utils.set_all_seeds_local_rank(1, rank=rank, parallelism=parallelism) # Set all GPUs to the same seed so that they read the same data           
    else:
        utils.set_all_seeds(1, rank=rank)
    
    train_dataloader = utils.get_dataloader(config_dict, config_dict['data']['train_data_path'], mode='train', distributed=(not model_parallel), batch_size=config_dict['training']['train_batch'], subset_size=train_subset, rank=rank)
    valid_dataloader = utils.get_dataloader(config_dict, config_dict['data']['valid_data_path'], mode='validation', distributed=(not model_parallel), batch_size=config_dict['training']['valid_batch'], subset_size=valid_subset, rank=rank)
    
    if rank == 0:
        print(f"model has {utils.count_parameters(mlp)*parallelism/1e6} million parameters")

    # load
    try:
        file_path = config_dict['training']['load_path']
        mlp, optimizer, epoch = utils.load_model(mlp, optimizer=optimizer, file_path=file_path, rank=rank, model_parallel=model_parallel)
        epoch = epoch + 1
        
    except Exception as err:
        print(err)
        epoch = 0
        if rank == 0:
            print("Initializing model from scratch")
        pass
    
    try:
        if rank == 0 and config_dict['training']['preload']:
            print("loading pretrained convolutional weights")

        if config_dict['training']['preload'] and epoch == 0:
            if config_dict['training']['preload_embedding']:
                mlp = utils.load_pretrained_embedding(mlp, config_dict)
                
            else: # Loading from pangu model
                mlp = utils.load_pretrained_convolution(mlp, config_dict)
            
    except Exception as err:
        if rank == 0:
            print(f"Unable to load pretrained convolutional weights {err}")

    mlp = mlp.to(device)

    loss_fn = torch.nn.MSELoss()
    
    if config_dict['training']['lr_constant']:
        scheduler = StepLRScheduler(optimizer, decay_rate=1., decay_t=100)
    else:
        scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-5, cycle_decay=0.5, cycle_mul=1, cycle_limit=10)
    scheduler.step(epoch)
    
    start_lr = 1e-6
    torch.backends.cuda.matmul.allow_tf32 = True

    handles = []
    
    if parallelism == 4:
        buffer  = torch.zeros(config_dict['training']['train_batch'], xlat, xlon//2, 36, device=device)
        buffer2 = torch.zeros(config_dict['training']['train_batch'], xlat, xlon//2, 36, device=device)
    else:
        buffer, buffer2 = None, None
    
    for epoch in range(epoch, 200):
        #if parallelism < 4 :
        train_dataloader.sampler.set_epoch(epoch)
        
        epoch_average_loss = 0

        if rank == 0:
            print(f"Epoch {epoch}, learning rate {scheduler._get_lr(epoch)}")
            
        mlp.train()
        
        start_time = time.perf_counter()

        for i, data in enumerate(train_dataloader):
            # learning rate ramp-up in first iteration
            if epoch == 0:
                for g in optimizer.param_groups:
                    g['lr'] = i * (lr-start_lr) / len(train_dataloader) + start_lr
                    
            if parallelism == 1:
                optimizer.zero_grad() # zero gradient
                input_data  = data[0].to(device)
            else:
                # Loading data: Always with async ops and handles
                input_data_or_handle,  async_input = utils.load_and_push_data(config_dict, data[0], parallelism, buffer, rank, device, local_groups)
                target_data_or_handle, async_target = utils.load_and_push_data(config_dict, data[1][0], parallelism, buffer2, rank, device, local_groups)
            # DDP and layer norm handles
                if i > 0 and parallelism == 4:
                    handles.wait() # layer norm 
                    vector_to_parameters(flat_grad, gradients)
                # Optimizer step and zero grad 
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                    
                if async_input:
                    input_data_or_handle.wait()
                    input_data = buffer
                else:
                    input_data = input_data_or_handle
               
            output = mlp(input_data, rollout_=0+1) # output shape [n_batch, 720, 720, 36]
                       
                # output: n_batch, 720, 1440, 72//4
                # target data shape: (n_batch, 69, 720, 1440)        
                # target data: [n_batch, 720, 1440, 69]
                    
            if parallelism == 1:
                target_data = data[1][0].to(device)
            
                target_data = target_data.permute(0, 2, 3, 1)
                
                output_loss = output * area_weights * variable_weights
                target_loss = target_data * area_weights *variable_weights
                output_loss[:,:,:,4:] = output_loss[:,:,:,4:] * underground_mask
                target_loss[:,:,:,4:] = target_loss[:,:,:,4:] * underground_mask
                loss = loss_fn(output_loss, target_loss)
            else:
                if async_target:
                    target_data_or_handle.wait()
                    target_data = buffer2
                else:
                    target_data = target_data_or_handle
                
                target_loss = target_data * area_weights * variable_weights
                        
                if rank % parallelism in [0,2]:
                    target_loss[:,:,:,4:] = target_loss[:,:,:,4:] * underground_mask
                else:
                    target_loss = target_loss * underground_mask
                # Reshape output
                output = (
                            output.reshape(output.shape[0], xlat, xlon//2, 36) 
                            if parallelism == 4 
                            else output.reshape(output.shape[0], xlat, xlon, 36)
                        )                
                    
                output_loss = output * area_weights * variable_weights
                    
                if rank % parallelism in [0,2]:
                    output_loss[:,:,:,4:] = output_loss[:,:,:,4:] * underground_mask
                else:
                    output_loss = output_loss * underground_mask
                
                loss = loss_fn(output_loss, target_loss)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), config_dict['training']['grad_clip'])
                if parallelism == 4:
                    handles, flat_grad, gradients = utils.reduce_ddp_layernorms(parallelism, mlp, local_groups, global_groups, rank, node_rank)
                    
                if parallelism == 1:
                    loss = loss / world_size 
                
                    optimizer.step()
                        
                else:
                    loss = loss / world_size
                
                    #torch.distributed.all_reduce(loss)    
#                if rank == 0 and print_on:
#
#                    print(f"{rank}:  {epoch}, after Step {i}/{len(train_dataloader)}. rollout {forecast_leadtime} loss: {loss.detach()}")               
#                    epoch_average_loss = epoch_average_loss + loss.item()
                
        end_time = time.perf_counter()
            
        mlp.eval()

        if rank == 0:
            epoch_average_loss = epoch_average_loss / len(train_dataloader)
            print(f"Epoch {epoch} time [min]: {(end_time - start_time)/60.}, average loss {epoch_average_loss}")

        torch.cuda.synchronize()
        val_loss = 1e7
        val_loss = eval.get_validation_mse(config_dict, mlp, valid_dataloader, device, rank, epoch, file_path=save_path, model_parallel=model_parallel, world_size=world_size, local_groups=local_groups, node_groups=node_groups, global_groups=global_groups)
    #
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if rank < parallelism:
                utils.save_model(mlp, optimizer, save_path, rank, epoch)
                

        scheduler.step(epoch)    
        #torch.distributed.barrier()
    torch.distributed.destroy_process_group()
                

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
