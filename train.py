
import numpy as np
import torch
import torch.distributed
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
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

    comm = None
    model_parallel = config_dict['model']['model_parallel']
    parallelism = config_dict['model_parallel']['parallelism'] if model_parallel else 1
    
    utils.set_all_seeds(1, rank=(rank%parallelism)) if parallelism > 1 else utils.set_all_seeds(1, rank=None) # ensure ranks 0, 4, ...8... have the same initial parameters
    
    train_subset, valid_subset = config_dict['data']['train_subset'], config_dict['data']['valid_subset']
    train_subset = None if train_subset < 0 else train_subset
    valid_subset    = None if valid_subset < 0 else valid_subset
    
    best_val_loss = 1e5
    
    xlat, xlon = config_dict['data']['xlat'], config_dict['data']['xlon']
    surface_weights  = torch.tensor([1.5, 0.77, 0.77, 3.0]) * 4
    pressure_weights = torch.tensor([3.00, 0.6, 1.5, 0.9, 0.9]).unsqueeze(1).expand(5, 13).flatten()
    variable_weights = torch.concat([surface_weights, pressure_weights]).view(1, 69).to(device)
    area_weights = torch.tensor(eval.calc_weight(xlat)).to(device)

    if parallelism == 1:
        local_groups = None
        node_groups = None
    elif parallelism == 2:
        local_group = utils.get_local_groups_twoway(rank)
        local_groups = None
        node_groups = None
        global_groups = utils.get_ddp_groups(parallelism)
        variable_weights = torch.nn.functional.pad(variable_weights, (0,3))
        variable_weights = variable_weights[0,:36] if rank in [0, 2] else variable_weights[0,36:]
        node_rank     = rank // parallelism
    elif  parallelism == 4:
        local_groups  = utils.get_local_groups_fourway(rank) # groups are [[0, 2], [1, 3], [4, 6], [5, 7]]
                                                         # group [0, 2]: index 0. group [1, 3]: index 1. group [4, 6]: index 3. 
                                                         # index = rank % 2 + node_rank    
        global_groups = utils.get_ddp_groups(parallelism)
        node_groups   = utils.get_local_group_per_node(rank)
        node_rank     = rank // parallelism  
#        layer_norm_indices = [i for i, k in enumerate(mlp_ddp.state_dict().keys()) if 'layer_norm' in k]
        # weights
        variable_weights = torch.nn.functional.pad(variable_weights, (0,3))
        variable_weights = variable_weights[0,:36] if rank in [0, 2] else variable_weights[0,36:]

    # Define model
    if parallelism == 1:
        mlp = Model.ModelSequential(config_dict, device, rank).to(torch.float32).to(device)

        mlp = DDP(
            mlp,
            device_ids=[localid],
            output_device=localid
        )
    elif parallelism == 2:
        mlp = Model.Model_twoway(config_dict, device, comm, rank, local_group).to(device)
        mlp = DDP(
            mlp,
            device_ids=[localid],
            output_device=localid,
            process_group=global_groups[rank % parallelism]
        )

    elif parallelism == 4:
        mlp = Model.Model(config_dict, device, comm, rank).to(device)
    
    lr = config_dict['training']['lr']
    save_path = config_dict['training']['save_dir']
    
    if parallelism > 1:
        optimizer = optim.Adam([
                {'params': mlp.module.params.embedding.parameters(), 'lr': config_dict['training']['lr_embedding']},
                {'params': mlp.module.params.recovery.parameters(), 'lr': config_dict['training']['lr_recovery']},
                {'params': mlp.module.params.common.parameters()}
                ], 
                lr=lr, 
                weight_decay=3e-6
    )
    else:
        optimizer = optim.Adam([
            {'params': mlp.params.embedding.parameters(), 'lr': config_dict['training']['lr_embedding']},
            {'params': mlp.params.recovery.parameters(), 'lr': config_dict['training']['lr_recovery']},
            {'params': mlp.params.common.parameters()}
            ], 
            lr=lr, 
            weight_decay=3e-6
        )
    #if config_dict['training']['preload']:
    #    optimizer = optim.Adam([
    #        {'params': mlp_ddp.module.params.embedding.parameters(), 'lr': config_dict['training']['lr_embedding']},
    #        {'params': mlp_ddp.module.params.recovery.parameters(), 'lr': config_dict['training']['lr_recovery']},
    #        {'params': mlp_ddp.module.params.common.parameters()}
    #        ], 
    #        lr=lr, weight_decay=3e-6
    #    )
    #elif parallelism == 1:
    #    optimizer = optim.Adam(mlp_ddp.parameters(), lr=lr, weight_decay=3e-6)
    #else:
    #    mlp_ddp = mlp
    #    optimizer = optim.Adam(mlp_ddp.parameters(), lr=lr, weight_decay=3e-6)

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

    # Completely DDP model, no model parallelism
    try: 
        if config_dict['training']['loss'] == 'L1':
            loss_fn = torch.nn.L1Loss()
        else:
            loss_fn = torch.nn.MSELoss()
    except Exception as err:
        loss_fn = torch.nn.MSELoss()
    
    

    if config_dict['training']['lr_constant']:
        scheduler = StepLRScheduler(optimizer, decay_rate=1., decay_t=100)
    else:
        scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-5, cycle_decay=0.5, cycle_mul=1, cycle_limit=10)
    scheduler.step(epoch)
    
    start_lr = 1e-6
    torch.backends.cuda.matmul.allow_tf32 = True

    
    handles = []
    
    buffer  = torch.zeros(config_dict['training']['train_batch'], xlat, xlon//2, 36, device=device)
    buffer2 = torch.zeros(config_dict['training']['train_batch'], xlat, xlon//2, 36, device=device)
    
    for epoch in range(epoch, 200):
        if parallelism < 4 :
            train_dataloader.sampler.set_epoch(epoch)
        
        epoch_average_loss = 0

        if rank == 0:
            print(f"Epoch {epoch}, learning rate {scheduler._get_lr(epoch)}")
            
        mlp.train()
        start_time = time.perf_counter()

        for i, data in enumerate(train_dataloader):
            # learning rate ramp-up in first iteration
            wait_start = time.perf_counter()    
            if epoch == 0:
                for g in optimizer.param_groups:
                    g['lr'] = i * (lr-start_lr) / len(train_dataloader) + start_lr
            
            # randomize forecast output
            forecast_leadtime = 0
            if config_dict['model']['latent_rollout'] > 1:
                forecast_leadtime = torch.tensor(np.random.randint(0, config_dict['model']['latent_rollout'])).to(device)
                torch.distributed.broadcast(forecast_leadtime, 0) 
                forecast_leadtime = forecast_leadtime.item() 

            section_start = time.perf_counter()
            if parallelism == 1:
                optimizer.zero_grad() # zero gradient
                input_data  = data[0].to(device)
            else:
                # Loading data: Always with async ops and handles
                
                input_data_or_handle,  async_input = utils.load_and_push_data(config_dict, data[0], parallelism, buffer, rank, device, local_groups)

                # DDP and layer norm handles
                #for h in handles:
                #    h.wait()
                #if i > 0 :
                #    handles.wait() # ddp
                #
                #    vector_to_parameters(flat_grad, gradients)
                #   # Optimizer step and zero grad 

                optimizer.step()
                optimizer.zero_grad()
                    
                
                if async_input:
                    input_data_or_handle.wait()
                    input_data = buffer
                else:
                    input_data = input_data_or_handle
                

                
            output = mlp(input_data, rollout_=forecast_leadtime+1) # output shape [n_batch, 720, 720, 36]
                   
            # output: n_batch, 720, 1440, 72//4
            # target data shape: (n_batch, 69, 720, 1440)        
            # target data: [n_batch, 720, 1440, 69]
            
            if parallelism == 1:
                target_data = data[1][forecast_leadtime].to(device)
            
                target_data = target_data.permute(0, 2, 3, 1)
                if config_dict['training']['area_weighted']:
                    loss = loss_fn(torch.einsum('bxyz, x->bxyz', output, area_weights)*variable_weights, torch.einsum('bxyz, x->bxyz', target_data, area_weights)*variable_weights)         
                else:
                    loss = loss_fn(output*variable_weights, target_data*variable_weights)         
            else:
                
                target_data_or_handle, async_target = utils.load_and_push_data(config_dict, data[1][0], parallelism, buffer2, rank, device, local_groups)

                if async_target:
                    target_data_or_handle.wait()
                    target_data = buffer2
                else:
                    target_data = target_data_or_handle
                
                # Reshape output
                output = (
                            output.reshape(output.shape[0], xlat, xlon//2, 36) 
                            if parallelism == 4 
                            else output.reshape(output.shape[0], xlat, xlon, 36)
                         )                
                
                
                if config_dict['training']['area_weighted']:
                    loss = loss_fn(torch.einsum('bxyz, x->bxyz', output, area_weights)*variable_weights, torch.einsum('bxyz, x->bxyz', target_data, area_weights)*variable_weights) 
                else:
                    loss = loss_fn(output*variable_weights, target_data*variable_weights)
                

            #backward_start = time.perf_counter()
            
            loss.backward()
            #
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), config_dict['training']['grad_clip'])

            if parallelism > 1:
                x=1
                #handles, flat_grad, gradients = utils.reduce_ddp_layernorm_flat(parallelism, mlp, local_groups, global_groups, rank, node_rank)
                
                #loss = loss / world_size
                #torch.distributed.all_reduce(loss)    
                
                #for param_group in optimizer.param_groups:
                #    for index, param in enumerate(param_group['params']):
                #        if param is None or index not in layer_norm_indices:
                #            continue
#
                #        state = optimizer.state.get(param, None)
#
                #        if not state:
                #            continue
                #        if 'exp_avg' in state:
                #            exp_avg = state['exp_avg']
                #            h = torch.distributed.all_reduce(exp_avg, op=torch.distributed.torch.distributed.ReduceOp.AVG, group=local_groups[rank % 2 + node_rank * 2], async_op=True)
                #            handles.append(h)
                #        
                #        # 'exp_avg_sq' = second moment of gradient
                #        if 'exp_avg_sq' in state:
                #            exp_avg_sq = state['exp_avg_sq']
                #            if len(exp_avg_sq.shape) == 1:
                #                h = torch.distributed.all_reduce(exp_avg_sq, op=torch.distributed.ReduceOp.AVG, group=local_groups[rank % 2 + node_rank * 2], async_op=True)
                #                handles.append(h)
                #
                # 
              #  backward_end = time.perf_counter()
            if parallelism == 1:
                loss = loss / world_size 
                torch.distributed.all_reduce(loss)    
                optimizer.step()
            if rank == 0:
                wait_end = time.perf_counter() 
                print(f"{rank}:  {epoch}, after Step {i}/{len(train_dataloader)}. rollout {forecast_leadtime} loss: {loss.detach()}, epoch time {wait_end-wait_start} s")               
                epoch_average_loss = epoch_average_loss + loss.item()

        end_time = time.perf_counter()
            
        mlp.eval()

        if rank == 0:
            epoch_average_loss = epoch_average_loss / len(train_dataloader)
            print(f"Epoch {epoch} time [min]: {(end_time - start_time)/60.}, average loss {epoch_average_loss}")

        torch.cuda.synchronize()
        val_loss = 1e7
     #   val_loss = eval.get_validation_mse(config_dict, mlp, valid_dataloader, device, rank, epoch, file_path=save_path, model_parallel=model_parallel, world_size=world_size, groups=local_groups, node_groups=node_groups)
    #
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if parallelism == 1:
                if rank == 0:
                    utils.save_model(mlp, optimizer, save_path, rank, epoch)
                else:
                    pass
            else:
                # For now, save model on every rank, so we can compare the weights.
                utils.save_model(mlp, optimizer, save_path,  rank, epoch)
        # TODO: stop saving @ every epoch
        utils.save_model(mlp, optimizer, save_path,  rank, epoch)

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

    device, localid, rank, world_size = utils.init_CUDA()

    
    training_loop(config_dict, device, localid, rank)
    torch.distributed.destroy_process_group()
