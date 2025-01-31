
import numpy as np
import torch
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
import sys
sys.path.insert(1, '/hkfs/work/workspace/scratch/ke4365-summa/')
sys.path.insert(1, '/hkfs/work/workspace/scratch/ke4365-summa/src')
import utils, eval
import TestModel
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import yaml
import matplotlib.pyplot as plt

def get_local_groups(rank):
    # Rank 4, 6 should be in group together
    gpus_per_node = 4 # horeka- specific
    node_rank     = rank // gpus_per_node # if rank between 0 and 3, node_rank = 0 
                                    # if rank between 4 and 7, node_rank = 1
    starting_group_index = node_rank
    group_0_2 = torch.distributed.new_group(ranks=[starting_group_index,   starting_group_index+2])
    group_1_3 = torch.distributed.new_group(ranks=[starting_group_index+1, starting_group_index+3])
    print(f'local groups {[starting_group_index, starting_group_index+2]}')
    return [group_0_2, group_1_3]

def get_global_groups():
    # Ranks 0, 4, ... should be in a group
    gpus_per_node = 4
    n_nodes = torch.distributed.get_world_size() // gpus_per_node
    # 
    global_group_list = []
    global_groups     = []

    for node in np.arange(gpus_per_node):
        global_group_list.append(((np.arange(n_nodes)*gpus_per_node+node)).tolist())
               
    for global_group in global_group_list:
        global_groups.append(torch.distributed.new_group(ranks=global_group))
    
    return global_groups


def training_loop(config_dict, device, localid, rank):
    comm = None # TODO

    model_parallel = config_dict['model']['model_parallel']
    
    if not model_parallel:
        utils.set_all_seeds(1, rank=None)
    else:
        utils.set_all_seeds(1, rank=(rank%4)) # ensure ranks 0, 4, ...8... have the same initial parameters
    train_subset, valid_subset = config_dict['data']['train_subset'], config_dict['data']['valid_subset']
    if train_subset < 0:
        train_subset = None
    if valid_subset < 0:
        valid_subset = None

    best_val_loss = 1e5
    
    if not model_parallel:
        mlp = TestModel.ModelSequential(config_dict, device, rank).to(torch.float32).to(device)

        mlp_ddp = DDP(
            mlp,
            device_ids=[localid],
            output_device=localid
        )
    else:
        mlp = TestModel.Model(config_dict, device, comm, rank).to(torch.float32).to(device)

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
    elif not model_parallel:
        optimizer = optim.Adam(mlp_ddp.parameters(), lr=lr, weight_decay=3e-6)
    else:
        mlp_ddp = mlp
        optimizer = optim.Adam(mlp_ddp.parameters(), lr=lr, weight_decay=3e-6)

    if config_dict['model']['model_parallel']:
        utils.set_all_seeds_local_rank(1, rank=rank) # Set all GPUs to the same seed so that they read the same data           
    
    train_dataloader = utils.get_dataloader(config_dict, config_dict['data']['train_data_path'], mode='train', distributed=(not model_parallel), batch_size=config_dict['training']['train_batch'], subset_size=train_subset, rank=rank)
    valid_dataloader = utils.get_dataloader(config_dict, config_dict['data']['valid_data_path'], mode='validation', distributed=(not model_parallel), batch_size=config_dict['training']['valid_batch'], subset_size=valid_subset, rank=rank)
    
    if rank == 0:
        print(f"model has {utils.count_parameters(mlp_ddp)/1e6} million parameters")

    # load
    try:
        file_path = config_dict['training']['load_path']
        mlp_ddp, optimizer, epoch = utils.load_model(mlp_ddp, optimizer=optimizer, file_path=file_path, rank=rank, model_parallel=model_parallel)
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
                mlp_ddp = utils.load_pretrained_embedding(mlp_ddp, config_dict)
                
            else: # Loading from pangu model
                mlp_ddp = utils.load_pretrained_convolution(mlp_ddp, config_dict)
            
    except Exception as err:
        if rank == 0:
            print(f"Unable to load pretrained convolutional weights {err}")

    mlp_ddp = mlp_ddp.to(device)
    # Completely DDP model, no model parallelism

    try: 
        if config_dict['training']['loss'] == 'L1':
            loss_fn = torch.nn.L1Loss()
        else:
            loss_fn = torch.nn.MSELoss()
    except Exception as err:
        loss_fn = torch.nn.MSELoss()
    
    xlat, xlon = config_dict['data']['xlat'], config_dict['data']['xlon']
    surface_weights  = torch.tensor([1.5, 0.77, 0.66, 3.0])
    pressure_weights = torch.tensor([3.00, 0.6, 1.5, 0.77, 0.54]).unsqueeze(1).expand(5, 13).flatten()
    variable_weights = torch.concat([surface_weights, pressure_weights]).view(1, 69).to(device)
    area_weights = torch.tensor(eval.calc_weight(xlat)).to(device)

    if config_dict['training']['lr_constant']:
        scheduler = StepLRScheduler(optimizer, decay_rate=1., decay_t=100)
    else:
        scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-5, cycle_decay=0.5, cycle_mul=1, cycle_limit=10)
    scheduler.step(epoch)
    
    start_lr = 1e-6
    if config_dict['model']['model_parallel']:
        groups = get_local_groups(rank)
        global_groups = get_global_groups()
        # weights
        variable_weights = torch.nn.functional.pad(variable_weights, (0,3))
        variable_weights = variable_weights[0,:36] if rank in [0, 2] else variable_weights[0,36:]
    else:
        groups = None
    
    for epoch in range(epoch, 200):
        if not config_dict['model']['model_parallel']:
            train_dataloader.sampler.set_epoch(epoch)
        
        epoch_average_loss = 0

        if rank == 0:
            print(f"Epoch {epoch}, learning rate {scheduler._get_lr(epoch)}")
        start_time = time.perf_counter()

        mlp_ddp.train()
        
        for i, data in enumerate(train_dataloader):
            # learning rate ramp-up in first iteration
            if epoch == 0:
                for g in optimizer.param_groups:
                    g['lr'] = i * (lr-start_lr) / len(train_dataloader) + start_lr

            # randomize forecast output
            forecast_leadtime = torch.tensor(np.random.randint(0, config_dict['model']['latent_rollout'])).to(device)
            if config_dict['model']['latent_rollout'] > 1:
                torch.distributed.broadcast(forecast_leadtime, 0) 
                forecast_leadtime = forecast_leadtime.item() 
            
            optimizer.zero_grad()
            
            if config_dict['model']['model_parallel']:
                # Defining senders and receivers # [0, 1] send, or [2, 3] send
                buffer = torch.zeros(config_dict['training']['train_batch'], 36, xlat, xlon//2).to(device)
                if config_dict['model_parallel']['split'] == 2:
                    #data[0][0] = data[0][0].squeeze()
                    #data[0][1] = data[0][1].squeeze()
                    input_data = utils.send_data_twoway(data[0], i, buffer, rank, device, groups,config_dict['model_parallel']['alt_sender'])
                elif config_dict['model_parallel']['split'] == 4:
                    input_data = data[0].permute(0, 2, 3, 1).to(device)
                elif config_dict['model_parallel']['split'] == 1:
                    input_data = utils.send_data_fourway(data[0], buffer, rank, device)
            else:
                input_data  = data[0].to(torch.float32).to(device)
                target_data = data[1][0].to(torch.float32).to(device)
                            
            output = mlp_ddp(input_data, rollout_=forecast_leadtime+1) # output shape [n_batch, 720, 720, 36]

            # output: n_batch, 720, 1440, 72//4
            # target data shape: (n_batch, 69, 720, 1440)        
            # target data: [n_batch, 720, 1440, 69]
            
            if model_parallel:
                # load data on sender ranks
                buffer = torch.zeros(config_dict['training']['train_batch'], 36, xlat, xlon//2).to(device)
                if config_dict['model_parallel']['split'] == 2:
                    #if len(data[1][0]) == 2:
                    #    data[1][0][0] = data[1][0][0].squeeze()
                    #    data[1][0][1] = data[1][0][1].squeeze()
                    target_data = utils.send_data_twoway(data[1][forecast_leadtime], i, buffer, rank, device, groups, config_dict['model_parallel']['alt_sender'])
                elif config_dict['model_parallel']['split'] == 4:
                    target_data = data[1][forecast_leadtime].permute(0, 2, 3, 1).to(device)
                elif config_dict['model_parallel']['split'] == 1:
                    target_data = utils.send_data_fourway(data[0], buffer, rank, device)

                output = output.reshape(output.shape[0], xlat, xlon//2, 36)                
                if config_dict['training']['area_weighted']:
                    loss = loss_fn(torch.einsum('bxyz, x->bxyz', output, area_weights)*variable_weights, torch.einsum('bxyz, x->bxyz', target_data, area_weights)*variable_weights) 
                else:
                    loss = loss_fn(output*variable_weights, target_data*variable_weights)
            else:
                target_data = target_data.permute(0, 2, 3, 1)
                if config_dict['training']['area_weighted']:
                    loss = loss_fn(torch.einsum('bxyz, x->bxyz', output, area_weights)*variable_weights, torch.einsum('bxyz, x->bxyz', target_data, area_weights)*variable_weights)         
                else:
                    loss = loss_fn(output*variable_weights, target_data*variable_weights)         
                                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp_ddp.parameters(), config_dict['training']['grad_clip'])

            if config_dict['model']['model_parallel']:
                # Now manually all-reduce each parameter's grad
                for name, param in mlp_ddp.named_parameters():
                    if param.grad is not None:
                        if 'layer_norm' in name:
                            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG, group=groups[rank % 2])
                        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG, group=global_groups[rank % 4]) # reduce over ranks
                                            
            torch.distributed.all_reduce(loss)
            optimizer.step()
            
            loss = loss / world_size 
            
            if rank == 0:
                print(f"{rank}:  {epoch}, after Step {i}/{len(train_dataloader)}. rollout {forecast_leadtime} loss: {loss.detach()}")
                epoch_average_loss += loss.item()

        end_time = time.perf_counter()
            
        mlp_ddp.eval()
        if rank == 0:
            epoch_average_loss = epoch_average_loss / len(train_dataloader)
            print(f"Epoch {epoch} time [min]: {(end_time - start_time)/60.}, average loss {epoch_average_loss}")

        val_loss = eval.get_validation_mse(config_dict, mlp_ddp, valid_dataloader, device, rank, epoch, file_path=save_path, model_parallel=model_parallel, world_size=world_size, groups=groups)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not model_parallel:
                if rank == 0:
                    utils.save_model(mlp_ddp, optimizer, save_path, rank, epoch)
                else:
                    pass
            else:
                utils.save_model(mlp_ddp, optimizer, save_path,  rank, epoch)
        else:
            if model_parallel:
                utils.save_model(mlp_ddp, optimizer, save_path, rank, epoch)

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
