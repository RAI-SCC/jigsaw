
import numpy as np
import torch
import torch.distributed
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler

import sys
sys.path.insert(1, '/hkfs/work/workspace/scratch/ke4365-Jigsaw/')
sys.path.insert(1, '/hkfs/work/workspace/scratch/ke4365-Jigsaw/src')
import utils, eval
import Model
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import yaml
import matplotlib.pyplot as plt
import os, psutil
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import helper

def training_loop(config_dict, device, localid, rank):

    
    model_parallel = config_dict['model']['model_parallel']
    parallelism = config_dict['model_parallel']['parallelism'] if model_parallel else 1
    
    utils.set_all_seeds(1, rank=(rank%parallelism)) if parallelism > 1 else utils.set_all_seeds(1, rank=None) # ensure ranks 0, 4, ...8... have the same initial parameters
    
    train_subset, valid_subset = config_dict['data']['train_subset'], config_dict['data']['valid_subset']
    train_subset = None if train_subset < 0 else train_subset
    valid_subset    = None if valid_subset < 0 else valid_subset
    if rank == 0:
        print(f'Training parameters {config_dict["training"]}')
        print(f'Model parameters {config_dict["model"]}')
        print(f'Training subset size {train_subset}')
        print(f'Data parameters {config_dict["data"]}')
        
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
    weights = torch.load('/hkfs/work/workspace/scratch/ke4365-kangu/kangu_finetuning/finetuning_weights/feature_weights.pt', map_location='cpu')
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
        
    if rank == 0:
        print(f"model has {utils.count_parameters(mlp)*parallelism/1e6} million parameters")
        flop_forward, comm_forward = helper.calc_flops_comm(config_dict)
        print(f'Estimated forward TFLOPs and GB_COMM: {flop_forward}, {comm_forward}')

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

    if config_dict['training']['tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    epoch_times = np.zeros(11)

    for epoch in range(11):
        #if parallelism < 4 :
        train_dataloader.sampler.set_epoch(epoch)


        mlp.train()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        for i, data in enumerate(train_dataloader):
            # learning rate ramp-up in first iteration
            optimizer.zero_grad() # zero gradient
            
            input_data  = data[0].to(device, non_blocking=False)
            target_data = data[1][0].to(device, non_blocking=False)
                        

        end.record()
        torch.cuda.synchronize()

        if rank == 0:
            epoch_times[epoch] = start.elapsed_time(end) / 1000 / 60 
            print(f"Epoch {epoch} time [min]: {(epoch_times[epoch])}")
        
        scheduler.step(epoch)    
    if rank == 0:
        print(f'Average epoch time is {np.average(epoch_times)}')
        print(f'Std epoch time is {np.std(epoch_times)}')

        
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
