
import argparse
import time

import torch
import torch.distributed
import torch.optim as optim
import yaml
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP

import Dataloaders
import eval
import Model
import utils


def training_loop(config_dict, device, localid, rank):

    model_parallel = config_dict['model']['model_parallel']
    n_parallel = config_dict['model_parallel']['parallelism'] if model_parallel else 1
    
    utils.set_all_seeds(1, rank=(rank%n_parallel)) if n_parallel > 1 else utils.set_all_seeds(1, rank=None) # ensure ranks 0, 4, ...8... have the same initial parameters
    
    train_subset, valid_subset = config_dict['data']['train_subset'], config_dict['data']['valid_subset']
    train_subset = None if train_subset < 0 else train_subset
    valid_subset    = None if valid_subset < 0 else valid_subset
    if rank == 0:
        print(f'Training subset size {train_subset}')
        print(f'Reading from {config_dict["data"]["train_data_path"]}')
    best_val_loss = 1e5
    
    xlat, xlon = config_dict['data']['xlat'], config_dict['data']['xlon']
    surface_weights  = torch.tensor([1.5, 0.77, 0.77, 3.0]) * 4
    # [z, q, t, u, v]
    pressure_weights = torch.tensor([3.00, 0.6, 1.5, 0.9, 0.9]).unsqueeze(1).expand(5, 13).flatten()
    surface_upper_pressure_weights = torch.concat(
        [torch.tensor([1, 1, 1, 1]), # surface variables
        torch.tensor([1, 1, 1, 1, 1, 1, 0.9, 0.8, .7, .6, .5, .4, .3]).repeat(5)]
    ).view(1, 69)

    variable_weights = ((torch.concat([surface_weights, pressure_weights]).view(1, 1, 1, 69))*surface_upper_pressure_weights).to(device)
    
    area_weights = torch.tensor(eval.calc_weight(xlat)).view(1, xlat, 1, 1).to(device).to(torch.float32)    
    col_group  = utils.get_model_group_columns(rank, n_parallel, 2)                                                     
    ddp_group   = utils.get_ddp_groups(rank, n_parallel)
    model_group = utils.get_model_group(rank, n_parallel)

    # weights
    variable_weights = torch.nn.functional.pad(variable_weights, (0,3))
    variable_weights = variable_weights[0,0,0,:36] if rank % n_parallel in [0, 2] else variable_weights[0,0,0,36:]

    # Define model
    
    mlp = Model.Model(config_dict, rank, device, n_parallel, model_group, col_group).to(device)
    mlp = DDP(
        mlp,
        device_ids=[localid],
        output_device=localid,
        process_group=ddp_group
    )

    lr = config_dict['training']['lr']
    save_path = config_dict['training']['save_dir']
    
    optimizer = optim.Adam([
            {'params': mlp.module.params.embedding.parameters(), 'lr': config_dict['training']['lr_embedding']},
            {'params': mlp.module.params.recovery.parameters(), 'lr': config_dict['training']['lr_recovery']},
            {'params': mlp.module.params.common.parameters()}
            ], 
            lr=lr, 
            weight_decay=3e-6
    )

    if n_parallel > 1: 
        utils.set_all_seeds_local_rank(1, rank=rank, n_parallel=n_parallel) # Set all GPUs to the same seed so that they read the same data           
    else:
        utils.set_all_seeds(1, rank=rank)
    
    train_dataloader = Dataloaders.get_dataloader(config_dict, config_dict['data']['train_data_path'], mode='train', distributed=(not model_parallel), batch_size=config_dict['training']['train_batch'], subset_size=train_subset, rank=rank)
    valid_dataloader = Dataloaders.get_dataloader(config_dict, config_dict['data']['valid_data_path'], mode='validation', distributed=(not model_parallel), batch_size=config_dict['training']['valid_batch'], subset_size=valid_subset, rank=rank)
        
    if rank == 0:
        print(f"model has {utils.count_parameters(mlp)*n_parallel/1e6} million parameters")

    # load
    try:
        file_path = config_dict['training']['load_path']
        mlp, optimizer, epoch = utils.load_model(mlp, optimizer=optimizer, file_path=file_path, rank=rank, model_parallel=model_parallel, parallelism=n_parallel)
        epoch = epoch + 1
        
    except Exception as err:
        if rank == 0:
            print(err)
        epoch = 0
        if rank == 0:
            print("Initializing model from scratch")
        pass
    
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

    epoch_average_loss = torch.tensor([0]).to(device)


    for epoch in range(epoch, 200):
        train_dataloader.sampler.set_epoch(epoch)
        epoch_average_loss[0] = 0

        if rank == 0:
            print(f"Epoch {epoch}, learning rate {scheduler._get_lr(epoch)}")
            
        mlp.train()
        start_time = time.perf_counter()
        
        for i, data in enumerate(train_dataloader):
            # learning rate ramp-up in first iteration
            if epoch == 0:
                for g in optimizer.param_groups:
                    g['lr'] = i * (lr-start_lr) / len(train_dataloader) + start_lr
            
            input_data = data[0].to(device, non_blocking=True)
            target_data = data[1][0].to(device, non_blocking=True)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            output = mlp(input_data, rollout_=1) # output shape [n_batch, 720, 720, 36]
            target_loss = target_data * area_weights * variable_weights
                        
            output = output.reshape(output.shape[0], xlat, xlon//2, 36)          
            output_loss = output * area_weights * variable_weights
                
            loss = loss_fn(output_loss, target_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), config_dict['training']['grad_clip'])
            temp_loss = loss.detach()

            epoch_average_loss[0] = epoch_average_loss[0] + temp_loss

            if rank == 0:
                print(f"{epoch}, after Step {i}/{len(train_dataloader)}. loss: {temp_loss}")

        end_time = time.perf_counter()
        torch.cuda.synchronize()

        mlp.eval()

        # average loss
        epoch_average_loss = epoch_average_loss / world_size
        torch.distributed.all_reduce(epoch_average_loss)
        if rank == 0:
            epoch_average_loss = epoch_average_loss / len(train_dataloader)
            print(f"Epoch {epoch} time [min]: {((end_time - start_time) / 60 )}, average loss {epoch_average_loss[0]}")

        torch.cuda.synchronize()
        
        val_loss = eval.get_validation_mse(config_dict, mlp, valid_dataloader, device, rank, epoch, file_path=save_path, model_parallel=model_parallel, world_size=world_size, col_group=col_group, model_group=model_group, ddp_group=ddp_group)

        if rank < n_parallel and val_loss < best_val_loss:
            best_val_loss = val_loss
            utils.save_model(mlp, optimizer, save_path,  rank, epoch)
        if rank == 0:
            print(f"Epoch {epoch} val loss [min]: {val_loss}")

        torch.cuda.synchronize()
        
        scheduler.step(epoch)    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    device, localid, rank, world_size = utils.init_CUDA()
    
    training_loop(config_dict, device, localid, rank)
    torch.distributed.destroy_process_group()