import utils, eval
import numpy as np
import torch
import torch.optim as optim
import sys
sys.path.insert(1, 'src/')
import TestModel

from mpi4py import MPI
from torch.nn.parallel import DistributedDataParallel as DDP

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

device, slurm_localid, torch_rank, world_size = utils.init_CUDA()
config_dict = {'data': {'dt': 6, 
                        'train_data_path': '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr',
                        'valid_data_path': '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr',
                        'prefetch_factor': 4,
                        'num_workers': 2
                        }
               }

model_parallel = True
train_subset = None

train_dataloader = utils.get_dataloader(config_dict, mode='train', distributed=(not model_parallel), batch_size=4, subset_size=train_subset)
valid_subset = 8
valid_dataloader = utils.get_dataloader(config_dict, mode='valid', distributed=(not model_parallel), subset_size=valid_subset)


if not model_parallel:
    mlp = TestModel.ModelSequential(device, comm, torch_rank).to(torch.float32).to(device)

    mlp = DDP(
        mlp,
        device_ids=[slurm_localid],
        output_device=slurm_localid
    )
else:
    mlp = TestModel.Model(device, comm, torch_rank).to(torch.float32).to(device)

lr = 5e-4
save_path = 'experiments/patch4x4_1000_ll'

optimizer = optim.Adam(mlp.parameters(), lr=lr)


if slurm_localid == 0:
    print(f"model has {utils.count_parameters(mlp)/1e6} million parameters")

# load
try:
    mlp = utils.load_model(mlp, file_path=save_path, rank=torch_rank)
except:
    if slurm_localid == 0:
        print("Initializing model from scratch")
    pass
# Completely DDP model, no model parallelism

loss_fn = torch.nn.MSELoss()

for epoch in range(100):
    for i, data in enumerate(train_dataloader):
    ## Get the batch data on rank 0
        input_data  = data[0].to(torch.float32)#.to(device)
        target_data = data[1].to(torch.float32).to(device)
        
        optimizer.zero_grad()
        output = mlp(input_data)
        

        # output: n_batch, 720, 1440, 72//4
        
        # target data shape: (n_batch, 69, 720, 1440)
        
        target_data = target_data.permute(0, 2, 3, 1)
        # target data: [n_batch, 720, 1440, 69]
        
        if model_parallel:
            fraction_1 = 720*1440//2
            fraction_2 = 72 // 2
            target_data = target_data.reshape(target_data.shape[0], -1, target_data.shape[3])
            target_data = torch.nn.functional.pad(target_data, (0,3)) 
            if torch_rank == 0:
                loss = loss_fn(output, target_data[:, :fraction_1,  :fraction_2])
            elif torch_rank == 1:
                loss = loss_fn(output, target_data[:, :fraction_1,  fraction_2:])
            elif torch_rank == 2:
                loss = loss_fn(output, target_data[:, fraction_1:,  fraction_2:])
            elif torch_rank == 3:
                loss = loss_fn(output, target_data[:, :fraction_1,  :fraction_2])

        else:
            loss = loss_fn(output, target_data)

        torch.distributed.all_reduce(loss)
        torch.distributed.barrier()
        loss = loss / 4
        
        if torch_rank == 0:
            print(f"{torch_rank}:  {epoch}, Step {i}/{len(train_dataloader)}. loss: {loss.detach()}")
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        optimizer.step()

        # print a picture and save a state
        frequency = 100
        
        if i % frequency == 0:# and (i != 0):
            #eval.get_validation_mse(mlp, valid_dataloader, device, torch_rank, epoch, file_path=save_path, model_parallel=model_parallel)
#            gather_list = [torch.empty(output.shape).cuda(torch_rank) for _ in range(world_size)] 
#
#            # Perform gather operation
#            torch.distributed.all_gather(gather_list, output)
#            torch.distributed.barrier()
            
            # Concatenate tensors on rank 0
            #if torch_rank == 0:
            #    result = torch.cat(gather_list, dim=-1)  # Combine tensors along the last dimension
            #    rmse_u10 = eval.calc_mse_single_variable(target_data, result, device, index=1)    
            #    rmse_t2m = eval.calc_mse_single_variable(target_data, result, device, index=3)    
            #    print(f"rmse u10: {rmse_u10}, rmse_t2m: {rmse_t2m}")
#
            #    output_print, target_print = output.cpu(), target_data[:, :, :, torch_rank*fraction:(torch_rank+1)*fraction].cpu()
            #    output_print = output_print.reshape(720, 1440, 18).permute(2, 0, 1)
            #    target_print = target_print.reshape(720, 1440, 18).permute(2, 0, 1)
            #    save_path = f"out/u10_{epoch}.png"
            #    utils.plot_fields(input_data.cpu(), output_print.cpu(), target_print, save_path=save_path, index=1)
            #    save_path = f"out/T2M_{epoch}.png"
            #    utils.plot_fields(input_data.cpu(), output_print.cpu(), target_print, save_path=save_path, index=3)
            if model_parallel and slurm_localid == 0:
                utils.save_model(mlp, save_path, torch_rank)
            else:
                utils.save_model(mlp, save_path, torch_rank)

torch.distributed.destroy_process_group()