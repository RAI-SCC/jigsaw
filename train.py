import utils
import numpy as np
import torch
import torch.optim as optim
import sys
sys.path.insert(1, 'src/')
from Linear import DistributedXWT, DistributedMLP, MLPDimension
import Linear as linear
import TestModel
import summa_dist
from mpi4py import MPI
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

device, slurm_localid, torch_rank, world_size = utils.init_CUDA()


loss_fn = torch.nn.L1Loss()

config_dict = {'data': {'dt': 6, 
                        'train_data_path': '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr',
                        'valid_data_path': '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr',
                        'prefetch_factor': 4,
                        'num_workers': 2
                        }
               }

#if mpi_rank == 0:
#    dataloader = utils.get_dataloader(config_dict)
#else:
#    dataloader = None

dataloader = utils.get_dataloader(config_dict, distributed=True)

mlp = TestModel.Model(device, comm, torch_rank).to(torch.float64).to(device)
#mlp = DDP(
#    mlp,
#    device_ids=[slurm_localid],
#    output_device=slurm_localid
#)

lr = 1e-4

optimizer = optim.Adam(mlp.parameters(), lr=lr)


if slurm_localid == 0:
    print(f"model has {utils.count_parameters(mlp)/1e6} million parameters")

# Completely DDP model, no model parallelism
for epoch in range(100):
    for i, data in enumerate(dataloader):
    ## Get the batch data on rank 0
        input_data  = data[0].to(torch.float64).to(device)
        target_data = data[1][0].to(torch.float64).to(device)
        
        optimizer.zero_grad()
        output = mlp(input_data)

        # output: n_batch, 720, 1440, 72//4
        
        # target data shape: (n_batch, 69, 720, 1440)
        fraction = 72 // 4
        target_data = target_data.permute(0, 2, 3, 1)
        # target data: [n_batch, 720, 1440, 69]

        if torch_rank != 3:
            loss = loss_fn(output, target_data[:, :, :, torch_rank*fraction:(torch_rank+1)*fraction])
        else:
            loss = loss_fn(output[:,:,:,:15], target_data[:,:,:,:15])
        
        torch.distributed.all_reduce(loss/4)
        
        
        if mpi_rank == 0:
            print(f"rank {mpi_rank}, loss: {loss.detach()}")

        loss.backward()
        optimizer.step()


torch.distributed.destroy_process_group()