import utils
import numpy as np
import torch
import torch.optim as optim
import sys
sys.path.insert(1, 'src/')
from Linear import DistributedLinear, DistributedMLP
import summa_dist
from mpi4py import MPI
from torch.utils.data import DataLoader, TensorDataset


comm, i, j = utils.init_mpi()
device, slurm_localid, rank, world_size = utils.init_CUDA()

m, n, p = 4, 2, 2
world_size = comm.Get_size()
rank = comm.Get_rank()
Xij, _, Cij, A, B = summa_dist.init_summa_algo(m=m, n=n, p=p, i=i, j=j, world_size=world_size)

subset_size_m = int(m//np.sqrt(world_size))
subset_size_n = int(n//np.sqrt(world_size))
subset_size_p = int(p//np.sqrt(world_size))

#goal = torch.tensor([0, 3, 0, -1, 3, 6, 3, 0, 1, 2, 1, 0, -1, 1, -1, -1]).reshape(m, p).to(torch.float32)
goal = torch.tensor([0, 1, 3, 6, 1, 2, -1, -1]).reshape(m, p).to(torch.float32)
goal_i = goal[i*subset_size_m:(i+1)*subset_size_m, j*subset_size_p:(j+1)*subset_size_p]

loss_fn = torch.nn.MSELoss()

config_dict = {'data': {'dt': 6, 
                        'train_data_path': '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr',
                        'valid_data_path': '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr',
                        'prefetch_factor': 6,
                        'num_workers': 1
                        }
               }

if rank == 0:
    dataloader = utils.get_dataloader(config_dict)
else:
    dataloader = None

#data = torch.arange(100, dtype=torch.float32).reshape(-1,1)
#dataset = TensorDataset(data)
#dataloader = DataLoader(dataset, batch_size=8)  # Adjust batch size as needed

#mlp = DistributedMLP(1, 6, 1, comm)
dim = 720
mlp = DistributedLinear(dim, dim, comm).to(torch.float32)

lr = 1e-4


optimizer = optim.SGD(mlp.parameters(), lr=lr)

# Iterate through batches on rank 0 and scatter each batch to other ranks
for batch_idx in range(1000):#: if rank == 0 else 0):
    if rank == 0:
        # Get the batch data on rank 0
        batch_data = next(iter(dataloader))
        batch_data[0] = batch_data[0].reshape(1, 69, 360, 2, 720, 2).permute(3, 5, 0, 1, 2, 4).reshape(-1, 69, 360, 720) # [1, 69, 360, 720]
        batch_data[1] = batch_data[1][0].reshape(1, 69, 360, 2, 720, 2).permute(3, 5, 0, 1, 2, 4).reshape(-1, 69, 360, 720) # [1, 69, 720, 1440]
        #batch_data = []
        #batch_data.append(torch.normal(mean=0, std=1, size=(4, 5, 10, dim)))
        #batch_data.append(batch_data[0])
        # Split batch data among ranks
        batch_split_in = torch.chunk(batch_data[0], 4, dim=0)
        batch_split_out = torch.chunk(batch_data[1], 4, dim=0)
    else:
        batch_split_in  = None  # No data on other ranks initially
        batch_split_out = None

    # Scatter the current batch split to all ranks
    
    scattered_batch_in = comm.scatter(batch_split_in, root=0)
    scattered_batch_out = comm.scatter(batch_split_out, root=0)

    # shape (1, 69, 360, 720),
    scattered_batch_in  = scattered_batch_in.reshape(69*360, -1)
    scattered_batch_out = scattered_batch_in.reshape(69*360, -1)
    

#    print(f"rank {rank}, data shape {scattered_batch_in.shape}, {scattered_batch_out.shape}")

    optimizer.zero_grad()
    
    output = mlp(scattered_batch_in)
    #.reshape(1, 69, 360, 720)

    
    loss = loss_fn(output, scattered_batch_out)
    loss = comm.allreduce(loss) / world_size
    #
    loss.backward()
    optimizer.step()
    if rank == 0:
        print(loss.detach())
#
            #if i % 1000 == 0:
            #    for param in mlp.parameters():
            #        print(f"rank {rank}, output {output})#, param {param[0]}")

