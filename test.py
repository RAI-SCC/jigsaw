from summa_dist import init_mpi, init_summa_algo, ABT, verifyT, gather, mul_summa, ATB
import torch
import numpy as np
from tqdm import tqdm
n = 4           # problem size
comm, i, j = init_mpi()
world_size = comm.Get_size()
rank = comm.Get_rank()

m, n, p = 4, 2, 2
subset_size_m = int(m//np.sqrt(world_size))
subset_size_n = int(n//np.sqrt(world_size))
subset_size_p = int(p//np.sqrt(world_size))

Xij, Wij, Cij, A, B = init_summa_algo(m=m, n=n, p=p, i=i, j=j, world_size=world_size)
Cij = mul_summa(Xij, Wij, comm.Get_rank(), world_size, comm)
#print(f"BEFORE Xij shape {Xij.shape}, Wij shape {Wij.shape}")
goal = torch.tensor([0, 1, 3, 6, 1, 2, -1, -1]).reshape(m, p).to(torch.float32)

goal_i = goal[i*subset_size_m:(i+1)*subset_size_m, j*subset_size_n:(j+1)*subset_size_n]
lr = 1e-3

for i in tqdm(range(20000)):
    dD   = Cij - goal_i
    dW = ATB(Xij, dD,  n, comm.Get_rank(), world_size, comm)

    W_gather = gather(Wij, comm, world_size)
    X_gather = gather(Xij, comm, world_size)

    Wij = Wij - dW * lr

    Cij = mul_summa(Xij, Wij, comm.Get_rank(), world_size, comm)

    # NON DISTRIBUTED d
    if rank == 0:
        D = X_gather @ W_gather
        if i % 100 == 0:
            print(W_gather)
        print(torch.sum(torch.abs(D - goal)))
    