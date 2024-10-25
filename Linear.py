import torch
import torch.optim as optim
import summa_dist
from mpi4py import MPI
import numpy as np
from torch.nn import GELU, Dropout
from lineartest import MyLinearFunction


class DistributedLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, comm):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights_ij = torch.nn.parameter.Parameter(torch.normal(0, 1, size=(out_dim, in_dim)))
        
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()
        self.summa = SUMMA().apply
    
    # def get_weights(self):
    #     return self.weights_ij
    
    def forward(self, x):
        #Cij = summa_dist.mul_summa(x, self.weights_ij, self.rank, self.world_size, self.comm)
        Cij = self.summa(x, self.weights_ij, self.comm)
        
        return Cij
    
class SUMMA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, comm):
        ctx.save_for_backward(x, W)
        ctx.comm = comm
        
        return summa_dist.ABT(x, W, n, rank, world_size, comm)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        comm2 = ctx.comm
        #print(f"grad output {grad_output.shape}, weight {weight.shape}")
        grad_input = summa_dist.mul_summa(grad_output, weight, rank, world_size, comm)
        grad_weight = summa_dist.ATB(grad_output, input, n=n, rank=rank, world_size=world_size, comm=comm2).clone()
        
        return grad_input, grad_weight, None
        

class DistributedMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, comm):
        super().__init__()
        self.linear1 = DistributedLinear(in_dim, hidden_dim, comm)
        self.linear2 = DistributedLinear(hidden_dim, hidden_dim, comm)
        self.linear3 = DistributedLinear(hidden_dim, out_dim, comm)
        self.activation = GELU()
        self.drop = Dropout(p=1e-6)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        #x = self.drop(x)
        return x



comm, i, j = summa_dist.init_mpi()
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



mlp = DistributedMLP(1, 6, 1, comm)

#if rank == 0:
#    for param in mlp.parameters():
#        print(param.shape)

lr = 1e-3

optimizer = optim.SGD(mlp.parameters(), lr=lr)

for i in range(10000):
    optimizer.zero_grad()
    output = mlp(Xij)
    loss = loss_fn(output, goal_i)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        for param in mlp.parameters():
            print(f"rank {rank}, output {output})#, param {param[0]}")
