import torch
import summa_dist
from mpi4py import MPI
import numpy as np
from torch.nn import ReLU, Dropout
import utils

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
    
    def forward(self, x):
        Cij = self.summa(x, self.weights_ij, self.comm)
        
        return Cij
    
class SUMMA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, comm):
        ctx.save_for_backward(x, W)
        ctx.comm = comm
        assert(x.shape[1] == W.shape[1])
        return summa_dist.ABT(x, W, comm.Get_rank(), comm.Get_size(), comm)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        comm = ctx.comm
        
        grad_input = summa_dist.mul_summa(grad_output, weight, comm.Get_rank(), comm.Get_size(), comm)
        
        grad_weight = summa_dist.ATB(grad_output, input, rank=comm.Get_rank(), world_size=comm.Get_size(), comm=comm).clone()
        
        return grad_input, grad_weight, None
         

class DistributedMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, comm):
        super().__init__()
        self.linear1 = DistributedLinear(in_dim, hidden_dim, comm)
        self.linear2 = DistributedLinear(hidden_dim, hidden_dim, comm)
        self.linear3 = DistributedLinear(hidden_dim, out_dim, comm)
        self.activation = ReLU()
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

