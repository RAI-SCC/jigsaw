import torch
import summa_gpu
from mpi4py import MPI
import numpy as np
from torch.nn import GELU, Dropout

import utils
from einops import rearrange

class DistributedLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, comm, device, torch_rank):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights_ij = torch.nn.parameter.Parameter(torch.normal(0, 1, size=(out_dim, in_dim)))
        self.comm = comm
        self.device = device
        self.torch_rank = torch_rank
        self.world_size = self.comm.Get_size()
        self.summa = SUMMA().apply
    
    def forward(self, x):
        Cij = self.summa(x, self.weights_ij, self.comm, self.device, self.torch_rank)
        
        return Cij
    
class SUMMA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, comm, device, torch_rank):
        ctx.save_for_backward(x, W)
        ctx.comm = comm
        ctx.device = device
        ctx.torch_rank = torch_rank
        ctx.world_size = 4
        assert(x.shape[-1] == W.shape[-1])
        #print(f"Forward pass, my device {device}, world size {torch.cuda.device_count()}, my_torch_rank {torch_rank}")
        return summa_gpu.ABT(x, W, torch_rank, ctx.world_size, device)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        
        device=ctx.device        
        torch_rank = ctx.torch_rank
        world_size = ctx.world_size

        grad_input = summa_gpu.mul_summa(grad_output, weight, rank=torch_rank, world_size=world_size, device=device)
        grad_weight = summa_gpu.ATB(grad_output, input, rank=torch_rank, world_size=world_size, device=device).clone()
        
        return grad_input, grad_weight, None, None, None
         

class DistributedMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, comm, device, torch_rank):
        super().__init__()
        self.linear1 = DistributedLinear(in_dim, hidden_dim, comm, device, torch_rank)
        
        self.linear2 = DistributedLinear(hidden_dim, out_dim, comm, device, torch_rank)
        self.activation = GELU()
        self.drop = Dropout(p=1e-6)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


class MLPAll(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = MLPDimension(192, 192, 192)
        self.mlp2 = MLPDimension(16200, 8100, 16200)
        

    def forward(self, x):
        """
        x in : (n_batch, 69, 360, 720)
        """
        x_res = x.clone()

        x = self.mlp1(x) # output : [n_batch, 90*180, vars]
        x = x.permute(0, 2, 1) # output : [n_batch, var, 90*180]
        x = self.mlp2(x)
        x = x.permute(0, 2, 1) # output: [n_batch, 90*180, vars]
        
        return x 
class MLPDimension(torch.nn.Module):
    def __init__(self, in_dim, mlp_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, mlp_dim)
        self.fc2 = torch.nn.Linear(mlp_dim, in_dim)

    def forward(self, x):
        y = torch.nn.functional.gelu(self.fc1(x))
        return self.fc2(y)

class MixerBlock(torch.nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, hidden_dim):
        super(MixerBlock, self).__init__()
        self.token_mixing = MLPDimension(tokens_mlp_dim, hidden_dim)
        self.channel_mixing = MLPDimension(channels_mlp_dim, hidden_dim)
        self.layer_norm1 = torch.nn.LayerNorm(channels_mlp_dim)
        self.layer_norm2 = torch.nn.LayerNorm(channels_mlp_dim)

    def forward(self, x):
        y = self.layer_norm1(x)
        y = rearrange(y, 'n p c -> n c p')
        y = self.token_mixing(y)
        y = rearrange(y, 'n c p -> n p c')
        x = x + y
        y = self.layer_norm2(x)
        
        return x + self.channel_mixing(y)




