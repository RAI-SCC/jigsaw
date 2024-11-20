import torch
import summa_gpu
from mpi4py import MPI
import numpy as np
from torch.nn import GELU, Dropout

import utils
from einops import rearrange

# TODO: initialize all parameters in NN
class DistributedXW(torch.nn.Module):
    def __init__(self, in_dim, out_dim, comm, device, torch_rank):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights_ij = torch.nn.parameter.Parameter(torch.normal(0, 1, size=(in_dim, out_dim)))
        self.comm = comm
        self.device = device
        self.torch_rank = torch_rank
        self.world_size = self.comm.Get_size()
        self.summa = SUMMA_AB().apply
    
    def forward(self, x):
        Cij = self.summa(x, self.weights_ij, self.comm, self.device, self.torch_rank)
        
        return Cij
    
class SUMMA_AB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, comm, device, torch_rank):
        ctx.save_for_backward(x, W)
        ctx.comm = comm
        ctx.device = device
        ctx.torch_rank = torch_rank
        ctx.world_size = 4
        assert(x.shape[-1] == W.shape[0])
        #print(f"Forward pass, my device {device}, world size {torch.cuda.device_count()}, my_torch_rank {torch_rank}")
        return summa_gpu.AB(x, W, torch_rank, ctx.world_size, device)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        
        device=ctx.device        
        torch_rank = ctx.torch_rank
        world_size = ctx.world_size

        grad_input = summa_gpu.ABT(grad_output, weight, rank=torch_rank, world_size=world_size, device=device)
        grad_weight = summa_gpu.ATB(input, grad_output, rank=torch_rank, world_size=world_size, device=device).clone()
        
        return grad_input, grad_weight, None, None, None
         
class DistributedXTW(torch.nn.Module):
    def __init__(self, in_dim, out_dim, comm, device, torch_rank):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights_ij = torch.nn.parameter.Parameter(torch.normal(0, 1, size=(in_dim, out_dim)))
        self.comm = comm
        self.device = device
        self.torch_rank = torch_rank
        self.world_size = self.comm.Get_size()
        self.summa = SUMMA_ATB().apply
    
    def forward(self, x):
        Cij = self.summa(x, self.weights_ij, self.comm, self.device, self.torch_rank)
        
        return Cij
    
class SUMMA_ATB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, comm, device, torch_rank):
        ctx.save_for_backward(x, W)
        ctx.comm = comm
        ctx.device = device
        ctx.torch_rank = torch_rank
        ctx.world_size = 4
        #assert(x.shape[-1] == W.shape[0]) # TODO ASSERT
        
        return summa_gpu.ATB(x, W, torch_rank, ctx.world_size, device)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        
        device=ctx.device        
        torch_rank = ctx.torch_rank
        world_size = ctx.world_size

        grad_input = summa_gpu.ABT(weight, grad_output, rank=torch_rank, world_size=world_size, device=device)
        grad_weight = summa_gpu.AB(input, grad_output, rank=torch_rank, world_size=world_size, device=device).clone()
        
        return grad_input, grad_weight, None, None, None
    
class DistributedXWT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, comm, device, torch_rank):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights_ij = torch.nn.parameter.Parameter(torch.normal(0, 1, size=(out_dim, in_dim)))
        self.comm = comm
        self.device = device
        self.torch_rank = torch_rank
        self.world_size = self.comm.Get_size()
        self.summa = SUMMA_ABT().apply
    
    def forward(self, x):
        Cij = self.summa(x, self.weights_ij, self.comm, self.device, self.torch_rank)
        
        return Cij
    

class SUMMA_ABT(torch.autograd.Function):
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

        grad_input = summa_gpu.AB(grad_output, weight, rank=torch_rank, world_size=world_size, device=device)
        grad_weight = summa_gpu.ATB(grad_output, input, rank=torch_rank, world_size=world_size, device=device).clone()
        
        return grad_input, grad_weight, None, None, None

class DistributedTranspose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose = DistributedXT().apply

    def forward(self, X, rank, device):
        X = self.transpose(X, rank, device)
        return X

class DistributedXT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rank, device):
        ctx.device = device
        ctx.rank = rank
        x = summa_gpu.matrix_transpose(x, rank, device)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        rank = ctx.rank
        device = ctx.device
        grad_output = summa_gpu.matrix_transpose(grad_output, rank, device)
        return grad_output, None, None

class DistributedMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, comm, device, torch_rank):
        super().__init__()
        self.linear1 = DistributedXWT(in_dim, hidden_dim, comm, device, torch_rank)
        self.linear2 = DistributedXWT(hidden_dim, out_dim, comm, device, torch_rank)
        self.activation = GELU()
        self.drop = Dropout(p=1e-5)
        self.rank = torch_rank

    def forward(self, x):
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x

class DistributedXTWMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, comm, device, torch_rank):
        super().__init__()
        self.linear1 = DistributedXTW(in_dim, hidden_dim, comm, device, torch_rank)
        self.linear2 = DistributedXTW(hidden_dim, out_dim, comm, device, torch_rank)
        self.activation = GELU()
        self.drop = Dropout(p=1e-5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
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
    """
    Mixer Block
    """
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, hidden_dim, comm, device, torch_rank):
        super(MixerBlock, self).__init__()
        self.token_mixing = DistributedXTWMLP(tokens_mlp_dim, hidden_dim, tokens_mlp_dim, comm, device, torch_rank) # TODO: for now, Linear
        self.channel_mixing = DistributedMLP(channels_mlp_dim, hidden_dim, channels_mlp_dim, comm, device, torch_rank)
        self.transpose = DistributedTranspose()
        self.layer_norm1 = torch.nn.LayerNorm(channels_mlp_dim)
        self.layer_norm2 = torch.nn.LayerNorm(channels_mlp_dim)
        self.rank = torch_rank
        self.device = device

    def forward(self, x):
        

        y = self.layer_norm1(x)
        #y = rearrange(y, 'n p c -> n c p')
        x_res = y.clone()
        y = self.token_mixing(y)
        
        y = self.transpose(y, self.rank, self.device)
        #y = rearrange(y, 'n c p -> n p c')
        
        y = self.layer_norm2(y)
    
        x = x_res + self.channel_mixing(x_res + y)
        
        return x




