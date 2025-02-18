import torch
import summa_gpu
#from mpi4py import MPI
import numpy as np
from torch.nn import GELU, Dropout

class DistributedXW_fourway(torch.nn.Module):
    """
    Torch Module to run distributed matrix-matrix multiplication.
    """
    def __init__(self, in_dim, out_dim, comm, device, rank, bias=True):
        """
        Initialize distributed matrix-matrix multiplication module.

        in_dim: int
            input dimension of local weight matrix
        out_dim: int
            output dimension of local weight matrix
        comm: MPI communicator
        device: string
            name of device
        rank: int
            GPU rank        
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        stdev = 1. / np.sqrt(in_dim*out_dim*4)
        
        self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev)) if bias else None
        self.comm = comm
        self.device = device
        self.rank = rank
        self.AB = AB_fourway().apply
    
    def forward(self, x):
        """
        Forward pass.

        x: Tensor
            local input tensor of shape (B, x_in, x_out)

        Returns
        -------
        C: Tensor
            of shape (B, x_in, out_dim)
        """
        Cij = self.AB(x, self.weights, self.bias, self.comm, self.device, self.rank)
        
        return Cij
    
class AB_fourway(torch.autograd.Function):
    """
    Autograd function of forward pass for distributed matrix-matrix multiplication
    """
    @staticmethod
    def forward(ctx, x, W, b, comm, device, rank):
        """
        forward call.

        x: Tensor
            of shape (B, in_dim, out_dim)
        W: Tensor
            of shape (out_dim, out_dim2)
        device: String
        rank: 
            
        """
        ctx.save_for_backward(x, W)
        ctx.comm = comm
        ctx.device = device
        ctx.rank = rank
        ctx.world_size = 4
        assert(x.shape[-1] == W.shape[0])
        AB = summa_gpu.AB_custom(x, W, rank, ctx.world_size, device)
        if b is not None:
            return AB.add(b)
        else:
            return AB
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        
        device=ctx.device        
        rank = ctx.rank
        world_size = ctx.world_size

        grad_input  = summa_gpu.ABT_custom_precompute(grad_output, weight, rank=rank, n_gpus_per_node=world_size, device=device)
        grad_weight = summa_gpu.ATB_custom_precompute(input, grad_output, rank=rank, n_gpus_per_node=world_size, device=device)
        grad_b      = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_b, None, None, None
    
class DistributedXW_twoway(torch.nn.Module):
    """
    Torch Module to run distributed matrix-matrix multiplication.
    """
    def __init__(self, in_dim, out_dim, comm, device, rank, n_channels=1, bias=True, stdev=None, init_method='linear'):
        """
        Initialize distributed matrix-matrix multiplication module.

        in_dim: int
            input dimension of local weight matrix
        out_dim: int
            output dimension of local weight matrix
        comm: MPI communicator
        device: string
            name of device
        rank: int
            GPU rank        
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        stdev = 1. / np.sqrt(in_dim*out_dim*4)
        
        self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev)) if bias else None
        self.AB_twoway = AB_twoway().apply
        
        self.device = device
        self.rank = rank
    
    def forward(self, x):
        """
        Forward pass.

        x: Tensor
            local input tensor of shape (B, x_in, x_out)

        Returns
        -------
        C: Tensor
            of shape (B, x_in, out_dim)
        """
        Cij = self.AB_twoway(x, self.weights, self.bias, self.device, self.rank)
        
        return Cij
    
class AB_twoway(torch.autograd.Function):
    """
    Autograd function of forward pass for distributed matrix-matrix multiplication
    """
    @staticmethod
    def forward(ctx, x, W, b, device, rank):
        """
        forward call.

        x: Tensor
            of shape (B, in_dim, out_dim)
        W: Tensor
            of shape (out_dim, out_dim2)
        device: String
        rank: 
            
        """
        ctx.save_for_backward(x, W)
        ctx.device = device
        ctx.rank = rank
        
        if b is not None:
            return summa_gpu.AB_twoway(x, W, device, rank).add(b)
        else:
            return summa_gpu.AB_twoway(x, W, device, rank)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        rank = ctx.rank
        device = ctx.device

        grad_input  = summa_gpu.ABT_twoway(grad_output, weight, rank=rank, device=device)
        grad_weight = summa_gpu.ATB_twoway(input, grad_output, rank=rank, device=device)
        grad_bias   = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None, None
         
class DistributedXTW_fourway(torch.nn.Module):
    def __init__(self, in_dim, out_dim, comm, device, rank, n_channels=72, bias=True, init_method='linear'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if init_method == 'linear':
            stdev = 1. / np.sqrt(in_dim*2)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        elif init_method == 'conv':
            stdev = np.sqrt(6.0)/(60*n_channels)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev))
        else:
            self.bias = None
        self.comm = comm
        self.device = device
        self.rank = rank
        self.world_size = 4
        self.ATB = ATB_four_way().apply
    
    def forward(self, x):
        Cij = self.ATB(x, self.weights, self.bias, self.comm, self.device, self.rank) 
        
        return Cij
    
class ATB_four_way(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b, comm, device, rank):
        ctx.save_for_backward(x, W)
        ctx.comm = comm
        ctx.device = device
        ctx.rank = rank
        ctx.world_size = 4
        
        xw = summa_gpu.ATB_custom_precompute(x, W, rank, ctx.world_size, device)    

        return xw.add(b)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        
        device = ctx.device        
        rank = ctx.rank
        world_size = ctx.world_size
        
        grad_input = summa_gpu.ABT_custom_precompute(weight, grad_output, rank=rank, n_gpus_per_node=world_size, device=device)
        grad_weight = summa_gpu.AB_custom(input, grad_output, rank=rank, n_gpus_per_node=world_size, device=device)
        grad_b = grad_output.sum(dim=0) 
        
        return grad_input, grad_weight, grad_b, None, None, None
    
class DistributedXTW_twoway(torch.nn.Module):
    def __init__(self, in_dim, out_dim, comm, device, rank, n_channels=72, bias=True, init_method='linear'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if init_method == 'linear':
            stdev = 1. / np.sqrt(in_dim*2)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        elif init_method == 'conv':
            stdev = np.sqrt(6.0)/(60*n_channels)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev))
        else:
            self.bias = None
        self.comm = comm
        self.device = device
        self.rank = rank
        self.world_size = 2
        
        self.ATB_twoway = ATB_twoway().apply
    
    def forward(self, x):
        
        Cij = self.ATB_twoway(x, self.weights, self.bias, self.device, self.rank )
        
        return Cij
    
class ATB_twoway(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b, device, rank):
        ctx.save_for_backward(x, W)
        ctx.device = device
        ctx.rank = rank
        
        
        xw = summa_gpu.ATB_twoway(x, W, device, rank)    

        return xw.add(b) 
    
    @staticmethod
    def backward(ctx, grad_output):


        input, weight = ctx.saved_tensors        
        device = ctx.device
        rank = ctx.rank 

        grad_input = summa_gpu.ABT_twoway(weight, grad_output, device, rank)
        grad_weight = summa_gpu.AB_twoway(input, grad_output, device, rank)
        grad_b = grad_output.sum(dim=0) 
        return grad_input, grad_weight, grad_b, None, None
    
class DistributedLayerNorm(torch.nn.Module):
    def __init__(self, dim, rank, device):
        super().__init__()
        
        self.layer_norm = torch.nn.LayerNorm(dim).to(device)
        self.layer_norm_apply = DistributedLN.apply
    
    def forward(self, x, rank, device):
        x = self.layer_norm_apply(x, rank, device)
        x = self.layer_norm(x)

        return x
    
class DistributedLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rank, device):
        ctx.device = device
        ctx.rank = rank
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        # super backward
        return grad_output, None, None

    
class DistributedXWT_fourway(torch.nn.Module):
    def __init__(self, in_dim, out_dim, comm, device, rank, n_channels=1, bias=True, stdev=None, init_method='linear'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if init_method == 'linear':
            stdev = 1. / np.sqrt(in_dim*2)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev))
        elif init_method == 'conv':
            stdev = np.sqrt(6.0)/(8*8*n_channels)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev))
        else:
            self.bias = None
        
        self.comm = comm
        self.device = device
        self.rank = rank
        self.world_size = 4
        self.ABT = ABT_fourway().apply
    
    def forward(self, x):
        
        Cij = self.ABT(x, self.weights, self.bias, self.comm, self.device, self.rank)
        
        return Cij
    

class ABT_fourway(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b, comm, device, rank):
        ctx.save_for_backward(x, W)
        ctx.comm = comm
        ctx.device = device
        ctx.rank = rank
        ctx.world_size = 4
        xw = summa_gpu.ABT_custom_precompute(x, W, rank, ctx.world_size, device)
        if b is not None:
            return xw.add(b)
        else:
            return xw
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        
        device=ctx.device        
        rank = ctx.rank
        world_size = ctx.world_size
        grad_input = summa_gpu.AB_custom(grad_output, weight, rank=rank, n_gpus_per_node=world_size, device=device)
        grad_weight = summa_gpu.ATB_custom_precompute(grad_output, input, rank=rank, n_gpus_per_node=world_size, device=device)
        grad_b = grad_output.sum(dim=0) 

        return grad_input, grad_weight, grad_b, None, None, None

class DistributedXWT_twoway(torch.nn.Module):
    def __init__(self, in_dim, out_dim, comm, device, rank, n_channels=1, bias=True, stdev=None, init_method='linear'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if init_method == 'linear':
            stdev = 1. / np.sqrt(in_dim*2)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev))
        elif init_method == 'conv':
            stdev = np.sqrt(6.0)/(8*8*n_channels)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim//2).uniform_(-stdev, stdev))
        else:
            self.bias = None

        self.comm = comm
        self.device = device
        self.rank = rank
        self.world_size = 4
        self.ABT = ABT_twoway().apply
    
    def forward(self, x):
        Cij = self.ABT(x, self.weights, self.bias, self.device, self.rank)
        
        return Cij
    

class ABT_twoway(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b,  device, rank):
        ctx.save_for_backward(x, W)
        ctx.device = device
        ctx.rank = rank
        
        xw = summa_gpu.ABT_twoway(x, W, device, rank)
        if b is not None:
            return xw.add(b)
        else:
            return xw
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        device = ctx.device
        rank = ctx.rank
        
        grad_input  = summa_gpu.AB_twoway(grad_output, weight, device=device, rank=rank)
        grad_weight = summa_gpu.ATB_twoway(grad_output, input, device=device, rank=rank)
        grad_b = grad_output.sum(dim=0) 

        return grad_input, grad_weight, grad_b, None, None, None

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

class DistributedTranspose_twoway(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose = DistributedXT_twoway().apply

    def forward(self, X, rank, device):
        X = self.transpose(X, rank, device)
        return X

class DistributedXT_twoway(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rank, device):
        ctx.device = device
        ctx.rank = rank
        x = summa_gpu.matrix_transpose_twoway(x, rank, device)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        rank = ctx.rank
        device = ctx.device
        grad_output = summa_gpu.matrix_transpose_twoway(grad_output, rank, device)
        return grad_output, None, None


class DistributedMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, comm, device, rank, parallelism=4, bias=True):
        super().__init__()
        if parallelism == 4:
            self.linear1 = DistributedXWT_fourway(in_dim, hidden_dim, comm, device, rank, bias=bias, init_method='linear')
            self.linear2 = DistributedXWT_fourway(hidden_dim, out_dim, comm, device, rank, bias=bias, init_method='linear')
        else:
            self.linear1 = DistributedXWT_twoway(in_dim//2, hidden_dim, comm, device, rank, bias=bias, init_method='linear')
            self.linear2 = DistributedXWT_twoway(hidden_dim//2, out_dim, comm, device, rank, bias=bias, init_method='linear')
        self.activation = GELU()
        self.drop = Dropout(p=1e-6)
        self.drop2 = Dropout(p=1e-6)
        self.rank = rank
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop2(x)
        
        return x

class DistributedXTWMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, comm, device, rank, parallelism=4, bias=True):
        super().__init__()
        if parallelism == 4:
            self.linear1 = DistributedXTW_fourway(in_dim, hidden_dim, comm, device, rank, bias=bias, init_method='linear')
            self.linear2 = DistributedXW_fourway(hidden_dim, out_dim, comm, device, rank, bias=bias)
        elif parallelism == 2:
            self.linear1 = DistributedXTW_twoway(in_dim, hidden_dim//2, comm, device, rank, bias=bias, init_method='linear')
            self.linear2 = DistributedXW_twoway(hidden_dim, out_dim//2, comm, device, rank, bias=bias, init_method='linear')
        self.activation = GELU()
        self.drop = Dropout(p=1e-6)
        self.drop2 = Dropout(p=1e-6)
        self.rank = rank

    def forward(self, x):
        #if self.rank == 0:
        #    print(f'Input shape {x.shape}')
        x = self.linear1(x)
        #if self.rank == 0:
        #    print(f'Output shape {x.shape}')
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x


class MixerBlock(torch.nn.Module):
    """
    Mixer Block
    """
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, hidden_dim, comm, device, rank, parallelism):
        super(MixerBlock, self).__init__()
        self.token_mixing = DistributedXTWMLP(tokens_mlp_dim, hidden_dim, tokens_mlp_dim, comm, device, rank, parallelism=parallelism) # TODO: for now, Linear
        self.channel_mixing = DistributedMLP(channels_mlp_dim, hidden_dim, channels_mlp_dim, comm, device, rank, parallelism=parallelism)
        self.transpose = DistributedTranspose()
        self.rank = rank
        self.device = device
        
        self.layer_norm1 = DistributedLayerNorm(channels_mlp_dim, self.rank, self.device)
        self.layer_norm2 = DistributedLayerNorm(channels_mlp_dim, self.rank, self.device)
 
    def forward(self, x):
        
        x_res = x.clone() # transpose here when the rest is doing stuff
        y = self.layer_norm1(x, self.rank, self.device)                                                                                                                                                                 
        
        # y shape before: [n_batch, n_patches, hidden_dim]
        y = self.token_mixing(y)
        # y shape after: [n_batches, hidden dim, n_patches]
        y = self.transpose(y, self.rank, self.device) # dont transpose
       
        x = x_res + y
        y = self.layer_norm2(x, self.rank, self.device) # would need to redefine layer norm to be dim = [1, channels_mlp]
        x = x + self.channel_mixing(y) # do XTW
        
        return x
    

class MixerBlock_twoway(torch.nn.Module):
    """
    Mixer Block
    """
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, hidden_dim, comm, device, rank):
        super(MixerBlock_twoway, self).__init__()
        self.token_mixing = DistributedXTWMLP(tokens_mlp_dim, hidden_dim, tokens_mlp_dim, comm, device, rank, parallelism=2)
        self.channel_mixing = DistributedMLP(channels_mlp_dim, hidden_dim, channels_mlp_dim, comm, device, rank, parallelism=2)
        self.transpose = DistributedTranspose_twoway()
        self.rank = rank
        self.device = device
        self.layer_norm1 = torch.nn.LayerNorm(hidden_dim//2) 
        self.layer_norm2 = torch.nn.LayerNorm(hidden_dim//2) 
 
    def forward(self, x):
        
        x_res = x.clone() 
        
        y = self.layer_norm1(x)         
        
        y = self.token_mixing(y)
        
        #x = x_res + y.mT
        y = self.transpose(y, self.rank, self.device)
        x = x_res + y

        y = self.layer_norm2(x) # would need to redefine layer norm to be dim = [1, channels_mlp]
        x = x + self.channel_mixing(y) # do XTW
        
        return x
    

class SequentialMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=1e-1):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, out_dim)
        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop2(x)
        
        return x

class MixerBlockSequential(torch.nn.Module):
    """
    Mixer Block
    """
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, hidden_dim, dropout=1e-6):
        super(MixerBlockSequential, self).__init__()
        self.token_mixing = SequentialMLP(tokens_mlp_dim, hidden_dim, tokens_mlp_dim, dropout=dropout)
        self.channel_mixing = SequentialMLP(channels_mlp_dim, hidden_dim, channels_mlp_dim, dropout=dropout)
        self.layer_norm1 = torch.nn.LayerNorm(channels_mlp_dim)
        self.layer_norm2 = torch.nn.LayerNorm(channels_mlp_dim)
        
    def forward(self, x):
        
        x_res = x.clone()
        y = self.layer_norm1(x)
        y = y.mT
        y = self.token_mixing(y)
        y = y.mT
        x = x_res + y
        y = self.layer_norm2(x)
        x = x + self.channel_mixing(y)
        
        return x







