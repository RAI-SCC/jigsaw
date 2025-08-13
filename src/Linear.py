import numpy as np
import torch
from einops import rearrange
from torch.nn import GELU, Dropout

import distributed_matmul as dist_mm


def select_xw_transpose_kernel(n_parallel: int):
    """Return correct xw_t kernel."""
    if n_parallel == 2:
        return dist_mm.xw_transpose_twoway
    elif n_parallel == 4:
        return dist_mm.xw_transpose_fourway
    elif n_parallel == 8:
        return dist_mm.xw_transpose_eightway  
    
def select_xw_kernel(n_parallel: int):
    """Return xw kernel."""
    if n_parallel == 2:
        return dist_mm.xw_twoway
    elif n_parallel == 4:
        return dist_mm.xw_fourway
    elif n_parallel == 8:
        return dist_mm.xw_eightway

def select_x_transpose_w_kernel(n_parallel: int):
    """Return x_tw kernel."""
    if n_parallel == 2:
        return dist_mm.x_transpose_w_twoway
    elif n_parallel == 4:
        return dist_mm.x_transpose_w_fourway
    elif n_parallel == 8:
        return dist_mm.x_transpose_w_eightway

def select_x_transpose_kernel(n_parallel: int):
    """Return x_tw kernel."""
    if n_parallel == 2:
        return dist_mm.x_transpose_twoway
    elif n_parallel == 4:
        return dist_mm.x_transpose_fourway
    elif n_parallel == 8:
        return dist_mm.x_transpose_eightway

class DistributedXWT(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
            self, 
            in_dim: int, 
            out_dim: int, 
            rank: int, 
            device, n_parallel, model_group, n_channels=1, bias=True, stdev=None, init_method='linear'):
        """
        Initialize the XWT Linear layer sharded over two processes.

        Parameters
        ----------
        n_parallel: int
            number of processes it is distributed over
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        n_channels : int, optional
            The number of channels (default is 1). Only used for parameter initialization.
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        stdev : float, optional
            The standard deviation for weight initialization (default is None).
        init_method : str, optional
            The method for weight initialization, either 'linear' or 'conv' (default is 'linear').
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if init_method == 'linear':
            stdev = 1. / np.sqrt(in_dim*2) # TODO: should this 2 be n_parallel?
            self.weights = torch.nn.parameter.Parameter(torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev))
        elif init_method == 'conv':
            stdev = np.sqrt(6.0)/(8*8*n_channels) # TODO
            self.weights = torch.nn.parameter.Parameter(torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev))
        if bias and n_parallel == 2:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim//2).uniform_(-stdev, stdev))
        elif bias and n_parallel == 4:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev))
        elif bias and n_parallel == 8:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim*2).uniform_(-stdev, stdev))
        else:
            self.bias = None
        
        self.device = device
        self.rank = rank
        self.n_parallel = n_parallel
        self.model_group = model_group

        self.XWT = XWT().apply
    
    def forward(self, x):
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
            B, out_dim are global values.
            in_dim is local value.
        """
        Cij = self.XWT(x, self.weights, self.bias, self.rank, self.device, self.n_parallel, self.model_group)
        
        return Cij
    
class XWT(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""
    @staticmethod
    def forward(ctx, x, W, b,  rank, device, n_parallel, model_group):
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        W : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim).
        b : torch.Tensor
            Local bias tensor of shape (out_dim).
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        n_parallel : int
            Number of processes to distribute operation across
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        
        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
            B, out_dim are global values.
            in_dim is local value.
        """
        ctx.save_for_backward(x, W)
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.model_group = model_group
                
        xwt = select_xw_transpose_kernel(n_parallel)(x, W, rank, device, model_group)
        ctx.b = b

        if b is None:
            return xwt
        else:
            return xwt.add(b)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of linear layer (XWT) distributed across two processes.
        
        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : tensor
            The gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        input, weight = ctx.saved_tensors
        b = ctx.b
        rank = ctx.rank
        device = ctx.device
        n_parallel = ctx.n_parallel
        model_group = ctx.model_group
        
        grad_input  = select_xw_kernel(n_parallel)(grad_output, weight, rank=rank, device=device, model_group=model_group)
        grad_weight = select_x_transpose_w_kernel(n_parallel)(grad_output, input, rank=rank, device=device, model_group=model_group)
        grad_b = grad_output

        if b is None:
            return grad_input, grad_weight, None, None, None, None, None
        else:
            return grad_input, grad_weight, grad_b, None, None, None, None
        
class DistributedXW(torch.nn.Module):
    """Batched matrix-matrix (XW) multiplication distributed across two processes."""

    def __init__(self, in_dim, out_dim, rank, device, n_parallel, model_group, n_channels=1, bias=True, stdev=None, init_method='linear'):
        """
        Initialize distributed matrix-matrix multiplication module.
        
        n_parallel: int
            Number of processes the operation is distributed over
        in_dim: int
            Global input dimension of local weight matrix.
        out_dim: int
            Local output dimension of local weight matrix.
        device: torch.device
            Device to perform computation on.
        rank : int
            Global rank
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        stdev = 1. / np.sqrt(in_dim*out_dim*n_parallel)
        
        self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev)) if bias else None

        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group

        self.XW = XW().apply
    
    def forward(self, x):
        """
        Forward pass of linear layer (XW) distributed across two processes.

        x: torch.Tensor
            local input tensor of shape (B, x_in, x_out)
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        C: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim).
            B, x_in are global values.
            out_dim is local value.
        """
        Cij = self.XW(x, self.weights, self.bias, self.rank, self.device, self.n_parallel, self.model_group)
        
        return Cij
    
class XW(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XW) multiplication distributed across two processes."""

    @staticmethod
    def forward(ctx, x, w, b, rank, device, n_parallel, model_group):
        """
        Forward pass of linear layer (XW) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
            B, x_in are global dimensions.
            in_dim is local dimension.
        W : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim).
            in_dim is global dimension.
        b : torch.Tensor
            Local bias tensor of shape (out_dim)
        device : Device to perform computation on.
        rank : int
            Global rank
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        C: torch.Tensor 
            Local output tensor of shape (B, x_in, out_dim).
            B, x_in are global values.
            out_dim is local value.            
        """
        ctx.save_for_backward(x, w)
        ctx.b = b
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.model_group = model_group
        
        xw = select_xw_kernel(n_parallel)(x, w, rank, device, model_group)
        if b is None:
            return xw
        else:
            return xw.add(b)
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of linear layer (XW) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        input, weight = ctx.saved_tensors
        rank = ctx.rank
        device = ctx.device
        n_parallel = ctx.n_parallel
        model_group = ctx.model_group

        grad_output = grad_output.contiguous() # TODO: needed add after introduction of multi-head attention valueconcat
        grad_input  = select_xw_transpose_kernel(n_parallel)(grad_output, weight, rank=rank, device=device, model_group=model_group)
        grad_weight = select_x_transpose_w_kernel(n_parallel)(input, grad_output, rank=rank, device=device, model_group=model_group)
        grad_bias   = grad_output

        if ctx.b is None:
            return grad_input, grad_weight, None, None, None, None, None
        else:
            return grad_input, grad_weight, grad_bias, None, None, None, None


class DistributedXTW(torch.nn.Module):
    """Batched matrix-matrix (XTW) linear layer distributed across n processes."""

    def __init__(self, in_dim, out_dim, rank, device, n_parallel, model_group, n_channels=72, bias=True, init_method='linear'):
        """
        n_parallel: int
            Number of processes to distribute over
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        device: torch.device 
            Device to perform computation on.
        rank : int
            Global rank.
        n_channels : int, optional
            Number of channels for weight initialization in 'conv' mode. Default is 72.
        bias : bool, optional
            Whether to include a bias term. Default is True.
        init_method : str, optional
            Initialization method for weights ('linear' or 'conv'). Default is 'linear'.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if init_method == 'linear':
            stdev = 1. / np.sqrt(in_dim*2) # TODO: 2 or n_parallel?
            self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        elif init_method == 'conv':
            stdev = np.sqrt(6.0)/(60*n_channels) # TODO: 60?
            self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev))
        else:
            self.bias = None

        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group
        
        self.XTW = XTW().apply
    
    def forward(self, x):
        """
        Forward pass of linear layer (XTW) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        
        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim).
            B, x_in are global values.
            out_dim is local value.
        """
        Cij = self.XTW(x, self.weights, self.bias, self.rank, self.device, self.n_parallel, self.model_group)
        
        return Cij
    
class XTW(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XTW) multiplication distributed across two processes."""

    @staticmethod
    def forward(ctx, x, w, b, rank, device, n_parallel, model_group):
        """
        Forward pass of linear layer (XTW) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        W : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim).
        b : torch.Tensor
            Local bias tensor of shape (out_dim).
            Bias vector.
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        xtw: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim).
            B, x_in are global values.
            out_dim is local value.
        """
        ctx.save_for_backward(x, w)
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.model_group = model_group
        
        xtw = select_x_transpose_w_kernel(n_parallel)(x, w, rank, device, model_group)

        return xtw.add(b) 
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of linear layer (XTW) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : torch.Tensor
            The gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        input, weight = ctx.saved_tensors        
        rank = ctx.rank 
        device = ctx.device
        n_parallel = ctx.n_parallel
        model_group = ctx.model_group 

        grad_input = select_xw_transpose_kernel(n_parallel)(weight, grad_output, rank, device, model_group)
        grad_weight = select_xw_kernel(n_parallel)(input, grad_output, rank, device, model_group)
        grad_b = grad_output
        return grad_input, grad_weight, grad_b, None, None, None, None
    
class DistributedXTranpose(torch.nn.Module):
    """Batched transpose (XT) operation distributed across n processes."""

    def __init__(self, rank, device, n_parallel, model_group):
        """Initialize the Linear class."""
        super().__init__()
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group
        self.transpose = XT().apply

    def forward(self, x):
        """
        Forward pass of transpose (XT) distributed across two processes.
        Parameters.
        ----------

        X : torch.Tensor
            The input tensor.
        rank : int
            Global rank.
        device : torch.device
            Device to perform computation on.
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        Returns
        -------
        XT: torch.Tensor
            Local transposed tensor.
        """
        x_transpose = self.transpose(x, self.rank, self.device, self.n_parallel, self.model_group)
        return x_transpose

class XT(torch.autograd.Function):
    """
    Autograd function for batched transpose (XT) operation distributed across two processes.
    """
    @staticmethod
    def forward(ctx, x, rank, device, n_parallel, model_group):
        """
        Forward pass of transpose (XT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        x : torch.Tensor
            The input tensor.
        rank : int
            Global rank.
        device : torch.device
            The device on which the computation is performed.
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        Returns
        -------
        XT: torch.Tensor
            Local transposed tensor.
        """

        ctx.device = device
        ctx.rank = rank
        ctx.n_parallel = n_parallel
        ctx.model_group = model_group

        x = select_x_transpose_kernel(n_parallel)(x, rank, device, model_group)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of transpose (XT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : tensor
            Gradient of the loss with respect to the output of the layer.
        Returns
        -------
        grad_output:
            Transposed gradient of the loss with respect to the input of the layer.
        """

        rank = ctx.rank
        device = ctx.device
        n_parallel = ctx.n_parallel
        model_group = ctx.model_group

        grad_output = select_x_transpose_kernel(n_parallel)(grad_output, rank, device, model_group)
        return grad_output, None, None, None, None
    
class DistributedLayerNorm(torch.nn.Module): 
    """Batched layer norm distributed across four processes."""

    def __init__(self, dim, rank, device, model_group):
        """
        Initialize layer normalization.
        Parameters.
        ----------
        dim : int
            The dimension of the input features.
        rank : int
            Global rank.
        device : torch.device
            Device to perform computation on.
        """
        super().__init__()

        self.rank = rank
        self.device = device
        self.model_group = model_group
        self.layer_norm = torch.nn.LayerNorm(dim).to(device)
        self.layer_norm_apply = DistributedLN.apply
    
    def forward(self, x):
        """Forward pass of layer norm distributed across n processes."""
        x = self.layer_norm_apply(x)
        x = self.layer_norm(x)

        return x
    
class DistributedLN(torch.autograd.Function):
    """Autograd function for layer norm function distributed across four processes."""

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of layer norm distributed across four processes.

        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        """
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of layer norm distributed across four processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output: torch.torch.Tensor
            Gradient of the loss with respect to the input of the layer.
        """
        return grad_output, None, None
    
class DistributedSoftMax(torch.nn.Module):
    """Distributed Softmax operation."""

    #TODO: check for correctness
    def __init__(self, dim=-1, group=None):
        super().__init__()
        self.dim = dim
        self.group = group if group is not None else torch.distributed.group.WORLD
        self.func = _DistributedSoftmax.apply

    def forward(self, x):
        return self.func(x, self.dim, self.group)

class _DistributedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, group):
        """Overloaded Autograd function for distirbuted SoftMax."""
        # store context
        ctx.dim = dim
        ctx.group = group

        # local exponentials

        y = torch.exp(x)#-x.amax(dim=dim, keepdim=True)) # softmax fails with norm if only 1 variable
        local_sum = y.sum(dim=dim, keepdim=True)
        # global sum
        global_sum = local_sum.clone()

        torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM, group=group)
        
        # final softmax
        out = y / global_sum
        ctx.save_for_backward(out, global_sum)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        out, global_sum = ctx.saved_tensors
        dim = ctx.dim
        group = ctx.group

        # local grad numerator
        local_grad = grad_out * out
        local_sum = local_grad.sum(dim=dim, keepdim=True)
        # global sum for gradient
        global_grad_sum = local_sum.clone()
        torch.distributed.all_reduce(global_grad_sum, op=torch.distributed.ReduceOp.SUM, group=group)

        grad_x = local_grad - out * global_grad_sum
        return grad_x, None, None

class DistributedMatMulXWTranspose(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(self, rank, device, n_parallel, model_group):
        """Initialize the XWT Linear layer sharded over two processes."""
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group

        super().__init__()
        self.XWT = XWT().apply

    def forward(self, x, w, b):
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
            B, out_dim are global values.
            in_dim is local value.
        """
        Cij = self.XWT(x, w, b, self.rank, self.device, self.n_parallel, self.model_group)
        
        return Cij
    
class DistributedMatMulXW(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across two processes."""

    def __init__(self, rank, device, n_parallel, model_group):
        """Initialize the XWT Linear layer sharded over n processes."""
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel # TODO: no bias term
        self.model_group = model_group

        super().__init__()
        self.XW = XW().apply
    
    def forward(self, x, w, b):
        """
        Forward pass of linear layer (XW) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
            B, out_dim are global values.
            in_dim is local value.
        """
        Cij = self.XW(x, w, b, self.rank, self.device, self.n_parallel, self.model_group)
        
        return Cij
    
class DistributedMLP(torch.nn.Module):
    """Distributed MLP across n processes."""

    def __init__(self, in_dim, hidden_dim, out_dim, rank, device, n_parallel, model_group, bias=True, dropout=1e-3):
        """
        Initialize the MLP module for either two- or four-way model parallelism.

        Parameters
        ----------
        in_dim : int
            Input dimension of MLP.
            Two-way model: global value.
            Four-way model: local value.
        hidden_dim : int
            Hidden dimension of MLP.
            Two-way model: global value.
            Four-way model: local value.
        out_dim : int
            Output dimension of MLP.
            Two-way model: global value.
            Four-way model: local value.
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        parallelism : int, optional
            Degree of parallelism (default is 4).
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        dropout : int, optional
            dropout rate. Default value is 1e-3.
        """
        super().__init__()
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group

        if n_parallel == 2:
            # in, hidden, out dim are global sizes for two-way model parallelism
            # required to maintain partitioning across final dimension
            self.linear1 = DistributedXWT(in_dim//2, hidden_dim, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
            self.linear2 = DistributedXWT(hidden_dim//2, out_dim, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
        elif n_parallel == 4:
            self.linear1 = DistributedXWT(in_dim, hidden_dim, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
            self.linear2 = DistributedXWT(hidden_dim, out_dim, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
        elif n_parallel == 8:
            self.linear1 = DistributedXWT(in_dim, hidden_dim, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
            self.linear2 = DistributedXWT(hidden_dim*2, out_dim, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)
        self.rank = rank
        
    def forward(self, x):
        """
        Forward pass of MLP distributed across two or four processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop2(x)
        
        return x

class DistributedXTWMLP(torch.nn.Module):
    """Distributed MLP layer where inputs are transposed."""

    def __init__(self, in_dim, hidden_dim, out_dim, rank, device, n_parallel, model_group, bias=True, dropout=1e-3):
        """
        Initialize the transposed XTW MLP module for either two- or four-way model parallelism.

        Parameters
        ----------
        in_dim : int
            Input dimension of MLP.
            Two-way model: global value.
            Four-way model: local value.
        hidden_dim : int
            Hidden dimension of MLP.
            Two-way model: global value.
            Four-way model: local value.
        out_dim : int
            Output dimension of MLP.
            Two-way model: global value.
            Four-way model: local value.
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        parallelism : int, optional
            The level of parallelism (default is 4).
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        """
        super().__init__()
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group

        if n_parallel == 2:
            self.linear1 = DistributedXTW(in_dim, hidden_dim//2, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
            self.linear2 = DistributedXW(hidden_dim, out_dim//2, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
        elif n_parallel == 4:
            self.linear1 = DistributedXTW(in_dim, hidden_dim, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
            self.linear2 = DistributedXW(hidden_dim, out_dim, rank, device, n_parallel, model_group, bias=bias)
        elif n_parallel == 8:
            self.linear1 = DistributedXTW(in_dim, hidden_dim, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
            self.linear2 = DistributedXW(hidden_dim*2, out_dim, rank, device, n_parallel, model_group, bias=bias)
        
        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)
        self.rank = rank

    def forward(self, x):
        """
        Forward pass of transposed MLP distributed across two or four processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x


class MixerBlock(torch.nn.Module):
    """Mixing block that performs token and channel mixing using two-way distributed MLPs and layer normalization."""
    
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, spatial_hidden_dim_fraction, channels_hidden_dim_fraction, rank, device, n_parallel, model_group):
        """
        Parameters
        ----------
        tokens_mlp_dim : int
            Global dimension of the tokens MLP.
        channels_mlp_dim : int
            Global dimension of the channels MLP.
        spatial_hidden_dim_fraction : float
            Fraction of the hidden dimension for spatial mixing.
        channels_hidden_dim_fraction : float
            Fraction of the hidden dimension for channel mixing.
        device : torch.device
            Device to run the computations on.
        rank : int
            Global rank.
        """  # noqa: D205
        super(MixerBlock, self).__init__()
        self.token_mixing = DistributedXTWMLP(tokens_mlp_dim, int(tokens_mlp_dim * spatial_hidden_dim_fraction), tokens_mlp_dim, rank, device, n_parallel=n_parallel, model_group=model_group)
        self.channel_mixing = DistributedMLP(channels_mlp_dim, int(channels_mlp_dim * channels_hidden_dim_fraction), channels_mlp_dim, rank, device, n_parallel=n_parallel, model_group=model_group)
        self.transpose = DistributedXTranpose(rank, device, n_parallel, model_group)
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group
        
        if n_parallel == 2:
            self.layer_norm1 = torch.nn.LayerNorm(channels_mlp_dim//2) 
            self.layer_norm2 = torch.nn.LayerNorm(channels_mlp_dim//2) 
        elif n_parallel == 4: ## TODO; not used at the moment
            self.layer_norm1 = DistributedLayerNorm(channels_mlp_dim, self.rank, self.device, self.model_group) 
            self.layer_norm2 = DistributedLayerNorm(channels_mlp_dim, self.rank, self.device, self.model_group) 
 
    def forward(self, x):
        """
        Forward pass of MixerBlock distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, n_patches, embed_dimension)
            = shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        torch.Tensor
            Output tensor after applying layer normalization, token mixing, 
            transposition, and channel mixing.
        """
        x_res = x
        y = self.layer_norm1(x)         
        y = self.token_mixing(y)
        y = self.transpose(y)
        x = x_res + y
        y = self.layer_norm2(x) 
        x = x + self.channel_mixing(y) 
        
        return x
    

class SequentialMLP(torch.nn.Module):
    """Multi-Layer Perceptron (MLP) with one hidden layer."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=1e-3):
        """
        Initialize MLP.

        ----------
        in_dim : int
            Dimension of the input features.
        hidden_dim : int
            Dimension of the hidden layer.
        out_dim : int
            Dimension of the output layer.
        dropout : float, optional
            Dropout probability, by default 1e-3.
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, out_dim)
        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)

    def forward(self, x):
        """Forward pass of MLP."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop2(x)
        
        return x

class MixerBlockSequential(torch.nn.Module):
    """
    A sequential Mixer block for token and channel mixing.
    
    Parameters.
    ----------
    tokens_mlp_dim : int
        Dimension of the token MLP.
    channels_mlp_dim : int
        Dimension of the channel MLP.
    spatial_hidden_dim_fraction : float
        Fraction to determine the hidden dimension size for the token MLP.
    channels_hidden_dim_fraction : float
        Fraction to determine the hidden dimension size for the channel MLP.
    dropout : float, optional
        Dropout probability, by default 1e-3.
    """
    
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, spatial_hidden_dim_fraction, channels_hidden_dim_fraction, dropout=1e-3):
        super(MixerBlockSequential, self).__init__()
        
        self.token_mixing = SequentialMLP(tokens_mlp_dim, int(tokens_mlp_dim * spatial_hidden_dim_fraction), tokens_mlp_dim, dropout=dropout)
        self.channel_mixing = SequentialMLP(channels_mlp_dim, int(channels_mlp_dim * channels_hidden_dim_fraction), channels_mlp_dim, dropout=dropout)
        self.layer_norm1 = torch.nn.LayerNorm(channels_mlp_dim)
        self.layer_norm2 = torch.nn.LayerNorm(channels_mlp_dim)
        
    def forward(self, x):
        """
        Forward pass of MixerBlock.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, n_patches, embed_dimension)
            = shape (B, x_in, in_dim).

        Returns
        -------
        x : torch.Tensor
            Output tensor after applying layer normalization, token mixing, and channel mixing.
        """
        x_res = x.clone()
        y = self.layer_norm1(x)
        y = y.mT
        y = self.token_mixing(y)
        y = y.mT
        x = x_res + y
        y = self.layer_norm2(x)
        x = x + self.channel_mixing(y)
        
        return x
    
class DistributedAttention(torch.nn.Module):
  def __init__(self, x_items, x_features, embed_dim, output_dim, heads, rank, device, n_parallel, model_group):
    super().__init__()
    self.rank = rank
    self.device = device
    self.n_parallel = n_parallel
    self.model_group = model_group
    
    self.x_items = x_items
    self.embed_dim = embed_dim
    self.x_features = x_features
    
    if n_parallel == 2:
        self.qkv = DistributedXW(x_features, embed_dim//2*3, self.rank, self.device, self.n_parallel, self.model_group,  n_channels=1, init_method='linear', bias=False)
    elif n_parallel == 4:
        self.qkv = DistributedXW(x_features, embed_dim*3, self.rank, self.device, self.n_parallel, self.model_group,  n_channels=1, init_method='linear', bias=False)
    self.softmax = DistributedSoftMax(dim=-1, group=self.model_group) # TODO: define softmax groups
    self.xwt = DistributedMatMulXWTranspose(self.rank, self.device, self.n_parallel, model_group=self.model_group) # model group TODO 
    self.xw = DistributedMatMulXW(rank, device, self.n_parallel, model_group=self.model_group) # model group TODO 
    
    if n_parallel == 2:
        self.concatenator = DistributedXW(embed_dim, output_dim//2, self.rank, self.device, self.n_parallel, self.model_group, n_channels=1, init_method='linear', bias=False)
    else:
        self.concatenator = DistributedXW(embed_dim, output_dim, self.rank, self.device, self.n_parallel, self.model_group, n_channels=1, init_method='linear', bias=False)
    
    self.heads = heads
    self.scale = (embed_dim // self.heads) ** (-0.5) # should be (embed_dim/heads)
    

  def forward(self, x):
    """
    x: (n_batch, vars, n_lat, n_lon)
    """
    # Load constant masks from the disc
    
    # x_local: [n_batch, n_lat, n_lon, variables]
    x_patches_local = x

    qkv = self.qkv(x_patches_local) # shape: [n_batches, spatial_tokens, channel_tokens]
    q, k, v = torch.tensor_split(qkv, 3, dim=2)
    
    q = rearrange(q, 'n i (h d) -> n h i d', h=self.heads).contiguous()
    k = rearrange(k, 'n i (h d) -> n h i d', h=self.heads).contiguous()
    v = rearrange(v, 'n i (h d) -> n h i d', h=self.heads).contiguous()
    
    q = self.scale * q
    attention = self.xwt(q, k, None)
    attention = self.softmax(attention) # TODO: groups for softmax also need to be determined. probably same partners as layernorm
    values = self.xw(attention, v, None)    
    values = rearrange(values, 'n h s d -> n s (h d)').contiguous()
    
    values = self.concatenator(values)
    #values = self.mlp(values, model_group=None)
    # Output: [n_batch, seq_length, output_dim]
    return values