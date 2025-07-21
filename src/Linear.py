import torch
import distributed_matmul as dist_mm
import numpy as np
from torch.nn import GELU, Dropout

class DistributedXW_fourway(torch.nn.Module):
    """
    Batched matrix-matrix (XW) multiplication distributed across four processes.
    """
    def __init__(self, in_dim, out_dim, device, rank, bias=True):
        """
        Initialize distributed XW multiplication.

        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        bias : bool, optional
            Default is True.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        stdev = 1. / np.sqrt(in_dim*out_dim*4) # standard deviation accommodates for size of global weight matrix
        
        self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev)) if bias else None
        self.device = device
        self.rank = rank
        self.XW = XW_fourway().apply
    
    def forward(self, x, model_group):
        """
        Forward pass of linear layer (XW) distributed across four processes.

        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        C : torch.Tensor
            Local output tensor of shape (B, x_in, out_dim).
        """
        Cij = self.XW(x, self.weights, self.bias, self.device, self.rank, model_group)
        
        return Cij
    
class DistributedXW_attention_fourway(torch.nn.Module):
    """
    Batched matrix-matrix (XWT) multiplication distributed across two processes.
    """
    def __init__(self, device, rank, model_group=None):
        """
        Initialize the XWT Linear layer sharded over two processes.
        """
        self.device = device
        self.rank = rank
        self.model_group = model_group # TODO: no bias term

        super().__init__()
        self.XW = XW_fourway().apply
    
    def forward(self, x, w):
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

        Cij = self.XW(x, w, None, self.device, self.rank, self.model_group)
        
        return Cij
    
class XW_fourway(torch.autograd.Function):
    """
    Autograd function for batched matrix-matrix (XW) multiplication distributed across four processes.
    """
    @staticmethod
    def forward(ctx, x, W, b, device, rank, model_group):
        """
        Forward pass of (XW) linear layer distributed across four processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        x : torch.Tensor
            local input tensor of shape (B, x_in, in_dim)
        W : torch.Tensor
            Local weight matrix of shape (in_dim, out_dim)
        b : torch.Tensor
            Local bias vector of shape (out_dim)
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        C : torch.Tensor
            Local output tensor of shape (B, x_in, out_dim).
        """
        
        ctx.save_for_backward(x, W)
        
        ctx.device = device
        ctx.rank = rank
        ctx.model_group = model_group
        ctx.b = b

        XW = dist_mm.XW_fourway(x, W, rank,  device, model_group)
        
        if b is None:
            return XW
        else:
            return XW.add(b)
        
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of (XW) linear layer distributed across four processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : torch.Tensor
            The gradient of the loss with respect to the output of the linear layer.
        Returns
        -------
        grad_input : torch.Tensor
            The gradient of the loss with respect to the input of the linear layer.
        grad_weight : torch.Tensor
            The gradient of the loss with respect to the weights of the linear layer.
        grad_b : torch.Tensor
            The gradient of the loss with respect to the bias of the linear layer.
        """
        
        input, weight = ctx.saved_tensors
        device = ctx.device        
        rank = ctx.rank
        model_group = ctx.model_group

        grad_input  = dist_mm.XWT_fourway(grad_output, weight, rank=rank, device=device, model_group=model_group)
        grad_weight = dist_mm.XTW_fourway(input, grad_output, rank=rank, device=device, model_group=model_group)
        grad_b      = grad_output

        if ctx.b is None:
            return grad_input, grad_weight, None, None, None, None
        else:
            return grad_input, grad_weight, grad_b, None, None, None
    
class DistributedXW_twoway(torch.nn.Module):
    """
    Batched matrix-matrix (XW) multiplication distributed across two processes.
    """
    def __init__(self, in_dim, out_dim, device, rank, n_channels=1, bias=True, stdev=None, init_method='linear'):
        """
        Initialize distributed matrix-matrix multiplication module.

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
        stdev = 1. / np.sqrt(in_dim*out_dim*2)
        
        self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev)) if bias else None
        self.XW_twoway = XW_twoway().apply
        
        self.device = device
        self.rank = rank
    
    def forward(self, x, model_group):
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
        Cij = self.XW_twoway(x, self.weights, self.bias, self.device, self.rank, model_group)
        
        return Cij
    
class XW_twoway(torch.autograd.Function):
    """
    Autograd function for batched matrix-matrix (XW) multiplication distributed across two processes.
    """
    @staticmethod
    def forward(ctx, x, W, b, device, rank, model_group):
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
        ctx.save_for_backward(x, W)
        ctx.device = device
        ctx.rank = rank
        ctx.model_group = model_group
        ctx.b = b
        
        if b is None:
            return dist_mm.XW_twoway(x, W, device, rank, model_group)
        else:
            return dist_mm.XW_twoway(x, W, device, rank, model_group).add(b)
        
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
        model_group = ctx.model_group

        grad_input  = dist_mm.XWT_twoway(grad_output, weight, rank=rank, device=device, model_group=model_group)
        grad_weight = dist_mm.XTW_twoway(input, grad_output, rank=rank, device=device, model_group=model_group)
        grad_bias   = grad_output

        if ctx.b is None:
            return grad_input, grad_weight, None, None, None, None
        else:
            return grad_input, grad_weight, grad_bias, None, None, None
         
class DistributedXTW_fourway(torch.nn.Module):
    """
    Batched matrix-matrix (XTW) multiplication distributed across four processes.
    """

    def __init__(self, in_dim, out_dim,  device, rank, n_channels=72, bias=True, init_method='linear'):
        """
        Initialize the Linear layer.
        Parameters
        ----------
        in_dim : int
            input dimension of local weight matrix
        out_dim : int
            output dimension of local weight matrix
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        n_channels : int, optional
            Number of channels. Default, 72.
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        init_method : str, optional
            The method to initialize the weights, either 'linear' or 'conv' (default is 'linear').
        """

        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if init_method == 'linear':
            stdev = 1. / np.sqrt(in_dim*2)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        elif init_method == 'conv':
            stdev = np.sqrt(6.0)/(64*n_channels)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev))
        else:
            self.bias = None
        
        self.device = device
        self.rank = rank
        self.XTW = XTW_fourway().apply
    
    def forward(self, x, model_group):
        """
        Forward pass of linear layer (XTW) distributed across four processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        W : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim).
        b : torch.Tensor
            Local bias tensor of shape (out_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        Returns
        -------
        C : torch.Tensor
            The result of the linear transformation.
        """

        Cij = self.XTW(x, self.weights, self.bias, self.device, self.rank, model_group) 
        
        return Cij
    
class XTW_fourway(torch.autograd.Function):
    """
    Autograd function for batched matrix-matrix (XW) multiplication distributed across four processes.
    """
    
    @staticmethod
    def forward(ctx, x, W, b,  device, rank, model_group):
        """
        Forward pass of linear layer (XTW) distributed across four processes.

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
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        Returns
        -------
        torch.Tensor
            Result of the XTW operation with bias added.
        """
        ctx.save_for_backward(x, W)
        ctx.device = device
        ctx.rank = rank
        ctx.model_group = model_group

        xw = dist_mm.XTW_fourway(x, W, rank, device, model_group)
        return xw.add(b)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of linear layer (XTW) distributed across four processes.

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
        device = ctx.device        
        rank = ctx.rank
        model_group = ctx.model_group

        grad_input = dist_mm.XWT_fourway(weight, grad_output, rank=rank, device=device, model_group=model_group)
        grad_weight = dist_mm.XW_fourway(input, grad_output, rank=rank, device=device, model_group=model_group)
        grad_b = grad_output
        
        return grad_input, grad_weight, grad_b, None, None, None
    
class DistributedXTW_twoway(torch.nn.Module):
    """
    Batched matrix-matrix (XTW) multiplication distributed across two processes.
    """
    def __init__(self, in_dim, out_dim, device, rank, n_channels=72, bias=True, init_method='linear'):
        """
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
            stdev = 1. / np.sqrt(in_dim*2)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        elif init_method == 'conv':
            stdev = np.sqrt(6.0)/(60*n_channels)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev))
        else:
            self.bias = None
        
        self.device = device
        self.rank = rank
        
        self.XTW_twoway = XTW_twoway().apply
    
    def forward(self, x, model_group):
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
        Cij = self.XTW_twoway(x, self.weights, self.bias, self.device, self.rank, model_group)
        
        return Cij
    
class XTW_twoway(torch.autograd.Function):
    """
    Autograd function for batched matrix-matrix (XTW) multiplication distributed across two processes.
    """
    @staticmethod
    def forward(ctx, x, W, b, device, rank, model_group):
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

        ctx.save_for_backward(x, W)
        ctx.device = device
        ctx.rank = rank
        ctx.model_group = model_group
        
        xtw = dist_mm.XTW_twoway(x, W, device, rank, model_group)    

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
        device = ctx.device
        rank = ctx.rank 
        model_group = ctx.model_group 

        grad_input = dist_mm.XWT_twoway(weight, grad_output, device, rank, model_group)
        grad_weight = dist_mm.XW_twoway(input, grad_output, device, rank, model_group)
        grad_b = grad_output
        return grad_input, grad_weight, grad_b, None, None, None
    
class DistributedLayerNorm(torch.nn.Module): 
    """Batched layer norm distributed across four processes."""
    def __init__(self, dim, rank, device):
        """
        Initialize layer normalization.
        Parameters
        ----------
        dim : int
            The dimension of the input features.
        rank : int
            Global rank.
        device : torch.device
            Device to perform computation on.
        """

        super().__init__()
        
        self.layer_norm = torch.nn.LayerNorm(dim).to(device)
        self.layer_norm_apply = DistributedLN.apply
    
    def forward(self, x, rank, device):
        """
        Forward pass of layer norm distributed across four processes.
        """
        x = self.layer_norm_apply(x)
        x = self.layer_norm(x)

        return x
    
class DistributedLN(torch.autograd.Function):
    """
    Autograd function for layer norm function distributed across four processes.
    """
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
        # store context
        ctx.dim = dim
        ctx.group = group

        # local exponentials
<<<<<<< HEAD
        y = torch.exp(x)#-x.amax(dim=dim, keepdim=True)) # softmax fails with norm if only 1 variable
        local_sum = y.sum(dim=dim, keepdim=True)
        # global sum
        global_sum = local_sum.clone()
        
=======
        y = torch.exp(x - x.amax(dim=dim, keepdim=True))
        local_sum = y.sum(dim=dim, keepdim=True)
        # global sum
        global_sum = local_sum.clone()
>>>>>>> 6a6d4c2 (prelimiary attention implmenetation)
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

    
class DistributedXWT_fourway(torch.nn.Module):
    """
    Batched matrix-matrix (XWT) multiplication distributed across four processes.
    """
    def __init__(self, in_dim, out_dim, device, rank, n_channels=1, bias=True, stdev=None, init_method='linear'):
        """
        Initialize the XWT linear layer
        sharded over four processes.
        Parameters
        ----------
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        n_channels : int, optional
            The number of channels (default is 1).
        bias : bool, optional
            Whether to include a bias term (default is True).
        stdev : float, optional
            The standard deviation for weight initialization (default is None).
        init_method : str, optional
            The method for weight initialization, either 'linear' or 'conv' (default is 'linear').
        """

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
        
        self.device = device
        self.rank = rank
        self.XWT = XWT_fourway().apply
    
    def forward(self, x, model_group):
        """
        Forward pass of linear layer (XWT) distributed across four processes.
        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        Returns
        -------
        torch.Tensor
            The result of the linear transformation.
        """
        
        Cij = self.XWT(x, self.weights, self.bias, self.device, self.rank, model_group)
        
        return Cij
    
class DistributedXWT_attention_fourway(torch.nn.Module):
    """
    Batched matrix-matrix (XWT) multiplication distributed across two processes.
    """
    def __init__(self, device, rank, model_group=None):
        """
        Initialize the XWT Linear layer sharded over two processes.
        """
        self.device = device
        self.rank = rank
        self.model_group = model_group # TODO: no bias term

        super().__init__()
        self.XWT = XWT_fourway().apply
    
    def forward(self, x, w):
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

        Cij = self.XWT(x, w, None, self.device, self.rank, self.model_group)
        
        return Cij

class XWT_fourway(torch.autograd.Function):
    """
    Autograd function for batched matrix-matrix (XWT) multiplication distributed across four processes.
    """
    @staticmethod
    def forward(ctx, x, W, b, device, rank, model_group):
        """
        Forward pass of linear layer (XWT) distributed across four processes.

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
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        Returns
        -------
        torch.Tensor
            The result of the linear transformation with bias added.
        """

        ctx.save_for_backward(x, W)        
        ctx.device = device
        ctx.rank = rank
        ctx.model_group = model_group
        xwt = dist_mm.XWT_fourway(x, W, rank, device, model_group)
        ctx.b = b
        if b is None:
            return xwt
        else:
            return xwt.add(b)
        
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of linear layer (XWT) distributed across four processes.

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
        
        device=ctx.device        
        rank = ctx.rank
        model_group = ctx.model_group
        grad_input = dist_mm.XW_fourway(grad_output, weight, rank=rank,  device=device, model_group=model_group)
        grad_weight = dist_mm.XTW_fourway(grad_output, input, rank=rank, device=device, model_group=model_group)
        grad_b = grad_output

        if ctx.b is None:
            return grad_input, grad_weight, None, None, None, None
        else:
            return grad_input, grad_weight, grad_b, None, None, None
        

class DistributedXWT_twoway(torch.nn.Module):
    """
    Batched matrix-matrix (XWT) multiplication distributed across two processes.
    """
    def __init__(self, in_dim, out_dim,  device, rank, n_channels=1, bias=True, stdev=None, init_method='linear'):
        """
        Initialize the XWT Linear layer sharded over two processes.

        Parameters
        ----------
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
            stdev = 1. / np.sqrt(in_dim*2)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev))
        elif init_method == 'conv':
            stdev = np.sqrt(6.0)/(8*8*n_channels)
            self.weights = torch.nn.parameter.Parameter(torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim//2).uniform_(-stdev, stdev))
        else:
            self.bias = None
        
        self.device = device
        self.rank = rank
        self.XWT = XWT_twoway().apply
    
    def forward(self, x, model_group):
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

        Cij = self.XWT(x, self.weights, self.bias, self.device, self.rank, model_group)
        
        return Cij
    

class XWT_twoway(torch.autograd.Function):
    """
    Autograd function for batched matrix-matrix (XWT) multiplication distributed across two processes.
    """
    @staticmethod
    def forward(ctx, x, W, b,  device, rank, model_group):
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
        ctx.device = device
        ctx.rank = rank
        
        xw = dist_mm.XWT_twoway(x, W, device, rank, model_group)
        ctx.b = b

        if b is None:
            return xw
        else:
            return xw.add(b)
        
    
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
        device = ctx.device
        rank = ctx.rank
        b = ctx.b
        
        grad_input  = dist_mm.XW_twoway(grad_output, weight, device=device, rank=rank)
        grad_weight = dist_mm.XTW_twoway(grad_output, input, device=device, rank=rank)
        grad_b = grad_output

<<<<<<< HEAD
        if b is None:
            return grad_input, grad_weight, None, None, None, None
        else:
            return grad_input, grad_weight, grad_b, None, None, None
=======
        return grad_input, grad_weight, grad_b, None, None, None
>>>>>>> 6a6d4c2 (prelimiary attention implmenetation)
    
class DistributedXWT_attention_twoway(torch.nn.Module):
    """
    Batched matrix-matrix (XWT) multiplication distributed across two processes.
    """
<<<<<<< HEAD
    def __init__(self, device, rank, model_group=None):
=======
    def __init__(self, device, rank, model_group):
>>>>>>> 6a6d4c2 (prelimiary attention implmenetation)
        """
        Initialize the XWT Linear layer sharded over two processes.
        """
        self.device = device
        self.rank = rank
<<<<<<< HEAD
        self.model_group = model_group # TODO: no bias term
=======
        self.model_group = model_group
>>>>>>> 6a6d4c2 (prelimiary attention implmenetation)

        super().__init__()
        self.XWT = XWT_twoway().apply
    
<<<<<<< HEAD
    def forward(self, x, w):
=======
    def forward(self, x, w, b, model_group):
>>>>>>> 6a6d4c2 (prelimiary attention implmenetation)
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

<<<<<<< HEAD
        Cij = self.XWT(x, w, None, self.device, self.rank, self.model_group)
        
        return Cij
    
class DistributedXW_attention_twoway(torch.nn.Module):
    """
    Batched matrix-matrix (XWT) multiplication distributed across two processes.
    """
    def __init__(self, device, rank, model_group=None):
        """
        Initialize the XWT Linear layer sharded over two processes.
        """
        self.device = device
        self.rank = rank
        self.model_group = model_group # TODO: no bias term

        super().__init__()
        self.XW = XW_twoway().apply
    
    def forward(self, x, w):
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

        Cij = self.XW(x, w, None, self.device, self.rank, self.model_group)
=======
        Cij = self.XWT(x, w, b, self.device, self.rank, model_group)
>>>>>>> 6a6d4c2 (prelimiary attention implmenetation)
        
        return Cij
    
class DistributedTranspose(torch.nn.Module):
    """
    Batched transpose (XT) operation distributed across four processes.
    """
    def __init__(self):
        """
        Initializes the class for Distributed transpose sharded across four processes.
        Attributes
        ----------
        transpose : function
            A function that applies a distributed transpose transformation.
        """

        super().__init__()
        self.transpose = DistributedXT().apply

    def forward(self, X, rank, device):
        """
        Forward pass of transpose (XT) distributed across four processes.

        Parameters
        ----------
        X : torch.Tensor
            Local input tensor to be transposed.
        rank : int
            Global rank.
        device : torch.device
            Device to perform computation on.
        Returns
        -------
        X : torch.Tensor
            The transposed tensor.
        """

        X = self.transpose(X, rank, device)
        return X

class DistributedXT(torch.autograd.Function):
    """
    Autograd function for batched transpose (XT) operation distributed across four processes.
    """
    @staticmethod
    def forward(ctx, x, rank, device):
        """
        Forward pass of transpose (XT) distributed across four processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        x : torch.Tensor
            Input tensor.
        rank : int
            Global rank.
        device : torch.device
            Device to perform computation on.
        Returns
        -------
        torch.Tensor
            Transposed input tensor.
        """

        ctx.device = device
        ctx.rank = rank
        x = dist_mm.matrix_transpose(x, rank, device)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Forward pass of transpose (XT) distributed across four processes.
        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : tensor
            Gradient of the loss with respect to the output of the layer.
        Returns
        -------
        grad_output: tensor
            Gradient of the loss with respect to the input of the layer.
        """

        rank = ctx.rank
        device = ctx.device
        grad_output = dist_mm.matrix_transpose(grad_output, rank, device)
        return grad_output, None, None

class DistributedTranspose_twoway(torch.nn.Module):
    """
    Batched transpose (XT) operation distributed across two processes.
    """
    def __init__(self):
        """
        Initialize the Linear class.
        This constructor initializes the Linear class by calling the
        superclass constructor and setting up the transpose attribute
        with the DistributedXT_twoway apply method.
        """

        super().__init__()
        self.transpose = DistributedXT_twoway().apply

    def forward(self, X, rank, device, model_group):
        """
        Forward pass of transpose (XT) distributed across two processes.
        Parameters
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

        XT = self.transpose(X, rank, device, model_group)
        return XT

class DistributedXT_twoway(torch.autograd.Function):
    """
    Autograd function for batched transpose (XT) operation distributed across two processes.
    """
    @staticmethod
    def forward(ctx, x, rank, device, model_group):
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
        ctx.model_group = model_group

        x = dist_mm.matrix_transpose_twoway(x, rank, device, model_group)
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
        model_group = ctx.model_group

        grad_output = dist_mm.matrix_transpose_twoway(grad_output, rank, device, model_group)
        return grad_output, None, None, None

class DistributedMLP(torch.nn.Module):
    """
    Transposed MLP layer distributed across either two or four processes.

    Data and weights are formulated following PyTorch convention: forward pass=XWT
    """
    def __init__(self, in_dim, hidden_dim, out_dim,  device, rank, parallelism=4, bias=True, dropout=1e-3):
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
        if parallelism == 4:
            self.linear1 = DistributedXWT_fourway(in_dim, hidden_dim, device, rank, bias=bias, init_method='linear')
            self.linear2 = DistributedXWT_fourway(hidden_dim, out_dim, device, rank, bias=bias, init_method='linear')
        else:
            # in, hidden, out dim are global sizes for two-way model parallelism
            # required to maintain partitioning across final dimension
            self.linear1 = DistributedXWT_twoway(in_dim//2, hidden_dim, device, rank, bias=bias, init_method='linear')
            self.linear2 = DistributedXWT_twoway(hidden_dim//2, out_dim, device, rank, bias=bias, init_method='linear')
        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)
        self.rank = rank
        
    def forward(self, x, model_group):
        """
        Forward pass of MLP distributed across two or four processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        """
        x = self.linear1(x, model_group)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x, model_group)
        x = self.drop2(x)
        
        return x

class DistributedXTWMLP(torch.nn.Module):
    """
    Transposed MLP layer distributed across either two or four processes.

    Data and weights are formulated as: forward pass=XTW
    """
    def __init__(self, in_dim, hidden_dim, out_dim, device, rank, parallelism=4, bias=True, dropout=1e-3):
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
        if parallelism == 4:
            self.linear1 = DistributedXTW_fourway(in_dim, hidden_dim, device, rank, bias=bias, init_method='linear')
            self.linear2 = DistributedXW_fourway(hidden_dim, out_dim, device, rank, bias=bias)
        elif parallelism == 2:
            self.linear1 = DistributedXTW_twoway(in_dim, hidden_dim//2, device, rank, bias=bias, init_method='linear')
            self.linear2 = DistributedXW_twoway(hidden_dim, out_dim//2, device, rank, bias=bias, init_method='linear')
        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)
        self.rank = rank

    def forward(self, x, model_group):
        """
        Forward pass of transposed MLP distributed across two or four processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        """
        x = self.linear1(x, model_group)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x, model_group)
        x = self.drop2(x)
        return x


class MixerBlock(torch.nn.Module):
    """
    Mixing block that performs token and channel mixing using four-way distributed MLPs and layer normalization.
    """
    
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, spatial_hidden_dim_fraction, channels_hidden_dim_fraction, device, rank, parallelism):
        """
        Initialize MixerBlock.

        Parameters
        ----------
        tokens_mlp_dim : int
            Local dimension of the token MLP.
        channels_mlp_dim : int
            Local dimension of the channel MLP.
        spatial_hidden_dim_fraction : float
            Fraction to determine the hidden dimension for token mixing.
        channels_hidden_dim_fraction : float
            Fraction to determine the hidden dimension for channel mixing.
        device : torch.device
            Device to run the computations on.
        rank : int
            Global rank.
        parallelism : int
            Degree of model parallelism.
        """
        super(MixerBlock, self).__init__()
        self.token_mixing = DistributedXTWMLP(tokens_mlp_dim, int(tokens_mlp_dim * spatial_hidden_dim_fraction), tokens_mlp_dim, device, rank, parallelism=parallelism) # TODO: for now, Linear
        self.channel_mixing = DistributedMLP(channels_mlp_dim, int(channels_mlp_dim * channels_hidden_dim_fraction), channels_mlp_dim, device, rank, parallelism=parallelism)
        self.transpose = DistributedTranspose()
        self.rank = rank
        self.device = device
        
        self.layer_norm1 = DistributedLayerNorm(channels_mlp_dim, self.rank, self.device)
        self.layer_norm2 = DistributedLayerNorm(channels_mlp_dim, self.rank, self.device)
 
    def forward(self, x, model_group):
        """
        Forward pass of a MixerBlock distributed across four processes.

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
        
        x_res = x.clone() 
        y = self.layer_norm1(x, self.rank, self.device)                                                                                                                                                                 
        
        # y shape before: [n_batch, n_patches, hidden_dim]
        y = self.token_mixing(y, model_group)
        # y shape after: [n_batches, hidden dim, n_patches]
        y = self.transpose(y, self.rank, self.device) 
        # Residual connection
        x = x_res + y
        y = self.layer_norm2(x, self.rank, self.device) 
        x = x + self.channel_mixing(y, model_group) 
        
        return x
    

class MixerBlock_twoway(torch.nn.Module):
    """
    Mixing block that performs token and channel mixing using two-way distributed MLPs and layer normalization.
    """
    
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, spatial_hidden_dim_fraction, channels_hidden_dim_fraction, device, rank):
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
        """
        super(MixerBlock_twoway, self).__init__()
        self.token_mixing = DistributedXTWMLP(tokens_mlp_dim, int(tokens_mlp_dim * spatial_hidden_dim_fraction), tokens_mlp_dim, device, rank, parallelism=2)
        self.channel_mixing = DistributedMLP(channels_mlp_dim, int(channels_mlp_dim * channels_hidden_dim_fraction), channels_mlp_dim, device, rank, parallelism=2)
        self.transpose = DistributedTranspose_twoway()
        self.rank = rank
        self.device = device
        self.layer_norm1 = torch.nn.LayerNorm(channels_mlp_dim//2) 
        self.layer_norm2 = torch.nn.LayerNorm(channels_mlp_dim//2) 
 
    def forward(self, x, model_group):
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
        y = self.token_mixing(y, model_group)
        y = self.transpose(y, self.rank, self.device, model_group)
        x = x_res + y
        y = self.layer_norm2(x) 
        x = x + self.channel_mixing(y, model_group) 
        
        return x
    

class SequentialMLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) with one hidden layer.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=1e-3):
        """
        Parameters
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
    Parameters
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







