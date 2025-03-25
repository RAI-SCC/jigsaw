import torch
import summa_gpu
import numpy as np
from torch.nn import GELU, Dropout

class DistributedXW_fourway(torch.nn.Module):
    """
    Torch Module to run distributed matrix-matrix multiplication.
    """
    def __init__(self, in_dim, out_dim, device, rank, bias=True):
        """
        Initialize distributed matrix-matrix multiplication module.

        in_dim: int
            input dimension of local weight matrix
        out_dim: int
            output dimension of local weight matrix
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
        self.device = device
        self.rank = rank
        self.XW = XW_fourway().apply
    
    def forward(self, x, model_group):
        """
        Forward pass.

        x: Tensor
            local input tensor of shape (B, x_in, x_out)

        Returns
        -------
        C: Tensor
            of shape (B, x_in, out_dim)
        """
        Cij = self.XW(x, self.weights, self.bias, self.device, self.rank, model_group)
        
        return Cij
    
class XW_fourway(torch.autograd.Function):
    """
    Autograd function of forward pass for distributed matrix-matrix multiplication
    """
    @staticmethod
    def forward(ctx, x, W, b, device, rank, model_group):
        """
        Performs the forward pass of the linear layer.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward pass.
        x : Tensor
            Input tensor.
        W : Tensor
            Weight matrix.
        b : Tensor
            Bias vector.
        device : str
            Device to perform computation on.
        rank : int
            Rank of the process.
        model_group : torch.distributed.ProcessGroup
            Process group for processes in a single model-instance.

        Returns
        -------
        Tensor
            Result of the linear transformation with bias added.
        """
        
        ctx.save_for_backward(x, W)
        
        ctx.device = device
        ctx.rank = rank
        ctx.model_group = model_group
        XW = summa_gpu.XW_fourway(x, W, rank,  device, model_group)
        
        return XW.add(b)
        
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Perform the backward pass for the linear layer.
        Parameters
        ----------
        ctx : torch.autograd.Function
            The context object that stores information for the backward computation.
        grad_output : torch.Tensor
            The gradient of the loss with respect to the output of the linear layer.
        Returns
        -------
        tuple of torch.Tensor
            - grad_input : torch.Tensor
                The gradient of the loss with respect to the input of the linear layer.
            - grad_weight : torch.Tensor
                The gradient of the loss with respect to the weights of the linear layer.
            - grad_b : torch.Tensor
                The gradient of the loss with respect to the bias of the linear layer.
        """
        
        input, weight = ctx.saved_tensors
        
        device = ctx.device        
        rank = ctx.rank
        model_group = ctx.model_group

        grad_input  = summa_gpu.XWT_fourway(grad_output, weight, rank=rank, device=device, model_group=model_group)
        grad_weight = summa_gpu.XTW_fourway(input, grad_output, rank=rank, device=device, model_group=model_group)
        grad_b      = grad_output
        return grad_input, grad_weight, grad_b, None, None, None
    
class DistributedXW_twoway(torch.nn.Module):
    """
    Torch Module to run distributed matrix-matrix multiplication.
    """
    def __init__(self, in_dim, out_dim, device, rank, n_channels=1, bias=True, stdev=None, init_method='linear'):
        """
        Initialize distributed matrix-matrix multiplication module.

        in_dim: int
            input dimension of local weight matrix
        out_dim: int
            output dimension of local weight matrix
        
        device: string
            name of device
        rank: int
            GPU rank        
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
        Forward pass.

        x: Tensor
            local input tensor of shape (B, x_in, x_out)

        Returns
        -------
        C: Tensor
            of shape (B, x_in, out_dim)
        """
        Cij = self.XW_twoway(x, self.weights, self.bias, self.device, self.rank, model_group)
        
        return Cij
    
class XW_twoway(torch.autograd.Function):
    """
    Autograd function of forward pass for distributed matrix-matrix multiplication
    """
    @staticmethod
    def forward(ctx, x, W, b, device, rank, model_group):
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
        ctx.model_group = model_group
        
        
        return summa_gpu.XW_twoway(x, W, device, rank, model_group).add(b)
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Perform the backward pass for the linear layer.
        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object containing saved tensors and other information.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output of the layer.
        Returns
        -------
        tuple of torch.Tensor
            Gradients with respect to the input, weight, and bias.
        """

        input, weight = ctx.saved_tensors
        rank = ctx.rank
        device = ctx.device
        model_group = ctx.model_group

        grad_input  = summa_gpu.XWT_twoway(grad_output, weight, rank=rank, device=device, model_group=model_group)
        grad_weight = summa_gpu.XTW_twoway(input, grad_output, rank=rank, device=device, model_group=model_group)
        grad_bias   = grad_output#.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None, None
         
class DistributedXTW_fourway(torch.nn.Module):
    """
    A PyTorch module for distributed four-way tensor multiplication with optional bias.
    Parameters
    ----------
    in_dim : int
        The input dimension size.
    out_dim : int
        The output dimension size.
    device : torch.device
        The device on which to perform computations.
    rank : int
        The rank of the current process in distributed computation.
    n_channels : int, optional
        The number of channels, by default 72.
    bias : bool, optional
        If True, adds a learnable bias to the output, by default True.
    init_method : str, optional
        The method to initialize weights, either 'linear' or 'conv', by default 'linear'.
    Attributes
    ----------
    in_dim : int
        The input dimension size.
    out_dim : int
        The output dimension size.
    weights : torch.nn.Parameter
        The learnable weights of the module.
    bias : torch.nn.Parameter or None
        The learnable bias of the module if `bias` is True, otherwise None.
    device : torch.device
        The device on which to perform computations.
    rank : int
        The rank of the current process in distributed computation.
    XTW : callable
        The function to perform the four-way tensor multiplication.
    Methods
    -------
    forward(x, model_group)
        Performs the forward pass of the module.
    """

    def __init__(self, in_dim, out_dim,  device, rank, n_channels=72, bias=True, init_method='linear'):
        
        """
        Initialize the Linear layer.
        Parameters
        ----------
        in_dim : int
            The number of input dimensions.
        out_dim : int
            The number of output dimensions.
        device : torch.device
            The device on which to allocate the tensors.
        rank : int
            The rank of the tensor.
        n_channels : int, optional
            The number of channels (default is 72).
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        init_method : str, optional
            The method to initialize the weights, either 'linear' or 'conv' (default is 'linear').
        Attributes
        ----------
        in_dim : int
            The number of input dimensions.
        out_dim : int
            The number of output dimensions.
        weights : torch.nn.parameter.Parameter
            The learnable weights of the layer.
        bias : torch.nn.parameter.Parameter or None
            The learnable bias of the layer, if bias is True. Otherwise, None.
        device : torch.device
            The device on which the tensors are allocated.
        rank : int
            The rank of the tensor.
        XTW : callable
            The XTW_fourway function applied to the layer.
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
        Perform the forward pass of the linear model.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        model_group : torch.distributed.ProcessGroup
            Process group for processes in a single model-instance.
        Returns
        -------
        torch.Tensor
            The result of the linear transformation.
        """

        Cij = self.XTW(x, self.weights, self.bias, self.device, self.rank, model_group) 
        
        return Cij
    
class XTW_fourway(torch.autograd.Function):
    """
    Custom autograd function for a four-way XTW operation with precompute and isend.
    Methods
    -------
    forward(ctx, x, W, b, device, rank, model_group)
        Performs the forward pass of the XTW operation.
    backward(ctx, grad_output)
        Computes the gradients for the backward pass.
    Parameters
    ----------
    ctx : torch.autograd.Function
        Context object to save information for backward computation.
    x : torch.Tensor
        Input tensor.
    W : torch.Tensor
        Weight tensor.
    b : torch.Tensor
        Bias tensor.
    device : torch.device
        Device on which to perform the computation.
    rank : int
        Rank of the process in the distributed group.
    model_group : torch.distributed.ProcessGroup
        Process group for processes in a single model-instance.
    Returns
    -------
    torch.Tensor
        Result of the XTW operation with bias added.
    """

    @staticmethod
    def forward(ctx, x, W, b,  device, rank, model_group):
        ctx.save_for_backward(x, W)
        ctx.device = device
        ctx.rank = rank
        ctx.model_group = model_group

        xw = summa_gpu.XTW_fourway(x, W, rank, device, model_group)
        return xw.add(b)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Perform the backward pass for the linear layer.
        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object containing saved tensors and other information.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output of the layer.
        Returns
        -------
        tuple
            Gradients with respect to the input, weight, and bias, respectively.
        """

        input, weight = ctx.saved_tensors
        
        device = ctx.device        
        rank = ctx.rank
        model_group = ctx.model_group

        grad_input = summa_gpu.XWT_fourway(weight, grad_output, rank=rank, device=device, model_group=model_group)
        grad_weight = summa_gpu.XW_fourway(input, grad_output, rank=rank, device=device, model_group=model_group)
        grad_b = grad_output
        
        return grad_input, grad_weight, grad_b, None, None, None
    
class DistributedXTW_twoway(torch.nn.Module):
    def __init__(self, in_dim, out_dim, device, rank, n_channels=72, bias=True, init_method='linear'):
        
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
        
        Cij = self.XTW_twoway(x, self.weights, self.bias, self.device, self.rank, model_group)
        
        return Cij
    
class XTW_twoway(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b, device, rank, model_group):
        """
        Performs the forward pass of the linear layer.
        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward pass.
        x : torch.Tensor
            Input tensor.
        W : torch.Tensor
            Weight matrix.
        b : torch.Tensor
            Bias vector.
        device : torch.device
            Device to perform computation on.
        rank : int
            Rank of the process in distributed setting.
        model_group : torch.distributed.ProcessGroup
            Process group for processes in a single model-instance.
        Returns
        -------
        torch.Tensor
            Output tensor after applying linear transformation and adding bias.
        """

        ctx.save_for_backward(x, W)
        ctx.device = device
        ctx.rank = rank
        ctx.model_group = model_group
        
        xw = summa_gpu.XTW_twoway(x, W, device, rank, model_group)    

        return xw.add(b) 
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Perform the backward pass for the linear layer.
        Parameters
        ----------
        ctx : torch.autograd.Function
            The context object containing saved tensors and other information.
        grad_output : torch.Tensor
            The gradient of the loss with respect to the output of the layer.
        Returns
        -------
        tuple
            A tuple containing the gradients with respect to the input, weight, and bias.
        """
        input, weight = ctx.saved_tensors        
        device = ctx.device
        rank = ctx.rank 
        model_group = ctx.model_group 

        grad_input = summa_gpu.XWT_twoway(weight, grad_output, device, rank, model_group)
        grad_weight = summa_gpu.XW_twoway(input, grad_output, device, rank, model_group)
        grad_b = grad_output
        return grad_input, grad_weight, grad_b, None, None, None
    
class DistributedLayerNorm(torch.nn.Module):
    def __init__(self, dim, rank, device):
        """
        Initialize the Linear layer with layer normalization.
        Parameters
        ----------
        dim : int
            The dimension of the input features.
        rank : int
            The rank of the distributed layer normalization.
        device : torch.device
            The device to which the layer should be moved.
        """

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
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

    
class DistributedXWT_fourway(torch.nn.Module):
    def __init__(self, in_dim, out_dim, device, rank, n_channels=1, bias=True, stdev=None, init_method='linear'):
        """
        Initialize the Linear layer.
        Parameters
        ----------
        in_dim : int
            The number of input dimensions.
        out_dim : int
            The number of output dimensions.
        device : torch.device
            The device on which to allocate the tensors.
        rank : int
            The rank of the layer.
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
        Perform the forward pass of the linear transformation.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        model_group : torch.distributed.ProcessGroup
            Process group for processes in a single model-instance.
        Returns
        -------
        torch.Tensor
            The result of the linear transformation.
        """
        
        Cij = self.XWT(x, self.weights, self.bias, self.device, self.rank, model_group)
        
        return Cij
    

class XWT_fourway(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b, device, rank, model_group):
        """
        Performs the forward pass of the linear layer.
        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        x : torch.Tensor
            Input tensor.
        W : torch.Tensor
            Weight matrix.
        b : torch.Tensor
            Bias vector.
        device : torch.device
            Device on which the computation is performed.
        rank : int
            Rank of the current process in distributed training.
        model_group : torch.distributed.ProcessGroup
            Process group for processes in a single model-instance.
        Returns
        -------
        torch.Tensor
            The result of the linear transformation with bias added.
        """

        ctx.save_for_backward(x, W)        
        ctx.device = device
        ctx.rank = rank
        ctx.model_group = model_group
        xwt = summa_gpu.XWT_fourway(x, W, rank, device, model_group)
        
        return xwt.add(b)
        
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradients of the input, weight, and bias during the backward pass.
        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object containing saved tensors and other information.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output of the layer.
        Returns
        -------
        tuple
            Gradients with respect to the input, weight, and bias, respectively.
        """

        input, weight = ctx.saved_tensors
        
        device=ctx.device        
        rank = ctx.rank
        model_group = ctx.model_group
        grad_input = summa_gpu.XW_fourway(grad_output, weight, rank=rank,  device=device, model_group=model_group)
        grad_weight = summa_gpu.XTW_fourway(grad_output, input, rank=rank, device=device, model_group=model_group)
        grad_b = grad_output

        return grad_input, grad_weight, grad_b, None, None, None

class DistributedXWT_twoway(torch.nn.Module):
    
    def __init__(self, in_dim, out_dim,  device, rank, n_channels=1, bias=True, stdev=None, init_method='linear'):
        """
        Initialize the Linear layer.
        Parameters
        ----------
        in_dim : int
            The number of input dimensions.
        out_dim : int
            The number of output dimensions.
        device : torch.device
            The device on which to allocate the tensors.
        rank : int
            The rank of the process in distributed training.
        n_channels : int, optional
            The number of channels (default is 1).
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
        Perform the forward pass of the linear model.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        model_group : torch.distributed.ProcessGroup
            Process group for processes in a single model-instance.
        Returns
        -------
        torch.Tensor
            Resulting tensor after applying the linear transformation.
        """

        Cij = self.XWT(x, self.weights, self.bias, self.device, self.rank, model_group)
        
        return Cij
    

class XWT_twoway(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b,  device, rank, model_group):
        """
        Performs the forward pass of the linear layer.
        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward pass.
        x : torch.Tensor
            Input tensor.
        W : torch.Tensor
            Weight matrix.
        b : torch.Tensor
            Bias vector.
        device : torch.device
            Device to perform computation on.
        rank : int
            Rank of the process in distributed training.
        model_group : torch.distributed.ProcessGroup
            Process group for processes in a single model-instance.
        Returns
        -------
        torch.Tensor
            Output tensor after applying the linear transformation and adding the bias.
        """

        ctx.save_for_backward(x, W)
        ctx.device = device
        ctx.rank = rank
        
        xw = summa_gpu.XWT_twoway(x, W, device, rank, model_group)
        
        return xw.add(b)
        
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradients of the input, weight, and bias during the backward pass.
        Parameters
        ----------
        ctx : torch.autograd.Function
            The context object containing saved tensors and other information.
        grad_output : tensor
            The gradient of the loss with respect to the output of the layer.
        Returns
        -------
        tuple of tensors
            Gradients with respect to the input, weight, and bias, respectively.
        """

        input, weight = ctx.saved_tensors
        device = ctx.device
        rank = ctx.rank
        
        grad_input  = summa_gpu.XW_twoway(grad_output, weight, device=device, rank=rank)
        grad_weight = summa_gpu.XTW_twoway(grad_output, input, device=device, rank=rank)
        grad_b = grad_output

        return grad_input, grad_weight, grad_b, None, None, None

class DistributedTranspose(torch.nn.Module):
    def __init__(self):
        """
        Initializes the Linear class.
        Attributes
        ----------
        transpose : function
            A function that applies the DistributedXT transformation.
        """

        super().__init__()
        self.transpose = DistributedXT().apply

    def forward(self, X, rank, device):
        """
        Perform the forward pass by transposing the input tensor.
        Parameters
        ----------
        X : torch.Tensor
            The input tensor to be transposed.
        rank : int
            The rank of the tensor.
        device : torch.device
            The device on which the tensor is located.
        Returns
        -------
        torch.Tensor
            The transposed tensor.
        """

        X = self.transpose(X, rank, device)
        return X

class DistributedXT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rank, device):
        """
        Performs the forward pass of the linear layer.
        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to store information for backward computation.
        x : torch.Tensor
            Input tensor.
        rank : int
            Rank of the process in distributed computation.
        device : torch.device
            Device on which the computation is performed.
        Returns
        -------
        torch.Tensor
            Transposed input tensor.
        """

        ctx.device = device
        ctx.rank = rank
        x = summa_gpu.matrix_transpose(x, rank, device)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient of the loss with respect to the input of the layer.
        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object containing information about the forward pass.
        grad_output : tensor
            Gradient of the loss with respect to the output of the layer.
        Returns
        -------
        grad_output: tensor
            Gradient of the loss with respect to the input of the layer.
        """

        rank = ctx.rank
        device = ctx.device
        grad_output = summa_gpu.matrix_transpose(grad_output, rank, device)
        return grad_output, None, None

class DistributedTranspose_twoway(torch.nn.Module):
    """
    A PyTorch module for distributed two-way transpose operation.
    Methods
    -------
    forward(X, rank, device, model_group)
        Applies the distributed transpose operation on the input tensor.
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
        Perform the forward pass by transposing the input tensor.
        Parameters
        ----------
        X : torch.Tensor
            The input tensor.
        rank : int
            The rank of the tensor.
        device : torch.device
            The device on which the tensor is located.
        model_group : torch.distributed.ProcessGroup
            Process group for processes in a single model-instance.
        Returns
        -------
        torch.Tensor
            The transposed tensor.
        """

        X = self.transpose(X, rank, device, model_group)
        return X

class DistributedXT_twoway(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rank, device, model_group):
        """
        Performs the forward pass of the linear layer.
        Parameters
        ----------
        ctx : torch.autograd.Function
            The context object to store information for backward computation.
        x : torch.Tensor
            The input tensor.
        rank : int
            The rank of the current process in the distributed setting.
        device : torch.device
            The device on which the computation is performed.
        model_group : torch.distributed.ProcessGroup
            Process group for processes in a single model-instance.
        Returns
        -------
        torch.Tensor
            The transposed input tensor.
        """

        ctx.device = device
        ctx.rank = rank
        ctx.model_group = model_group

        x = summa_gpu.matrix_transpose_twoway(x, rank, device, model_group)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Perform the backward pass for the linear layer.
        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object containing information from the forward pass.
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

        grad_output = summa_gpu.matrix_transpose_twoway(grad_output, rank, device, model_group)
        return grad_output, None, None, None

class DistributedMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,  device, rank, parallelism=4, bias=True):
        """
        Initialize the Linear module with distributed weight tensors.
        Parameters
        ----------
        in_dim : int
            Input dimension size.
        hidden_dim : int
            Hidden layer dimension size.
        out_dim : int
            Output dimension size.
        device : torch.device
            Device on which to allocate tensors.
        rank : int
            Rank of the current process in distributed training.
        parallelism : int, optional
            Degree of parallelism (default is 4).
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        """

        super().__init__()
        if parallelism == 4:
            self.linear1 = DistributedXWT_fourway(in_dim, hidden_dim, device, rank, bias=bias, init_method='linear')
            self.linear2 = DistributedXWT_fourway(hidden_dim, out_dim, device, rank, bias=bias, init_method='linear')
        else:
            self.linear1 = DistributedXWT_twoway(in_dim//2, hidden_dim, device, rank, bias=bias, init_method='linear')
            self.linear2 = DistributedXWT_twoway(hidden_dim//2, out_dim, device, rank, bias=bias, init_method='linear')
        self.activation = GELU()
        self.drop = Dropout(p=1e-6)
        self.drop2 = Dropout(p=1e-6)
        self.rank = rank
        
    def forward(self, x, model_group):
        x = self.linear1(x, model_group)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x, model_group)
        x = self.drop2(x)
        
        return x

class DistributedXTWMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, device, rank, parallelism=4, bias=True, dropout=1e-6):
        """
        Initialize the Linear module with distributed layers.
        Parameters
        ----------
        in_dim : int
            The input dimension size.
        hidden_dim : int
            The hidden layer dimension size.
        out_dim : int
            The output dimension size.
        device : torch.device
            The device to run the computations on.
        rank : int
            The rank of the current process in distributed training.
        parallelism : int, optiona
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
        x = self.linear1(x, model_group)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x, model_group)
        x = self.drop2(x)
        return x


class MixerBlock(torch.nn.Module):
    """
    A neural network block that performs token and channel mixing using distributed MLPs and layer normalization.
    Parameters
    ----------
    tokens_mlp_dim : int
        Dimension of the token MLP.
    channels_mlp_dim : int
        Dimension of the channel MLP.
    spatial_hidden_dim_fraction : float
        Fraction to determine the hidden dimension for token mixing.
    channels_hidden_dim_fraction : float
        Fraction to determine the hidden dimension for channel mixing.
    device : torch.device
        Device to run the computations on.
    rank : int
        Rank of the current process in distributed training.
    parallelism : int
        Degree of model parallelism
    Methods
    -------
    forward(x, model_group)
        Forward pass through the MixerBlock.
    """
    
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, spatial_hidden_dim_fraction, channels_hidden_dim_fraction, device, rank, parallelism):
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
        Perform the forward pass of the model.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [n_batch, n_patches, hidden_dim].
        model_group : torch.distributed.ProcessGroup
            Process group for processes in a single model-instance.
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
       
        x = x_res + y
        y = self.layer_norm2(x, self.rank, self.device) 
        x = x + self.channel_mixing(y, model_group) 
        
        return x
    

class MixerBlock_twoway(torch.nn.Module):
    
    """
    A two-way Mixer Block for neural networks.
    Parameters
    ----------
    tokens_mlp_dim : int
        Dimension of the tokens MLP.
    channels_mlp_dim : int
        Dimension of the channels MLP.
    spatial_hidden_dim_fraction : float
        Fraction of the hidden dimension for spatial mixing.
    channels_hidden_dim_fraction : float
        Fraction of the hidden dimension for channel mixing.
    device : torch.device
        Device to run the computations on.
    rank : int
        Rank for distributed computations.
    Methods
    -------
    forward(x, model_group)
        Forward pass through the Mixer Block.
    """
    
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, spatial_hidden_dim_fraction, channels_hidden_dim_fraction, device, rank):
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
        Perform the forward pass of the model.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        model_group : torch.distributed.ProcessGroup
            Process group for processes in a single model-instance.
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

        y = self.layer_norm2(x) # would need to redefine layer norm to be dim = [1, channels_mlp]
        x = x + self.channel_mixing(y, model_group) # do XTW
        
        return x
    

class SequentialMLP(torch.nn.Module):
    """
    A simple sequential Multi-Layer Perceptron (MLP) with one hidden layer.
    Parameters
    ----------
    in_dim : int
        Dimension of the input features.
    hidden_dim : int
        Dimension of the hidden layer.
    out_dim : int
        Dimension of the output layer.
    dropout : float, optional
        Dropout probability, by default 1e-1.
    Methods
    -------
    forward(x)
        Forward pass through the network.
    """

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
        Dropout rate, by default 1e-6.
    Methods
    -------
    forward(x)
        Forward pass through the Mixer block.
    """
    
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, spatial_hidden_dim_fraction, channels_hidden_dim_fraction, dropout=1e-6):
        super(MixerBlockSequential, self).__init__()
        
        self.token_mixing = SequentialMLP(tokens_mlp_dim, int(tokens_mlp_dim * spatial_hidden_dim_fraction), tokens_mlp_dim, dropout=dropout)
        self.channel_mixing = SequentialMLP(channels_mlp_dim, int(channels_mlp_dim * channels_hidden_dim_fraction), channels_mlp_dim, dropout=dropout)
        self.layer_norm1 = torch.nn.LayerNorm(channels_mlp_dim)
        self.layer_norm2 = torch.nn.LayerNorm(channels_mlp_dim)
        
    def forward(self, x):
        """
        Perform the forward pass of the model.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        Returns
        -------
        torch.Tensor
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







