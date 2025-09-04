class DistributedXWT_fourway(torch.nn.Module):
    """
    Batched matrix-matrix (XWT) multiplication distributed across four processes.
    """
    def __init__(self, in_dim, out_dim, rank, device, n_channels=1, bias=True, stdev=None, init_method='linear'):
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
        
        Cij = self.XWT(x, self.weights, self.bias, self.rank, self.device, model_group)
        
        return Cij
    
class XWT_fourway(torch.autograd.Function):
    """
    Autograd function for batched matrix-matrix (XWT) multiplication distributed across four processes.
    """
    @staticmethod
    def forward(ctx, x, W, b, rank, device, model_group):
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
        xwt = dist_mm.xw_transpose_fourway(x, W, rank, device, model_group)
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
        grad_input = dist_mm.xw_fourway(grad_output, weight, rank=rank,  device=device, model_group=model_group)
        grad_weight = dist_mm.x_transpose_w_fourway(grad_output, input, rank=rank, device=device, model_group=model_group)
        grad_b = grad_output

        if ctx.b is None:
            return grad_input, grad_weight, None, None, None, None
        else:
            return grad_input, grad_weight, grad_b, None, None, None
        
        