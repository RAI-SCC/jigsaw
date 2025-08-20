import torch

class DistributedConv1D(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        random_seed
    ):
        """
        Distributed 1D Convolution.

        random seed is required because of initialization/synchronization between "partner processes"

        input shape of forward pass: (N, Cin, Lin) or (Cin, Lin)
        output shape of forward pass: (N, Cout, Lout)
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.random_seed = random_seed

        self.conv1d = torch.conv1d(self.in_channels, self.out_channels, self.kernel_size)
        
        self.conv1d_apply = Conv1d().apply
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        Cij = self.conv1d_apply(x, self.weights, self.bias, self.rank, self.device, self.n_parallel, self.model_group)
        
        return Cij
    
class Conv1d(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
            ctx: torch.autograd.Function, 
            x: torch.Tensor, 
            w: torch.Tensor, 
            b: torch.Tensor,
            rank: int, 
            device: torch.device, 
            n_parallel: int, 
            model_group: torch.distributed.ProcessGroup
        ) -> torch.Tensor:
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
        ctx.save_for_backward(x, w)
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.model_group = model_group
        ctx.b = b

        xwt = select_xw_transpose_kernel(n_parallel)(x, w, rank, device, model_group)
        
        xwt = xwt if b is None else xwt.add(b)
        return xwt
    
    @staticmethod
    def backward(ctx: torch.autograd.Function, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None]:
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
        
