from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

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

class DistributedXWT_Columnwise(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
        n_channels: int = 1,
        bias: bool = True,
        stdev: float = None,
        init_method: str = 'linear'
    ) -> None:
        """
        Initialize the XWT Linear layer sharded over n processes.

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
            # TODO: Dependent on the data sharding scheme
            stdev = 1. / np.sqrt(in_dim*2)
            self.weights = torch.nn.parameter.Parameter(
                torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev))
        elif init_method == 'conv':
            stdev = np.sqrt(6.0)/(8*8*n_channels)  # TODO
            self.weights = torch.nn.parameter.Parameter(
                torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev))
        if bias and n_parallel == 2:
            self.bias = torch.nn.parameter.Parameter(
                torch.zeros(out_dim//2).uniform_(-stdev, stdev))
        elif bias and n_parallel == 4:
            self.bias = torch.nn.parameter.Parameter(
                torch.zeros(out_dim).uniform_(-stdev, stdev))
        elif bias and n_parallel == 8:
            self.bias = torch.nn.parameter.Parameter(
                torch.zeros(out_dim*2).uniform_(-stdev, stdev))
        else:
            self.bias = None

        self.device = device
        self.rank = rank
        self.n_parallel = n_parallel
        self.model_group = model_group

        self.XWT_column = XWT_column().apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
            B, out_dim are global values.
            in_dim is local value.
        """
        Cij = self.XWT_column(x, self.weights, self.bias, self.rank,
                       self.device, self.n_parallel, self.model_group)
        return Cij


class XWT_column(torch.autograd.Function):
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
        model_group: dist.ProcessGroup
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
        model_group : dist.ProcessGroup, optional
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

        xwt = select_xw_transpose_kernel(n_parallel)(
            x, w, rank, device, model_group)

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

        grad_input = select_xw_kernel(n_parallel)(
            grad_output, weight, rank=rank, device=device, model_group=model_group)
        grad_weight = select_x_transpose_w_kernel(n_parallel)(
            grad_output, input, rank=rank, device=device, model_group=model_group)
        grad_b = grad_output

        if b is None:
            return grad_input, grad_weight, None, None, None, None, None
        else:
            return grad_input, grad_weight, grad_b, None, None, None, None