import math
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.init as init

import Jigsaw.distributed_matmul as dist_mm


class FirstLinearLayer(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(self, in_dim: int, hidden_dim: int, n_parallel: int, bias: bool, device, model_group: dist.ProcessGroup) -> None:
        super().__init__()

        self.model_group = model_group
        self.linear = torch.nn.Linear(in_dim, hidden_dim//n_parallel, bias=bias)
        init.uniform_(self.linear.weight, a=-1./math.sqrt(in_dim), b=1./math.sqrt(in_dim))
        if self.linear.bias is not None:
            init.uniform_(self.linear.bias, a=-1./math.sqrt(in_dim), b=1./math.sqrt(in_dim))
        self.all_reduce = MegatronFirstLinearLayer().apply

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
        Cij = self.all_reduce(x, self.linear.weight, self.linear.bias, self.model_group)
        return Cij


class MegatronFirstLinearLayer(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        ctx: torch.autograd.Function, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, model_group: dist.ProcessGroup
    ) -> torch.Tensor:
        """
        Forward pass of all reduce.
        """
        ctx.save_for_backward(x, weight)
        ctx.model_group = model_group
        ctx.has_bias = True if (bias is not None) else False
        x = x @ (weight.mT) 
        
        if bias is not None:
            x = x + bias
        return x

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None]:
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
        x, weight = ctx.saved_tensors
        grad_input = grad_output @ weight
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.model_group)
        grad_weight = grad_output.mT @ x
        
        grad_b = grad_output.sum(dim=0) if ctx.has_bias else None

        return grad_input, grad_weight, grad_b, None

class SecondLinearLayer(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        n_parallel: int,
        bias: bool,
        device: torch.device,
        model_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()

        self.model_group = model_group
        self.linear = torch.nn.Linear(hidden_dim//n_parallel, out_dim, bias=bias)
        init.uniform_(self.linear.weight, a=-1./math.sqrt(hidden_dim), b=1./math.sqrt(hidden_dim))
        if self.linear.bias is not None:
            init.uniform_(self.linear.bias, a=-1./math.sqrt(hidden_dim), b=1./math.sqrt(hidden_dim))
            
        self.all_reduce = MegatronSecondLinearLayer().apply

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
        
        Cij = self.all_reduce(x, self.linear.weight, self.linear.bias, self.model_group)
        return Cij


class MegatronSecondLinearLayer(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        ctx: torch.autograd.Function, x: torch.Tensor, weight, bias, model_group: dist.ProcessGroup
    ) -> torch.Tensor:
        """
        Forward pass of all reduce
        """
        ctx.save_for_backward(x, weight)
        ctx.model_group = model_group
        ctx.has_bias = True if (bias is not None) else False
        x = x @ (weight.mT)
        x = x.contiguous()
        torch.distributed.all_reduce(x, op=dist.ReduceOp.SUM, group=ctx.model_group)
        if bias is not None:
            x = x + bias
        return x

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None]:
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
        x, weight = ctx.saved_tensors
        grad_input = grad_output @ weight
        grad_weight = grad_output.mT @ x
        
        grad_b = grad_output.sum(dim=0) if ctx.has_bias else None
                
        return grad_input, grad_weight, grad_b, None
