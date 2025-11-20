from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

import distributed_matmul as dist_mm


class ForwardAllReduce(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        model_group: dist.ProcessGroup
    ) -> None:
        super().__init__()
        
        self.model_group = model_group
        self.all_reduce = MegatronForwardAllReduce().apply

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
        Cij = self.all_reduce(x, self.model_group)
        return Cij


class MegatronForwardAllReduce(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        x: torch.Tensor,
        model_group: dist.ProcessGroup
    ) -> torch.Tensor:
        """
        Forward pass of all reduce
        """
        ctx.save_for_backward(x)
        ctx.model_group = model_group
        x = x.contiguous()
        torch.distributed.all_reduce(x, op=dist.ReduceOp.SUM, group=ctx.model_group)
        x /= 4
        return x

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
        grad_input = grad_output
        
        return grad_input, None

class BackwardAllReduce(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        model_group: dist.ProcessGroup
    ) -> None:
        super().__init__()
        
        self.model_group = model_group
        self.all_reduce = MegatronBackwardAllReduce().apply

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
        Cij = self.all_reduce(x, self.model_group)
        return Cij


class MegatronBackwardAllReduce(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        x: torch.Tensor,
        model_group: dist.ProcessGroup
    ) -> torch.Tensor:
        """
        Forward pass of all reduce
        """
        ctx.model_group = model_group
        
        return x

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
        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.model_group)
        grad_output /= 4
        
        return grad_output, None
