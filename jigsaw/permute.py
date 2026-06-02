from typing import Optional

import torch
import torch.distributed as dist

from . import distributed_permute as dist_permute


class DistributedXTranspose(torch.nn.Module):
    """Batched transpose (XT) operation distributed across n processes."""

    def __init__(
        self,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
    ) -> None:
        """
        Initialize the X Transpose class.

        Parameters
        ----------
        local_rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes to distribute over
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.selector = dist_permute.TransposeSelectionStrategy(self.parallelism)

        self.transpose = XT.apply

    def get_parameter_count(self):
        """Return 0 parameter count for transpose operation."""
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transpose (XT) distributed across two processes.

        Parameters.
        ----------

        X : torch.Tensor
            The input tensor.

        Returns
        -------
        XT: torch.Tensor
            Local transposed tensor.
        """
        x_transpose = self.transpose(
            x, self.local_rank, self.device, self.parallelism, self.model_group
        )
        return x_transpose


class XT(torch.autograd.Function):
    """Autograd function for batched transpose (XT) operation distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        local_rank: int,
        device: torch.device,
        selector: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Forward pass of transpose (XT) distributed across n processes.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        local_rank : int
            Local model rank.
        device : torch.device
            The device on which the computation is performed.
        selector: TransposeSelectionStrategy
            Instance of transpose selection strategy class
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        XT: torch.Tensor
            Local transposed tensor.
        """
        x = selector.x_t(x, local_rank, device, model_group)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, local_rank, device, selector, model_group)
        output: xwt
        """
        x, local_rank, device, selector, model_group = inputs
        ctx.device = device
        ctx.local_rank = local_rank
        ctx.selector = selector
        ctx.model_group = model_group

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
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
        local_rank = ctx.local_rank
        device = ctx.device
        selector = ctx.selector
        model_group = ctx.model_group

        grad_output = selector.x_t(grad_output, local_rank, device, model_group)
        return grad_output, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        local_rank: int,
        device: torch.device,
        selector: int,
        model_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Perform vmap for transpose operation."""
        x = x.contiguous()
        # x under vmap has shape (V, B, x_in, in_dim).
        out = XT.apply(x, local_rank, device, selector, model_group)

        # out has vmapped dimension at dim 0
        return out, 0
