"""Autograd function of the distributed roll."""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from . import distributed_roll as dist_roll
from .config import RollConfig


class RollCtx(Protocol):
    """Attributes stored on the autograd context."""

    saved_tensors: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    local_rank: int
    shifts: tuple[int, ...]
    dims: tuple[int, ...]
    distributed_dims: tuple[int, ...]
    spatial_groups: tuple[dist.ProcessGroup, ...]
    empty: torch.Tensor
    config: RollConfig


class RollFunction(torch.autograd.Function):
    """Autograd function of the distributed roll."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        shifts: tuple[int, ...],
        dims: tuple[int, ...],
        distributed_dims: tuple[int, ...],
        spatial_groups: tuple[dist.ProcessGroup, ...],
        empty: torch.Tensor,
        config: RollConfig,
    ) -> torch.Tensor:
        """Roll the local tensor and exchange the halos.

        Parameters
        ----------
        x : torch.Tensor
            Local tensor.
        shifts : tuple[int, ...]
            Shift along each dimension of ``dims``.
        dims : tuple[int, ...]
            Dimensions to roll.
        distributed_dims : tuple[int, ...]
            Dimensions across which ``x`` is partitioned.
        spatial_groups : tuple[dist.ProcessGroup, ...]
            Process group of each entry of ``distributed_dims``.
        empty : torch.Tensor
            Placeholder tensor for ranks that exchange no data.
        config : RollConfig
            Kernel selection.

        Returns
        -------
        torch.Tensor
            Rolled local tensor.
        """
        return dist_roll.distributed_roll(
            x,
            shifts,
            dims,
            list(distributed_dims),
            list(spatial_groups),
            empty,
            config.comm_halo_1d_fn,
        )

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[
            torch.Tensor,
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[dist.ProcessGroup, ...],
            torch.Tensor,
            RollConfig,
        ],
        output: torch.Tensor,
    ) -> None:
        """Save the tensors and metadata for the backward pass."""
        _, shifts, dims, distributed_dims, spatial_groups, empty, config = inputs
        c = cast(RollCtx, ctx)
        c.shifts = shifts
        c.dims = dims
        c.distributed_dims = distributed_dims
        c.spatial_groups = spatial_groups
        c.empty = empty
        c.config = config

    @staticmethod
    def backward(
        ctx: RollCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        """Compute the gradient of the distributed roll.

        Parameters
        ----------
        ctx : RollCtx
            Autograd context.
        grad_output : torch.Tensor
            Gradient of the output.

        Returns
        -------
        tuple[torch.Tensor, None, None, None, None, None, None]
            Gradient of the input, followed by ``None`` for the non-differentiable
            arguments.
        """
        neg_shifts = tuple(-s for s in ctx.shifts)
        grad_input = dist_roll.distributed_roll(
            grad_output,
            neg_shifts,
            ctx.dims,
            list(ctx.distributed_dims),
            list(ctx.spatial_groups),
            ctx.empty,
            ctx.config.comm_halo_1d_fn,
        )
        return grad_input, None, None, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        shifts: tuple[int, ...],
        dims: tuple[int, ...],
        distributed_dims: tuple[int, ...],
        spatial_groups: tuple[dist.ProcessGroup, ...],
        empty: torch.Tensor,
        config: RollConfig,
    ) -> tuple[torch.Tensor, int]:
        """Batching rule for ``torch.vmap``."""
        out = RollFunction.apply(
            x, shifts, dims, distributed_dims, spatial_groups, empty, config
        )
        return out, 0
