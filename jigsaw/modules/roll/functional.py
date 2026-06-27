"""Autograd definition for roll functions."""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from . import distributed_roll as dist_roll
from .config import RollConfig


class RollCtx(Protocol):
    """Protocol for types stored within ctx."""

    saved_tensors: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    local_rank: int
    shifts: tuple[int, ...]
    dims: tuple[int, ...]
    distributed_dims: tuple[int, ...]
    spatial_groups: tuple[dist.ProcessGroup, ...]
    empty: torch.Tensor
    config: RollConfig


class RollFunction(torch.autograd.Function):
    """Autograd wrapper for distributed roll over an arbitrary process mesh."""

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
        """Forward pass of distributed roll."""
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
        """Context for roll function."""
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
        """Backward pass of distributed roll."""
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
        """Vmap for dealing with roll operations."""
        out = RollFunction.apply(
            x, shifts, dims, distributed_dims, spatial_groups, empty, config
        )
        return out, 0
