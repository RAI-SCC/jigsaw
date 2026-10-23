"""Autograd function of the distributed roll."""

from typing import Any, Protocol, cast

import torch
import torch.distributed as dist

from . import _distributed_roll as dist_roll
from ._config import RollConfig


class RollCtx(Protocol):
    """Attributes stored on the autograd context."""

    saved_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    local_rank: int
    shifts: tuple[int, ...]
    dims: tuple[int, ...]
    distributed_dims: tuple[int, ...]
    spatial_groups: tuple[dist.ProcessGroup, ...]
    config: RollConfig


class _Roll(torch.autograd.Function):
    """Autograd function of the distributed roll."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        shifts: tuple[int, ...],
        dims: tuple[int, ...],
        distributed_dims: tuple[int, ...],
        spatial_groups: tuple[dist.ProcessGroup, ...],
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
            RollConfig,
        ],
        output: torch.Tensor,
    ) -> None:
        """Save the tensors and metadata for the backward pass."""
        _, shifts, dims, distributed_dims, spatial_groups, config = inputs
        c = cast(RollCtx, ctx)
        c.shifts = shifts
        c.dims = dims
        c.distributed_dims = distributed_dims
        c.spatial_groups = spatial_groups
        c.config = config

    @staticmethod
    def backward(
        ctx: RollCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        """Compute the gradient of the distributed roll.

        Parameters
        ----------
        ctx : RollCtx
            Autograd context.
        grad_output : torch.Tensor
            Gradient of the output.

        Returns
        -------
        tuple[torch.Tensor, None, None, None, None, None]
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
            ctx.config.comm_halo_1d_fn,
        )
        return grad_input, None, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        x: torch.Tensor,
        shifts: tuple[int, ...],
        dims: tuple[int, ...],
        distributed_dims: tuple[int, ...],
        spatial_groups: tuple[dist.ProcessGroup, ...],
        config: RollConfig,
    ) -> tuple[torch.Tensor, int]:
        """Batching rule for ``torch.vmap``."""
        bdim = in_dims[0]
        assert bdim is not None  # ``x`` is the only tensor input
        x = x.movedim(bdim, 0)

        # ``dims`` are logical; the batch dimension now sits in front of them.
        def _shift(ds: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(d + 1 if d >= 0 else d for d in ds)

        out = _Roll.apply(
            x,
            shifts,
            _shift(dims),
            _shift(distributed_dims),
            spatial_groups,
            config,
        )
        return out, 0
