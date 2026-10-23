"""Shared primitives of the distributed loss functions.

The class dimension of a classification input is sharded across a
``channel_group`` and the remaining dimensions across ``reduce_groups``.
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

Groups = tuple[dist.ProcessGroup | None, ...]
GroupsArg = Sequence[dist.ProcessGroup | None] | dist.ProcessGroup | None


def _active(group: dist.ProcessGroup | None) -> bool:
    """Return whether ``group`` spans more than one rank."""
    return group is not None and group.size() > 1


def _normalize_groups(groups: GroupsArg) -> Groups:
    """Coerce a group argument into a tuple of groups."""
    if groups is None:
        return ()
    if isinstance(groups, dist.ProcessGroup):
        return (groups,)
    return tuple(groups)


def _group_size(group: dist.ProcessGroup | None) -> int:
    """Return the size of ``group``, or ``1`` if it is ``None``."""
    return 1 if group is None else group.size()


def _group_rank(group: dist.ProcessGroup | None) -> int:
    """Return the rank in ``group``, or ``0`` if it is ``None``."""
    return 0 if group is None else group.rank()


class _RedundantAllReduce(torch.autograd.Function):
    """Sum all-reduce whose gradient is the identity."""

    @staticmethod
    def forward(x: torch.Tensor, groups: Groups) -> torch.Tensor:
        """Sum ``x`` over every active group in ``groups``."""
        y = x.clone()
        for group in groups:
            if _active(group):
                dist.all_reduce(y, op=dist.ReduceOp.SUM, group=group)
        return y

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[torch.Tensor, Groups],
        output: torch.Tensor,
    ) -> None:
        """Save nothing for the backward pass."""
        return None

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """Return ``grad_output`` unchanged."""
        return grad_output, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        x: torch.Tensor,
        groups: Groups,
    ) -> tuple[torch.Tensor, int | None]:
        """Batching rule for ``torch.vmap``."""
        return _RedundantAllReduce.apply(x, groups), in_dims[0]


def redundant_all_reduce(x: torch.Tensor, groups: GroupsArg) -> torch.Tensor:
    """Sum ``x`` over ``groups`` with an identity gradient.

    Parameters
    ----------
    x : torch.Tensor
        Local contribution.
    groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None] | None
        Groups over which to sum.

    Returns
    -------
    torch.Tensor
        Sum over the ranks of every active group.
    """
    return _RedundantAllReduce.apply(x, _normalize_groups(groups))  # type: ignore[no-any-return]


def _mean_over_groups(
    local_sum: torch.Tensor,
    local_denom: torch.Tensor,
    reduce_groups: Groups,
) -> torch.Tensor:
    """Sum the numerator and denominator over ``reduce_groups`` and divide.

    Reduced and divided in at least float32, then cast back to the dtype of
    ``local_sum``: in bfloat16 the element count would round (1028 to 1024)
    and the cross-shard sum would lose precision.
    """
    acc = torch.promote_types(
        torch.promote_types(local_sum.dtype, local_denom.dtype), torch.float32
    )
    stacked = torch.stack((local_sum.to(acc), local_denom.to(acc)))
    total, norm = _RedundantAllReduce.apply(stacked, reduce_groups).unbind()
    return (total / norm).to(local_sum.dtype)  # type: ignore[no-any-return]


def reduce_scalar_loss(
    local_sum: torch.Tensor,
    local_denom: torch.Tensor,
    reduction: str,
    reduce_groups: Groups,
) -> torch.Tensor:
    """Reduce a summed local loss over ``reduce_groups``.

    Parameters
    ----------
    local_sum : torch.Tensor
        Summed local loss.
    local_denom : torch.Tensor
        Summed local normalizer of the ``"mean"`` reduction.
    reduction : str
        ``"sum"`` or ``"mean"``.
    reduce_groups : tuple[dist.ProcessGroup | None, ...]
        Groups across which the batch and spatial dimensions are partitioned.

    Returns
    -------
    torch.Tensor
        Reduced scalar loss.

    Raises
    ------
    ValueError
        If ``reduction`` is invalid.
    """
    if reduction == "sum":
        return _RedundantAllReduce.apply(local_sum, reduce_groups)  # type: ignore[no-any-return]
    if reduction == "mean":
        return _mean_over_groups(local_sum, local_denom, reduce_groups)
    raise ValueError(f"invalid reduction: {reduction!r}")


def reduce_loss(
    per_element: torch.Tensor,
    denominator: torch.Tensor | None,
    reduction: str,
    reduce_groups: Groups,
) -> torch.Tensor:
    """Apply the ``"none"``, ``"sum"`` or ``"mean"`` reduction to a loss.

    Parameters
    ----------
    per_element : torch.Tensor
        Local per-element loss.
    denominator : torch.Tensor | None
        Local per-element weights of the ``"mean"`` reduction.
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    reduce_groups : tuple[dist.ProcessGroup | None, ...]
        Groups across which the batch and spatial dimensions are partitioned.

    Returns
    -------
    torch.Tensor
        ``per_element`` for ``"none"``, otherwise the reduced scalar loss.

    Raises
    ------
    ValueError
        If ``reduction`` is invalid.
    """
    if reduction == "none":
        return per_element
    local_sum = per_element.sum()
    if reduction == "sum":
        return _RedundantAllReduce.apply(local_sum, reduce_groups)  # type: ignore[no-any-return]
    if reduction == "mean":
        assert denominator is not None
        return _mean_over_groups(local_sum, denominator.detach().sum(), reduce_groups)
    raise ValueError(f"invalid reduction: {reduction!r}")
