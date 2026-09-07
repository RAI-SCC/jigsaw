"""Distributed regression losses: L2 (MSE), L1 and smooth-L1 (Huber).

These losses are elementwise -- there is no distinguished class dimension -- so
``reduce_groups`` must list *every* group across which the input is partitioned
(spatial and channel alike).  Only the final ``"mean"`` / ``"sum"`` reduction
needs a collective; ``"mean"`` divides by the global element count so the result
matches the non-distributed loss on the gathered tensor.  The per-element term
and the backward pass are entirely local.
"""

import functools
from typing import Callable

import torch
import torch.nn.functional as F

from ..common import GroupsArg, _normalize_groups, reduce_scalar_loss

_ElementwiseLoss = Callable[..., torch.Tensor]


def _distributed_elementwise_loss(
    fn: _ElementwiseLoss,
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str,
    reduce_groups: GroupsArg,
) -> torch.Tensor:
    """Evaluate an elementwise regression loss and reduce it over the mesh.

    For ``"sum"`` / ``"mean"`` the local contribution is obtained straight from
    ``fn(..., reduction="sum")`` -- a single fused kernel with no per-element
    tensor to materialize -- and only the global element count is needed for the
    ``"mean"`` denominator.  ``"none"`` still returns the full local shard.

    Parameters
    ----------
    fn : Callable
        A ``torch.nn.functional`` loss taking ``reduction=`` (e.g. ``F.mse_loss``).
    input, target : torch.Tensor
        This rank's matching shards.
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    reduce_groups : GroupsArg
        Every group across which ``input`` is partitioned.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    groups = _normalize_groups(reduce_groups)
    if reduction == "none":
        return fn(input, target, reduction="none")
    local_sum = fn(input, target, reduction="sum")
    local_count = input.new_full((), float(input.numel()))
    return reduce_scalar_loss(local_sum, local_count, reduction, groups)


def distributed_mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    reduce_groups: GroupsArg = None,
) -> torch.Tensor:
    """Distributed mean-squared-error / L2 loss (see :func:`torch.nn.functional.mse_loss`).

    Parameters
    ----------
    input : torch.Tensor
        Predictions (this rank's shard).
    target : torch.Tensor
        Regression targets (same shard as ``input``).
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.  ``"none"`` returns the local shard.
    reduce_groups : GroupsArg
        Every group across which ``input`` is partitioned.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    return _distributed_elementwise_loss(
        F.mse_loss, input, target, reduction, reduce_groups
    )


def distributed_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    reduce_groups: GroupsArg = None,
) -> torch.Tensor:
    """Distributed mean-absolute-error / L1 loss (see :func:`torch.nn.functional.l1_loss`).

    Parameters
    ----------
    input : torch.Tensor
        Predictions (this rank's shard).
    target : torch.Tensor
        Regression targets (same shard as ``input``).
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    reduce_groups : GroupsArg
        Every group across which ``input`` is partitioned.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    return _distributed_elementwise_loss(
        F.l1_loss, input, target, reduction, reduce_groups
    )


def distributed_smooth_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 1.0,
    reduction: str = "mean",
    reduce_groups: GroupsArg = None,
) -> torch.Tensor:
    """Distributed smooth-L1 (Huber) loss (see :func:`torch.nn.functional.smooth_l1_loss`).

    Parameters
    ----------
    input : torch.Tensor
        Predictions (this rank's shard).
    target : torch.Tensor
        Regression targets (same shard as ``input``).
    beta : float
        Transition point between the quadratic and linear regimes.
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    reduce_groups : GroupsArg
        Every group across which ``input`` is partitioned.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    return _distributed_elementwise_loss(
        functools.partial(F.smooth_l1_loss, beta=beta),
        input,
        target,
        reduction,
        reduce_groups,
    )
