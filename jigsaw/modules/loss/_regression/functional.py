"""Distributed regression loss functions."""

import functools
from typing import Callable

import torch
import torch.nn.functional as F

from .._common import GroupsArg, _normalize_groups, reduce_scalar_loss

_ElementwiseLoss = Callable[..., torch.Tensor]


def _distributed_elementwise_loss(
    fn: _ElementwiseLoss,
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str,
    reduce_groups: GroupsArg,
) -> torch.Tensor:
    """Evaluate an elementwise loss and reduce it over ``reduce_groups``."""
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
    """Compute the mean-squared-error loss of a sharded input.

    Equivalent to :func:`torch.nn.functional.mse_loss`.

    Parameters
    ----------
    input : torch.Tensor
        Predictions.
    target : torch.Tensor
        Targets with the shape of ``input``.
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
        Every group across which ``input`` is partitioned.

    Returns
    -------
    torch.Tensor
        Local loss for ``"none"``, otherwise the reduced scalar loss.
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
    """Compute the mean-absolute-error loss of a sharded input.

    Equivalent to :func:`torch.nn.functional.l1_loss`.

    Parameters
    ----------
    input : torch.Tensor
        Predictions.
    target : torch.Tensor
        Targets with the shape of ``input``.
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
        Every group across which ``input`` is partitioned.

    Returns
    -------
    torch.Tensor
        Local loss for ``"none"``, otherwise the reduced scalar loss.
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
    """Compute the smooth-L1 loss of a sharded input.

    Equivalent to :func:`torch.nn.functional.smooth_l1_loss`.

    Parameters
    ----------
    input : torch.Tensor
        Predictions.
    target : torch.Tensor
        Targets with the shape of ``input``.
    beta : float
        Transition point between the quadratic and the linear regime.
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
        Every group across which ``input`` is partitioned.

    Returns
    -------
    torch.Tensor
        Local loss for ``"none"``, otherwise the reduced scalar loss.
    """
    return _distributed_elementwise_loss(
        functools.partial(F.smooth_l1_loss, beta=beta),
        input,
        target,
        reduction,
        reduce_groups,
    )
