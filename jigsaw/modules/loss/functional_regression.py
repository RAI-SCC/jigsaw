"""Distributed regression losses: L2 (MSE), L1 and smooth-L1 (Huber).

These losses are elementwise -- there is no distinguished class dimension -- so
``reduce_groups`` must list *every* group across which the input is partitioned
(spatial and channel alike).  Only the final ``"mean"`` / ``"sum"`` reduction
needs a collective; ``"mean"`` divides by the global element count so the result
matches the non-distributed loss on the gathered tensor.  The per-element term
and the backward pass are entirely local.
"""

import torch
import torch.nn.functional as F

from .common import Groups, GroupsArg, _normalize_groups, reduce_loss


def _reduce_elementwise(
    per_element: torch.Tensor,
    reduction: str,
    reduce_groups: Groups,
) -> torch.Tensor:
    """Reduce a per-element regression loss over the sharded element dimensions.

    ``"mean"`` divides by the *global* element count (the local counts are summed
    across ``reduce_groups``), matching how :mod:`torch.nn.functional` averages
    over every element of the tensor.

    Parameters
    ----------
    per_element : torch.Tensor
        Local unreduced loss shard.
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    reduce_groups : tuple[Optional[dist.ProcessGroup], ...]
        Every group across which the loss tensor is partitioned.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    denom = per_element.new_full((), float(per_element.numel()))
    return reduce_loss(per_element, denom, reduction, reduce_groups)


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
    groups = _normalize_groups(reduce_groups)
    per_element = F.mse_loss(input, target, reduction="none")
    return _reduce_elementwise(per_element, reduction, groups)


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
    groups = _normalize_groups(reduce_groups)
    per_element = F.l1_loss(input, target, reduction="none")
    return _reduce_elementwise(per_element, reduction, groups)


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
    groups = _normalize_groups(reduce_groups)
    per_element = F.smooth_l1_loss(input, target, reduction="none", beta=beta)
    return _reduce_elementwise(per_element, reduction, groups)
