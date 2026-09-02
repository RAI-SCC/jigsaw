"""Distributed classification losses: cross-entropy, NLL and binary cross-entropy.

The class dimension (``dim``, ``1`` by convention) may be sharded across
``channel_group``; the batch/spatial dimensions may be sharded across
``reduce_groups``.  See :mod:`jigsaw.modules.loss.common` for the shared
primitives and the collective-communication rationale.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..common import (
    GroupsArg,
    _DistributedLogSoftmax,
    _group_rank,
    _group_size,
    _normalize_groups,
    _RedundantAllReduce,
    reduce_loss,
)


def _nll_terms(
    log_probs: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor],
    ignore_index: int,
    dim: int,
    channel_group: Optional[dist.ProcessGroup],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(per_element_nll, per_element_weight, valid_mask)``.

    ``log_probs`` holds this rank's shard of the class dimension ``dim``.  The
    true-class log-probability is gathered locally where the target lands in this
    rank's class range and summed across ``channel_group``.

    Parameters
    ----------
    log_probs : torch.Tensor
        Log-probabilities, class dimension ``dim`` sharded.
    target : torch.Tensor
        Class indices, shape ``log_probs.shape`` without ``dim``.
    weight : Optional[torch.Tensor]
        Per-class weight of length ``C_global`` (replicated on every rank).
    ignore_index : int
        Target value whose elements contribute neither loss nor normalization.
    dim : int
        Class dimension of ``log_probs``.
    channel_group : Optional[dist.ProcessGroup]
        Group across which ``dim`` is partitioned.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Per-element NLL, per-element weight (``0`` on ignored elements), and the
        boolean validity mask.
    """
    n_local = log_probs.shape[dim]
    offset = _group_rank(channel_group) * n_local

    valid = target != ignore_index
    target_clamped = target.clamp_min(0)

    in_range = (target_clamped >= offset) & (target_clamped < offset + n_local)
    local_hit = valid & in_range
    local_idx = (target_clamped - offset).clamp_(0, n_local - 1).unsqueeze(dim)

    gathered = log_probs.gather(dim, local_idx).squeeze(dim)
    contrib = torch.where(local_hit, gathered, log_probs.new_zeros(()))
    true_log_prob = _RedundantAllReduce.apply(contrib, (channel_group,))

    if weight is not None:
        elem_weight = weight[target_clamped]
        elem_weight = torch.where(valid, elem_weight, weight.new_zeros(()))
    else:
        elem_weight = valid.to(log_probs.dtype)

    nll = -true_log_prob * elem_weight
    return nll, elem_weight, valid


def distributed_nll_loss(
    log_probs: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    dim: int = 1,
    channel_group: Optional[dist.ProcessGroup] = None,
    reduce_groups: GroupsArg = None,
) -> torch.Tensor:
    """Distributed negative log-likelihood loss (see :func:`torch.nn.functional.nll_loss`).

    Parameters
    ----------
    log_probs : torch.Tensor
        Log-probabilities with the (possibly sharded) class dimension at ``dim``.
    target : torch.Tensor
        Class indices, shape ``log_probs.shape`` with ``dim`` removed.
    weight : Optional[torch.Tensor]
        Per-class weight of length ``C_global`` (replicated on every rank).
    ignore_index : int
        Target value to skip.
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.  ``"none"`` returns the local shard.
    dim : int
        Class dimension.
    channel_group : Optional[dist.ProcessGroup]
        Group sharding the class dimension.
    reduce_groups : GroupsArg
        Groups sharding the batch/spatial dimensions.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    dim = dim % log_probs.ndim
    groups = _normalize_groups(reduce_groups)

    nll, elem_weight, _ = _nll_terms(
        log_probs, target, weight, ignore_index, dim, channel_group
    )
    return reduce_loss(nll, elem_weight, reduction, groups)


def distributed_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    dim: int = 1,
    channel_group: Optional[dist.ProcessGroup] = None,
    reduce_groups: GroupsArg = None,
) -> torch.Tensor:
    """Distributed cross-entropy loss (see :func:`torch.nn.functional.cross_entropy`).

    ``target`` may be class indices (shape ``logits.shape`` without ``dim``) or
    class probabilities (same shape as ``logits``, class dimension sharded like
    ``logits``).  Probability targets support neither ``ignore_index``,
    ``weight`` nor ``label_smoothing`` -- matching the constraints that would
    otherwise make the reference ambiguous.

    Parameters
    ----------
    logits : torch.Tensor
        Unnormalized scores with the (possibly sharded) class dimension at ``dim``.
    target : torch.Tensor
        Class indices or class probabilities.
    weight : Optional[torch.Tensor]
        Per-class weight of length ``C_global`` (replicated on every rank).
    ignore_index : int
        Target value to skip (index targets only).
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    label_smoothing : float
        Amount of label smoothing in ``[0, 1)``; requires ``weight is None``.
    dim : int
        Class dimension.
    channel_group : Optional[dist.ProcessGroup]
        Group sharding the class dimension.
    reduce_groups : GroupsArg
        Groups sharding the batch/spatial dimensions.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    dim = dim % logits.ndim
    groups = _normalize_groups(reduce_groups)
    log_probs = _DistributedLogSoftmax.apply(logits, dim, channel_group)

    is_prob_target = target.dim() == logits.dim() and target.is_floating_point()

    if is_prob_target:
        if weight is not None or label_smoothing != 0.0:
            raise NotImplementedError(
                "probability targets support neither weight nor label_smoothing"
            )
        local = -(target * log_probs).sum(dim=dim)
        per_element = _RedundantAllReduce.apply(local, (channel_group,))
        denom = per_element.new_full((), float(per_element.numel()))
        return reduce_loss(per_element, denom, reduction, groups)

    if label_smoothing != 0.0 and weight is not None:
        raise NotImplementedError("label_smoothing with weight is not supported")

    nll, elem_weight, valid = _nll_terms(
        log_probs, target, weight, ignore_index, dim, channel_group
    )

    if label_smoothing == 0.0:
        return reduce_loss(nll, elem_weight, reduction, groups)

    n_global = log_probs.shape[dim] * _group_size(channel_group)
    class_sum = _RedundantAllReduce.apply(log_probs.sum(dim=dim), (channel_group,))
    smooth = torch.where(valid, -class_sum / n_global, class_sum.new_zeros(()))

    eps = label_smoothing
    per_element = (1.0 - eps) * nll + eps * smooth
    return reduce_loss(per_element, valid.to(logits.dtype), reduction, groups)


def distributed_binary_cross_entropy(
    probs: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    reduce_groups: GroupsArg = None,
) -> torch.Tensor:
    """Distributed binary cross-entropy (see :func:`torch.nn.functional.binary_cross_entropy`).

    Every dimension is treated as an ordinary element dimension, so ``target`` and
    ``probs`` share a shape and ``reduce_groups`` must list *every* group across
    which that shape is partitioned (channel group included).

    Parameters
    ----------
    probs : torch.Tensor
        Predicted probabilities in ``[0, 1]`` (this rank's shard).
    target : torch.Tensor
        Binary / soft targets in ``[0, 1]`` (same shard as ``probs``).
    weight : Optional[torch.Tensor]
        Elementwise weight broadcastable to ``probs`` (this rank's shard).
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    reduce_groups : GroupsArg
        Every group across which ``probs`` is partitioned.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    groups = _normalize_groups(reduce_groups)
    per_element = F.binary_cross_entropy(probs, target, weight=weight, reduction="none")
    denom = per_element.new_full((), float(per_element.numel()))
    return reduce_loss(per_element, denom, reduction, groups)


def distributed_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    reduce_groups: GroupsArg = None,
) -> torch.Tensor:
    """Distributed BCE-with-logits (see :func:`torch.nn.functional.binary_cross_entropy_with_logits`).

    Parameters
    ----------
    logits : torch.Tensor
        Unnormalized scores (this rank's shard).
    target : torch.Tensor
        Binary / soft targets in ``[0, 1]`` (same shard as ``logits``).
    weight : Optional[torch.Tensor]
        Elementwise weight broadcastable to ``logits``.
    pos_weight : Optional[torch.Tensor]
        Weight of positive examples, broadcastable to the class dimension.
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    reduce_groups : GroupsArg
        Every group across which ``logits`` is partitioned.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    groups = _normalize_groups(reduce_groups)
    per_element = F.binary_cross_entropy_with_logits(
        logits, target, weight=weight, pos_weight=pos_weight, reduction="none"
    )
    denom = per_element.new_full((), float(per_element.numel()))
    return reduce_loss(per_element, denom, reduction, groups)
