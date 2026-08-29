"""Autograd definitions for distributed loss functions.

The classification losses in this module operate on tensors that are sharded
across a process mesh.  Two kinds of sharding are relevant:

* the *class* dimension (``dim`` of the logits, ``1`` by convention) may be
  partitioned across a ``channel_group``.  Any reduction over classes
  (``log_softmax``, gathering the target log-probability, the label-smoothing
  term) then needs a collective over that group;
* every remaining dimension (batch and spatial) may be partitioned across one or
  more ``reduce_groups``.  The final ``"mean"`` / ``"sum"`` reduction of the loss
  then needs a collective over those groups.

Only two custom autograd primitives are required:

``_DistributedLogSoftmax``
    ``log_softmax`` along a possibly-sharded dimension, mirroring
    :class:`jigsaw.modules.normalization.functional._DistributedSoftmax`.

``_RedundantAllReduce``
    A sum all-reduce whose result every rank recomputes *identically* (the loss
    value, the gathered target log-probability, ...).  Because the downstream
    computation is redundant across the ranks of the group, the gradient is the
    identity -- all-reducing it again would scale it by the group size.

Everything else (``gather``, ``where``, ``binary_cross_entropy``, ...) is an
ordinary differentiable ``torch`` op, so the backward passes of the losses
themselves are produced by autograd.
"""

from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

Groups = tuple[Optional[dist.ProcessGroup], ...]


def _active(group: Optional[dist.ProcessGroup]) -> bool:
    """Return whether ``group`` describes more than one rank.

    Parameters
    ----------
    group : Optional[dist.ProcessGroup]
        Process group, or ``None`` for a purely local dimension.

    Returns
    -------
    bool
        ``True`` if a collective over ``group`` would exchange data.
    """
    return group is not None and group.size() > 1


def _normalize_groups(
    groups: Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup],
) -> Groups:
    """Coerce a user-supplied group argument into a tuple of groups.

    Parameters
    ----------
    groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
        ``None``, a single process group, or an iterable of process groups.

    Returns
    -------
    tuple[Optional[dist.ProcessGroup], ...]
        Normalized, hashable tuple of groups.
    """
    if groups is None:
        return ()
    if isinstance(groups, dist.ProcessGroup):
        return (groups,)
    return tuple(groups)


def _group_size(group: Optional[dist.ProcessGroup]) -> int:
    """Return the number of ranks in ``group`` (``1`` when ``group`` is ``None``)."""
    return 1 if group is None else group.size()


def _group_rank(group: Optional[dist.ProcessGroup]) -> int:
    """Return the caller's rank within ``group`` (``0`` when ``group`` is ``None``)."""
    return 0 if group is None else group.rank()


class _RedundantAllReduce(torch.autograd.Function):
    """Sum all-reduce with an identity gradient.

    Used where the reduced value feeds an identical (redundant) computation on
    every rank of the group -- the loss scalar, the gathered target
    log-probability, the label-smoothing class sum.  In that situation each rank
    independently seeds the same upstream gradient, so the correct local
    gradient is exactly ``grad_output``; a second all-reduce would multiply it by
    the group size.
    """

    @staticmethod
    def forward(x: torch.Tensor, groups: Groups) -> torch.Tensor:
        """Sum ``x`` across every active group in ``groups``.

        Parameters
        ----------
        x : torch.Tensor
            Local contribution.
        groups : tuple[Optional[dist.ProcessGroup], ...]
            Groups to reduce over.

        Returns
        -------
        torch.Tensor
            ``x`` summed over the ranks of every active group.
        """
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
        """Store nothing; the gradient does not depend on saved state."""
        return None

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """Pass the gradient straight through."""
        return grad_output, None


def redundant_all_reduce(
    x: torch.Tensor,
    groups: Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup],
) -> torch.Tensor:
    """Differentiably sum ``x`` across ``groups`` (see :class:`_RedundantAllReduce`)."""
    return _RedundantAllReduce.apply(x, _normalize_groups(groups))  # type: ignore[no-any-return]


class _DistributedLogSoftmax(torch.autograd.Function):
    """``log_softmax`` along a dimension that may be sharded across a group."""

    @staticmethod
    def forward(
        x: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]
    ) -> torch.Tensor:
        """Compute ``log_softmax(x, dim)`` with an all-reduce over ``group``.

        Parameters
        ----------
        x : torch.Tensor
            Logits, with the class dimension ``dim`` holding this rank's shard.
        dim : int
            Dimension along which to normalize.
        group : Optional[dist.ProcessGroup]
            Group across which ``dim`` is partitioned.

        Returns
        -------
        torch.Tensor
            Log-probabilities for this rank's slice of ``dim``.
        """
        local_max = x.amax(dim=dim, keepdim=True)
        if _active(group):
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)

        shifted = x - local_max
        sum_exp = shifted.exp().sum(dim=dim, keepdim=True)
        if _active(group):
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=group)

        return shifted - sum_exp.log()

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[torch.Tensor, int, Optional[dist.ProcessGroup]],
        output: torch.Tensor,
    ) -> None:
        """Save the log-probabilities and reduction metadata for backward."""
        _, dim, group = inputs
        ctx.dim = dim  # type: ignore[attr-defined]
        ctx.group = group  # type: ignore[attr-defined]
        ctx.save_for_backward(output)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        """Backprop through the distributed ``log_softmax``.

        ``grad_x = grad_output - softmax * sum_over_classes(grad_output)`` where
        the class sum spans the whole (possibly sharded) dimension.
        """
        (log_probs,) = ctx.saved_tensors  # type: ignore[attr-defined]
        dim = ctx.dim  # type: ignore[attr-defined]
        group = ctx.group  # type: ignore[attr-defined]

        grad_sum = grad_output.sum(dim=dim, keepdim=True)
        if _active(group):
            dist.all_reduce(grad_sum, op=dist.ReduceOp.SUM, group=group)

        grad_x = grad_output - log_probs.exp() * grad_sum
        return grad_x, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        dim: int,
        group: Optional[dist.ProcessGroup],
    ) -> tuple[torch.Tensor, int]:
        """Vmap rule: apply the op to the batched logits unchanged."""
        return _DistributedLogSoftmax.apply(x, dim, group), 0


def distributed_log_softmax(
    x: torch.Tensor, dim: int = 1, channel_group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """Differentiable ``log_softmax`` over a dimension sharded across ``channel_group``."""
    return _DistributedLogSoftmax.apply(x, dim, channel_group)  # type: ignore[no-any-return]


def _reduce_loss(
    per_element: torch.Tensor,
    denominator: Optional[torch.Tensor],
    reduction: str,
    reduce_groups: Groups,
) -> torch.Tensor:
    """Apply the final ``"none"`` / ``"sum"`` / ``"mean"`` reduction.

    Parameters
    ----------
    per_element : torch.Tensor
        Local per-element loss shard.
    denominator : Optional[torch.Tensor]
        Local per-element normalization weights for ``"mean"`` (summed here);
        ignored otherwise.
    reduction : str
        One of ``"none"``, ``"sum"``, ``"mean"``.
    reduce_groups : tuple[Optional[dist.ProcessGroup], ...]
        Groups across which batch/spatial dimensions are partitioned.

    Returns
    -------
    torch.Tensor
        The reduced loss (a scalar for ``"sum"`` / ``"mean"``, the local shard
        for ``"none"``).
    """
    if reduction == "none":
        return per_element
    local_sum = per_element.sum()
    if reduction == "sum":
        return _RedundantAllReduce.apply(local_sum, reduce_groups)  # type: ignore[no-any-return]
    if reduction == "mean":
        assert denominator is not None
        total = _RedundantAllReduce.apply(local_sum, reduce_groups)
        norm = _RedundantAllReduce.apply(denominator.detach().sum(), reduce_groups)
        return total / norm  # type: ignore[no-any-return]
    raise ValueError(f"invalid reduction: {reduction!r}")


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
    reduce_groups: Optional[
        Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup
    ] = None,
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
    reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
        Groups sharding the batch/spatial dimensions.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    dim = dim % log_probs.ndim
    channel_group = channel_group
    groups = _normalize_groups(reduce_groups)

    nll, elem_weight, _ = _nll_terms(
        log_probs, target, weight, ignore_index, dim, channel_group
    )
    return _reduce_loss(nll, elem_weight, reduction, groups)


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
    reduce_groups: Optional[
        Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup
    ] = None,
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
    reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
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
        return _reduce_loss(per_element, denom, reduction, groups)

    if label_smoothing != 0.0 and weight is not None:
        raise NotImplementedError("label_smoothing with weight is not supported")

    nll, elem_weight, valid = _nll_terms(
        log_probs, target, weight, ignore_index, dim, channel_group
    )

    if label_smoothing == 0.0:
        return _reduce_loss(nll, elem_weight, reduction, groups)

    n_global = log_probs.shape[dim] * _group_size(channel_group)
    class_sum = _RedundantAllReduce.apply(log_probs.sum(dim=dim), (channel_group,))
    smooth = torch.where(valid, -class_sum / n_global, class_sum.new_zeros(()))

    eps = label_smoothing
    per_element = (1.0 - eps) * nll + eps * smooth
    return _reduce_loss(per_element, valid.to(logits.dtype), reduction, groups)


def distributed_binary_cross_entropy(
    probs: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    reduce_groups: Optional[
        Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup
    ] = None,
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
    reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
        Every group across which ``probs`` is partitioned.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    groups = _normalize_groups(reduce_groups)
    per_element = F.binary_cross_entropy(probs, target, weight=weight, reduction="none")
    denom = per_element.new_full((), float(per_element.numel()))
    return _reduce_loss(per_element, denom, reduction, groups)


def distributed_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    reduce_groups: Optional[
        Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup
    ] = None,
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
    reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
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
    return _reduce_loss(per_element, denom, reduction, groups)


def _distributed_elementwise_loss(
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
    return _reduce_loss(per_element, denom, reduction, reduce_groups)


def distributed_mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    reduce_groups: Optional[
        Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup
    ] = None,
) -> torch.Tensor:
    """Distributed mean-squared-error / L2 loss (see :func:`torch.nn.functional.mse_loss`).

    The loss is elementwise, so ``reduce_groups`` must list *every* group across
    which ``input`` is partitioned (spatial and channel alike).

    Parameters
    ----------
    input : torch.Tensor
        Predictions (this rank's shard).
    target : torch.Tensor
        Regression targets (same shard as ``input``).
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.  ``"none"`` returns the local shard.
    reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
        Every group across which ``input`` is partitioned.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    groups = _normalize_groups(reduce_groups)
    per_element = F.mse_loss(input, target, reduction="none")
    return _distributed_elementwise_loss(per_element, reduction, groups)


def distributed_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    reduce_groups: Optional[
        Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup
    ] = None,
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
    reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
        Every group across which ``input`` is partitioned.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    groups = _normalize_groups(reduce_groups)
    per_element = F.l1_loss(input, target, reduction="none")
    return _distributed_elementwise_loss(per_element, reduction, groups)


def distributed_smooth_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 1.0,
    reduction: str = "mean",
    reduce_groups: Optional[
        Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup
    ] = None,
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
    reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
        Every group across which ``input`` is partitioned.

    Returns
    -------
    torch.Tensor
        The reduced loss.
    """
    groups = _normalize_groups(reduce_groups)
    per_element = F.smooth_l1_loss(input, target, reduction="none", beta=beta)
    return _distributed_elementwise_loss(per_element, reduction, groups)
