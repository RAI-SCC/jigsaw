"""Distributed classification loss functions."""

from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ...activation._functional import _LogSoftmax
from .._common import (
    GroupsArg,
    _active,
    _group_rank,
    _group_size,
    _normalize_groups,
    _RedundantAllReduce,
    reduce_loss,
    reduce_scalar_loss,
)


def _nll_terms(
    log_probs: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor],
    ignore_index: int,
    dim: int,
    channel_group: Optional[dist.ProcessGroup],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the per-element NLL, weight and validity mask."""
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
    """Compute the negative log-likelihood loss of a class-sharded input.

    Equivalent to :func:`torch.nn.functional.nll_loss`.

    Parameters
    ----------
    log_probs : torch.Tensor
        Log-probabilities with the class dimension ``dim`` holding the local shard.
    target : torch.Tensor
        Class indices, with ``dim`` removed from the shape of ``log_probs``.
    weight : torch.Tensor, optional
        Per-class weight of the global class dimension.
    ignore_index : int
        Target value that contributes neither loss nor normalization.
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    dim : int
        Class dimension.
    channel_group : dist.ProcessGroup, optional
        Group across which the class dimension is partitioned.
    reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
        Groups across which the batch and spatial dimensions are partitioned.

    Returns
    -------
    torch.Tensor
        Local loss for ``"none"``, otherwise the reduced scalar loss.
    """
    dim = dim % log_probs.ndim
    groups = _normalize_groups(reduce_groups)

    nll, elem_weight, _ = _nll_terms(
        log_probs, target, weight, ignore_index, dim, channel_group
    )
    return reduce_loss(nll, elem_weight, reduction, groups)


class _ChannelParallelCrossEntropy(torch.autograd.Function):
    """Autograd function of the cross-entropy over a class-sharded input."""

    @staticmethod
    def forward(
        logits: torch.Tensor,
        local_idx: torch.Tensor,
        local_hit: torch.Tensor,
        elem_weight: torch.Tensor,
        label_smoothing: float,
        dim: int,
        channel_group: Optional[dist.ProcessGroup],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the per-element loss and the softmax saved for the backward pass."""
        n_global = logits.shape[dim] * _group_size(channel_group)

        shift = logits.amax(dim=dim, keepdim=True)
        if _active(channel_group):
            dist.all_reduce(shift, op=dist.ReduceOp.MAX, group=channel_group)
        shifted = logits - shift
        exp_shifted = shifted.exp()

        packed = [
            exp_shifted.sum(dim=dim),
            torch.where(
                local_hit,
                shifted.gather(dim, local_idx).squeeze(dim),
                shifted.new_zeros(()),
            ),
        ]
        if label_smoothing != 0.0:
            packed.append(shifted.sum(dim=dim))
        stacked = torch.stack(packed, dim=-1)
        if _active(channel_group):
            dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=channel_group)

        sum_exp = stacked[..., 0]
        shifted_target = stacked[..., 1]
        log_sum_exp = sum_exp.log()
        nll = log_sum_exp - shifted_target

        if label_smoothing == 0.0:
            per_element = nll * elem_weight
        else:
            eps = label_smoothing
            smooth = log_sum_exp - stacked[..., 2] / n_global
            per_element = ((1.0 - eps) * nll + eps * smooth) * elem_weight

        softmax = exp_shifted / sum_exp.unsqueeze(dim)
        return per_element, softmax

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple,
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Save the tensors and metadata for the backward pass."""
        _, local_idx, local_hit, elem_weight, label_smoothing, dim, channel_group = (
            inputs
        )
        _, softmax = output
        ctx.dim = dim  # type: ignore[attr-defined]
        ctx.label_smoothing = label_smoothing  # type: ignore[attr-defined]
        ctx.n_global = softmax.shape[dim] * _group_size(channel_group)  # type: ignore[attr-defined]
        ctx.mark_non_differentiable(softmax)
        ctx.save_for_backward(softmax, local_idx, local_hit, elem_weight)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_per_element: torch.Tensor,
        _grad_softmax: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Compute the gradient of the logits."""
        softmax, local_idx, local_hit, elem_weight = ctx.saved_tensors  # type: ignore[attr-defined]
        dim = ctx.dim  # type: ignore[attr-defined]
        eps = ctx.label_smoothing  # type: ignore[attr-defined]

        onehot = torch.zeros_like(softmax).scatter(
            dim, local_idx, local_hit.unsqueeze(dim).to(softmax.dtype)
        )

        grad_logits = softmax - (1.0 - eps) * onehot
        if eps != 0.0:
            grad_logits = grad_logits - eps / ctx.n_global  # type: ignore[attr-defined]
        grad_logits = grad_logits * (grad_per_element * elem_weight).unsqueeze(dim)
        return grad_logits, None, None, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        logits: torch.Tensor,
        local_idx: torch.Tensor,
        local_hit: torch.Tensor,
        elem_weight: torch.Tensor,
        label_smoothing: float,
        dim: int,
        channel_group: Optional[dist.ProcessGroup],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
        """Batching rule for ``torch.vmap``."""

        def _front(t: torch.Tensor, bdim: Optional[int]) -> torch.Tensor:
            if bdim is None:
                return t.expand(info.batch_size, *t.shape)
            return t.movedim(bdim, 0)

        logits, local_idx, local_hit, elem_weight = (
            _front(t, bdim)
            for t, bdim in zip((logits, local_idx, local_hit, elem_weight), in_dims[:4])
        )
        class_dim = dim + 1 if dim >= 0 else dim
        out = _ChannelParallelCrossEntropy.apply(
            logits,
            local_idx,
            local_hit,
            elem_weight,
            label_smoothing,
            class_dim,
            channel_group,
        )
        return out, (0, 0)


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
    """Compute the cross-entropy loss of a class-sharded input.

    Equivalent to :func:`torch.nn.functional.cross_entropy`.

    Parameters
    ----------
    logits : torch.Tensor
        Unnormalized scores with the class dimension ``dim`` holding the local shard.
    target : torch.Tensor
        Class indices, with ``dim`` removed from the shape of ``logits``, or class
        probabilities with the shape of ``logits``.
    weight : torch.Tensor, optional
        Per-class weight of the global class dimension.
    ignore_index : int
        Target value that contributes neither loss nor normalization. Only applies
        to class-index targets.
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    label_smoothing : float
        Label smoothing in ``[0, 1)``.
    dim : int
        Class dimension.
    channel_group : dist.ProcessGroup, optional
        Group across which the class dimension is partitioned.
    reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
        Groups across which the batch and spatial dimensions are partitioned.

    Returns
    -------
    torch.Tensor
        Local loss for ``"none"``, otherwise the reduced scalar loss.

    Raises
    ------
    NotImplementedError
        If ``weight`` is combined with probability targets or ``label_smoothing``,
        or if ``label_smoothing`` is combined with probability targets.
    """
    dim = dim % logits.ndim
    groups = _normalize_groups(reduce_groups)

    is_prob_target = target.dim() == logits.dim() and target.is_floating_point()

    if is_prob_target:
        if weight is not None or label_smoothing != 0.0:
            raise NotImplementedError(
                "probability targets support neither weight nor label_smoothing"
            )
        log_probs = _LogSoftmax.apply(logits, dim, channel_group)
        local = -(target * log_probs).sum(dim=dim)
        per_element = _RedundantAllReduce.apply(local, (channel_group,))
        denom = per_element.new_full((), float(per_element.numel()))
        return reduce_loss(per_element, denom, reduction, groups)

    if label_smoothing != 0.0 and weight is not None:
        raise NotImplementedError("label_smoothing with weight is not supported")

    n_local = logits.shape[dim]
    offset = _group_rank(channel_group) * n_local
    valid = target != ignore_index
    target_clamped = target.clamp_min(0)
    in_range = (target_clamped >= offset) & (target_clamped < offset + n_local)
    local_hit = valid & in_range
    local_idx = (target_clamped - offset).clamp_(0, n_local - 1).unsqueeze(dim)

    if weight is not None:
        elem_weight = torch.where(valid, weight[target_clamped], weight.new_zeros(()))
    else:
        elem_weight = valid.to(logits.dtype)

    per_element, _ = _ChannelParallelCrossEntropy.apply(
        logits, local_idx, local_hit, elem_weight, label_smoothing, dim, channel_group
    )
    return reduce_loss(per_element, elem_weight, reduction, groups)


def distributed_binary_cross_entropy(
    probs: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    reduce_groups: GroupsArg = None,
) -> torch.Tensor:
    """Compute the binary cross-entropy of a sharded input.

    Equivalent to :func:`torch.nn.functional.binary_cross_entropy`.

    Parameters
    ----------
    probs : torch.Tensor
        Predicted probabilities in ``[0, 1]``.
    target : torch.Tensor
        Targets in ``[0, 1]`` with the shape of ``probs``.
    weight : torch.Tensor, optional
        Elementwise weight broadcastable to ``probs``.
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
        Every group across which ``probs`` is partitioned.

    Returns
    -------
    torch.Tensor
        Local loss for ``"none"``, otherwise the reduced scalar loss.
    """
    groups = _normalize_groups(reduce_groups)
    if reduction == "none":
        return F.binary_cross_entropy(probs, target, weight=weight, reduction="none")
    local_sum = F.binary_cross_entropy(probs, target, weight=weight, reduction="sum")
    local_count = probs.new_full((), float(probs.numel()))
    return reduce_scalar_loss(local_sum, local_count, reduction, groups)


def distributed_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    reduce_groups: GroupsArg = None,
) -> torch.Tensor:
    """Compute the binary cross-entropy with logits of a sharded input.

    Equivalent to :func:`torch.nn.functional.binary_cross_entropy_with_logits`.

    Parameters
    ----------
    logits : torch.Tensor
        Unnormalized scores.
    target : torch.Tensor
        Targets in ``[0, 1]`` with the shape of ``logits``.
    weight : torch.Tensor, optional
        Elementwise weight broadcastable to ``logits``.
    pos_weight : torch.Tensor, optional
        Weight of the positive examples, broadcastable to the class dimension.
    reduction : str
        ``"none"``, ``"sum"`` or ``"mean"``.
    reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
        Every group across which ``logits`` is partitioned.

    Returns
    -------
    torch.Tensor
        Local loss for ``"none"``, otherwise the reduced scalar loss.
    """
    groups = _normalize_groups(reduce_groups)
    if reduction == "none":
        return F.binary_cross_entropy_with_logits(
            logits, target, weight=weight, pos_weight=pos_weight, reduction="none"
        )
    local_sum = F.binary_cross_entropy_with_logits(
        logits, target, weight=weight, pos_weight=pos_weight, reduction="sum"
    )
    local_count = logits.new_full((), float(logits.numel()))
    return reduce_scalar_loss(local_sum, local_count, reduction, groups)
