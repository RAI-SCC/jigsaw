"""Shared primitives for the distributed loss functions.

Two kinds of tensor sharding are relevant to a loss evaluated on a process mesh:

* the *class* dimension of a classification input (``dim``, ``1`` by convention)
  may be partitioned across a ``channel_group``.  Any reduction over classes
  (``log_softmax``, gathering the target log-probability, the label-smoothing
  term) then needs a collective over that group;
* every remaining dimension (batch and spatial) may be partitioned across one or
  more ``reduce_groups``.  The final ``"mean"`` / ``"sum"`` reduction then needs
  a collective over those groups.

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

Groups = tuple[Optional[dist.ProcessGroup], ...]
GroupsArg = Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]


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


def _normalize_groups(groups: GroupsArg) -> Groups:
    """Coerce a user-supplied group argument into a tuple of groups.

    Parameters
    ----------
    groups : GroupsArg
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


def redundant_all_reduce(x: torch.Tensor, groups: GroupsArg) -> torch.Tensor:
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


@torch.no_grad()
def distributed_argmax(
    x: torch.Tensor,
    dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Arg-max along a dimension that may be sharded across ``group``.

    Returns *global* indices: on each rank the local arg-max is offset by
    ``group.rank() * x.shape[dim]``, then the rank holding the global maximum
    wins.  Ties are broken towards the lowest global index (a MIN all-reduce over
    the winning candidates), which is deterministic across ranks.

    Parameters
    ----------
    x : torch.Tensor
        Values whose dimension ``dim`` holds this rank's shard.
    dim : int
        Dimension to reduce.
    group : Optional[dist.ProcessGroup]
        Group across which ``dim`` is partitioned.  ``None`` / size ``1`` means
        the dimension is local and a plain :func:`torch.argmax` is returned.

    Returns
    -------
    torch.Tensor
        Global arg-max indices, shape ``x.shape`` with ``dim`` removed.
    """
    local_max, local_arg = x.max(dim=dim)
    global_arg = local_arg + _group_rank(group) * x.shape[dim]
    if not _active(group):
        return global_arg

    n_global = x.shape[dim] * group.size()  # type: ignore[union-attr]
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)

    candidate = torch.where(
        local_max == global_max,
        global_arg,
        global_arg.new_full((), n_global),
    )
    dist.all_reduce(candidate, op=dist.ReduceOp.MIN, group=group)
    return candidate


def reduce_loss(
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
