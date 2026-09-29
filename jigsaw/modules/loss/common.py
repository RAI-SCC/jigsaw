"""Shared primitives of the distributed loss functions.

The class dimension of a classification input is sharded across a
``channel_group`` and the remaining dimensions across ``reduce_groups``.
"""

from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist

Groups = tuple[Optional[dist.ProcessGroup], ...]
GroupsArg = Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]


def _active(group: Optional[dist.ProcessGroup]) -> bool:
    """Return whether ``group`` spans more than one rank."""
    return group is not None and group.size() > 1


def _normalize_groups(groups: GroupsArg) -> Groups:
    """Coerce a group argument into a tuple of groups."""
    if groups is None:
        return ()
    if isinstance(groups, dist.ProcessGroup):
        return (groups,)
    return tuple(groups)


def _group_size(group: Optional[dist.ProcessGroup]) -> int:
    """Return the size of ``group``, or ``1`` if it is ``None``."""
    return 1 if group is None else group.size()


def _group_rank(group: Optional[dist.ProcessGroup]) -> int:
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
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        groups: Groups,
    ) -> tuple[torch.Tensor, Optional[int]]:
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


class _DistributedLogSoftmax(torch.autograd.Function):
    """Autograd function of ``log_softmax`` over a dimension sharded across a group."""

    @staticmethod
    def forward(
        x: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]
    ) -> torch.Tensor:
        """Compute ``log_softmax`` with all-reduces over ``group``."""
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
        """Save the tensors and metadata for the backward pass."""
        _, dim, group = inputs
        ctx.dim = dim  # type: ignore[attr-defined]
        ctx.group = group  # type: ignore[attr-defined]
        ctx.save_for_backward(output)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        """Compute the gradient of the logits."""
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
        """Batching rule for ``torch.vmap``."""
        return _DistributedLogSoftmax.apply(x, dim, group), 0


def distributed_log_softmax(
    x: torch.Tensor, dim: int = 1, channel_group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """Compute ``log_softmax`` over a dimension sharded across ``channel_group``.

    Parameters
    ----------
    x : torch.Tensor
        Logits with the class dimension ``dim`` holding the local shard.
    dim : int
        Class dimension.
    channel_group : dist.ProcessGroup, optional
        Group across which ``dim`` is partitioned.

    Returns
    -------
    torch.Tensor
        Log-probabilities of the local shard.
    """
    return _DistributedLogSoftmax.apply(x, dim, channel_group)  # type: ignore[no-any-return]


@torch.no_grad()
def distributed_argmax(
    x: torch.Tensor,
    dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Compute the arg-max along a dimension sharded across ``group``.

    Ties are resolved towards the lowest global index.

    Parameters
    ----------
    x : torch.Tensor
        Values with the dimension ``dim`` holding the local shard.
    dim : int
        Dimension to reduce.
    group : dist.ProcessGroup, optional
        Group across which ``dim`` is partitioned.

    Returns
    -------
    torch.Tensor
        Global arg-max indices with ``dim`` removed.
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


def _mean_over_groups(
    local_sum: torch.Tensor,
    local_denom: torch.Tensor,
    reduce_groups: Groups,
) -> torch.Tensor:
    """Sum the numerator and denominator over ``reduce_groups`` and divide."""
    stacked = torch.stack((local_sum, local_denom.to(local_sum.dtype)))
    total, norm = _RedundantAllReduce.apply(stacked, reduce_groups).unbind()
    return total / norm  # type: ignore[no-any-return]


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
    denominator: Optional[torch.Tensor],
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
