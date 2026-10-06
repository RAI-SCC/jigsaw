"""In-graph gradient synchronization of replicated parameters.

A parameter that is replicated over a process group (e.g. a conv weight over the
spatial grid, or a linear bias over the row group) sees only its local share of
the gradient on each rank; the shares have to be summed over the group. This
module does that *inside the autograd graph*: :func:`sync_grad` routes the
parameters through an identity :class:`torch.autograd.Function` whose backward
all-reduces the incoming gradients. The reduction therefore happens upstream of
the parameter's ``AccumulateGrad`` node, which gives three properties a
``register_post_accumulate_grad_hook`` cannot:

* It completes before anything that observes ``p.grad`` runs -- in particular
  ``DistributedDataParallel``'s reducer hook on the same ``AccumulateGrad``
  node. An asynchronous post-accumulate hook races that reducer, which then
  overwrites ``p.grad`` with the un-summed local share.
* Every backward pass reduces its own gradient, not the running ``p.grad`` sum,
  so gradient accumulation over microbatches stays correct.
* Several parameters sharing a group tuple and dtype are reduced in one flat
  collective.

Usage: mark a parameter once with :func:`mark_replicated` where it is created,
and route it through :func:`sync_grad` (or :func:`synced_parameters` for
submodules whose forward jigsaw does not own) in the forward.
"""

from collections.abc import Sequence
from typing import Any, Optional

import torch
import torch.distributed as dist

_ATTR = "_jigsaw_replica_groups"

Groups = tuple[dist.ProcessGroup, ...]


def _as_groups(
    groups: "dist.ProcessGroup | Sequence[Optional[dist.ProcessGroup]]",
) -> Groups:
    """Normalize to a tuple of the groups with a real (>1) split."""
    seq = [groups] if isinstance(groups, dist.ProcessGroup) else list(groups)
    return tuple(g for g in seq if g is not None and g.size() > 1)


def mark_replicated(
    param: torch.Tensor,
    groups: "dist.ProcessGroup | Sequence[Optional[dist.ProcessGroup]]",
) -> None:
    """Mark a parameter as replicated over ``groups``.

    Its gradient is then summed over each group in turn by :func:`sync_grad`.
    Groups of size 1 (and ``None``) are dropped; marking with only such groups
    is a no-op.

    Parameters
    ----------
    param : torch.Tensor
        Parameter to mark.
    groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None]
        Process group(s) the parameter is replicated over.

    Raises
    ------
    RuntimeError
        If ``param`` is already marked. A second SUM over the same group would
        scale its gradient by the group size.
    """
    if replica_groups(param):
        raise RuntimeError(
            "parameter is already marked replicated; a second mark would sum "
            "its gradient twice"
        )
    groups = _as_groups(groups)
    if groups:
        setattr(param, _ATTR, groups)


def replica_groups(param: Optional[torch.Tensor]) -> Groups:
    """Return the groups ``param`` is marked replicated over, or ``()``."""
    return getattr(param, _ATTR, ())


class _SyncGrad(torch.autograd.Function):
    """Identity in forward; SUM all-reduce of the gradients in backward."""

    @staticmethod
    def forward(groups: Groups, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return views of ``tensors``."""
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
        """Save the groups for the backward pass."""
        ctx.groups = inputs[0]

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> tuple:
        """Sum the gradients over each group, in one flat collective per group.

        Autograd materializes a zero gradient for an unused output, so every
        rank reaching this node sends the same-sized buffer.
        """
        # ``cat`` copies, so the incoming gradients are never written in place.
        flat = torch.cat([g.reshape(-1) for g in grads])
        for group in ctx.groups:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        chunks = flat.split([g.numel() for g in grads])
        return (None, *(c.view_as(g) for c, g in zip(chunks, grads)))

    @staticmethod
    def vmap(
        info: Any, in_dims: tuple[Optional[int], ...], groups: Groups, *tensors: Any
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Optional[int], ...]]:
        """Batching rule: the reduction is elementwise, so reduce the whole batch."""
        return _SyncGrad.apply(groups, *tensors), tuple(in_dims[1:])


def sync_grad(*tensors: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], ...]:
    """Route marked parameters through an in-graph gradient all-reduce.

    Unmarked tensors, ``None`` and tensors that do not require a gradient are
    returned unchanged, as is everything when grad mode is off. The marked ones
    are grouped by ``(groups, dtype)`` and each group is reduced in one flat
    collective per process group.

    Parameters
    ----------
    *tensors : torch.Tensor | None
        Parameters, typically a module's weight and bias.

    Returns
    -------
    tuple[torch.Tensor | None, ...]
        The tensors to use in the forward, in the same order.
    """
    out = list(tensors)
    if not torch.is_grad_enabled():
        return tuple(out)
    buckets: dict[tuple[Groups, torch.dtype], list[int]] = {}
    for i, t in enumerate(tensors):
        groups = replica_groups(t)
        if groups and t is not None and t.requires_grad:
            buckets.setdefault((groups, t.dtype), []).append(i)
    for (groups, _), idx in buckets.items():
        synced = _SyncGrad.apply(groups, *(tensors[i] for i in idx))
        for i, s in zip(idx, synced):
            out[i] = s
    return tuple(out)


def synced_parameters(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return the marked parameters of ``module`` routed through :func:`sync_grad`.

    Meant for ``torch.func.functional_call(module, synced_parameters(module),
    ...)`` around submodules whose forward jigsaw does not own (a plain
    ``torch.nn.Linear`` replicated over the spatial group, say). Inside that
    call the module sees the synced tensors, which carry no mark, so a jigsaw
    submodule that calls :func:`sync_grad` itself does not reduce twice.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters to sync.

    Returns
    -------
    dict[str, torch.Tensor]
        Synced tensor per marked parameter name; empty if none is marked or
        grad mode is off.
    """
    marked = {n: p for n, p in module.named_parameters() if replica_groups(p)}
    if not marked:
        return {}
    synced = sync_grad(*marked.values())
    return {
        n: s
        for (n, p), s in zip(marked.items(), synced)
        if s is not p and s is not None
    }
