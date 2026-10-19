"""In-graph gradient synchronization of replicated parameters."""

import os
from typing import Any, Optional, cast

import torch
import torch.distributed as dist

from ._ddp import check_active_ddp
from ._joint import JointCommHookState, active_joint_state, joint_state_of
from ._replicas import Groups, replica_groups

#: Mutation-testing switch: with ``JIGSAW_DISABLE_GRAD_SYNC=1``, :func:`sync_grad`
#: is a pass-through and replicated parameters keep only their local gradient
#: share. Training is then wrong; the test suite must notice.
_SYNC_DISABLED = os.environ.get("JIGSAW_DISABLE_GRAD_SYNC", "") == "1"


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
        # Summing over each group in turn sums over their product.
        for group in ctx.groups:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        # Unflatten into views of the reduced buffer, one per input.
        chunks = flat.split([g.numel() for g in grads])
        return (None, *(c.view_as(g) for c, g in zip(chunks, grads)))

    @staticmethod
    def vmap(
        info: Any, in_dims: tuple[Optional[int], ...], groups: Groups, *tensors: Any
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Optional[int], ...]]:
        """Batching rule: the reduction is elementwise, so reduce the whole batch."""
        return _SyncGrad.apply(groups, *tensors), tuple(in_dims[1:])


class _ScaleGrad(torch.autograd.Function):
    """Identity in forward; multiplies the gradients by ``factor`` in backward.

    Stands in for :class:`_SyncGrad` under the joint comm hook, which averages
    over ``factor`` times as many ranks as the plain ``dp`` mean.
    """

    @staticmethod
    def forward(factor: float, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return views of ``tensors``."""
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
        """Save the factor for the backward pass."""
        ctx.factor = inputs[0]

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> tuple:
        """Scale every gradient by ``factor``."""
        return (None, *(g * ctx.factor for g in grads))

    @staticmethod
    def vmap(
        info: Any, in_dims: tuple[Optional[int], ...], factor: float, *tensors: Any
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Optional[int], ...]]:
        """Batching rule: the scaling is elementwise, so scale the whole batch."""
        return _ScaleGrad.apply(factor, *tensors), tuple(in_dims[1:])


def sync_grad(*tensors: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], ...]:
    """Route marked parameters through an in-graph gradient all-reduce.

    Unmarked tensors, ``None`` and tensors not requiring grad pass through
    unchanged, as does everything when grad mode is off. Marked tensors are
    bucketed by ``(groups, dtype)``, one flat collective per bucket and group.
    Inside the forward of a DDP with :func:`register_joint_comm_hook` (anywhere,
    under DDP's Python reducer), the parameters DDP reduces are only scaled;
    the hook does the sum.

    Parameters
    ----------
    *tensors : torch.Tensor | None
        Parameters, typically a module's weight and bias.

    Returns
    -------
    tuple[torch.Tensor | None, ...]
        The tensors to use in the forward, in the same order.
    """
    check_active_ddp()
    out = list(tensors)
    if _SYNC_DISABLED or not torch.is_grad_enabled():
        return tuple(out)
    # Joint-hook state of the DDP whose forward is running, if any.
    joint = active_joint_state()
    # (groups, dtype, joint state if only scaled) -> indices
    buckets: "dict[tuple[Groups, torch.dtype, Optional[JointCommHookState]], list[int]]"
    # Buckets will hold the parameters that need sync., that also share an autograd node.
    # Examples include weight and bias of Conv layers and layer norms
    # Only bias term over a distributed linear
    # 
    buckets = {}
    for i, t in enumerate(tensors):
        groups = replica_groups(t)
        if groups and t is not None and t.requires_grad:
            # Parameters the joint hook reduces only need scaling here.
            # Only parameters carry a mark.
            state = joint_state_of(cast(torch.nn.Parameter, t), joint)
            buckets.setdefault((groups, t.dtype, state), []).append(i)
    for (groups, _, state), idx in buckets.items():
        ts = (tensors[i] for i in idx)
        if state is not None:
            synced = _ScaleGrad.apply(state.factor(groups), *ts)
        else:
            synced = _SyncGrad.apply(groups, *ts)
        # Scatter the synced views back to their input positions.
        for i, s in zip(idx, synced):
            out[i] = s
    return tuple(out)


def synced_parameters(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return the marked parameters of ``module`` routed through :func:`sync_grad`.

    Meant for ``torch.func.functional_call(module, synced_parameters(module),
    ...)``; :func:`synced_call` does this for a submodule. Call it directly
    from a module's own forward, re-entering the forward only if the result is
    non-empty. The synced tensors carry no mark, so the re-entered forward, and
    nested jigsaw modules calling :func:`sync_grad` on them, do not reduce
    twice.

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
    # Keep only what sync_grad actually rerouted (nothing with grad mode off).
    return {
        n: s
        for (n, p), s in zip(marked.items(), synced)
        if s is not p and s is not None
    }


def synced_call(module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
    """Call ``module`` with its marked parameters routed through :func:`sync_grad`.

    For submodules jigsaw does not own, e.g. a plain ``torch.nn.LayerNorm``
    replicated over the spatial group: their marked parameters are summed over
    their replica groups in the backward pass, unmarked ones pass through. Do
    not call it on ``self`` from a forward, which would recurse; use
    :func:`synced_parameters` there.

    Parameters
    ----------
    module : torch.nn.Module
        Module to call.
    *args : Any
        Positional inputs of ``module``.
    **kwargs : Any
        Keyword inputs of ``module``.

    Returns
    -------
    Any
        Output of ``module``.
    """
    synced = synced_parameters(module)
    if not synced:
        return module(*args, **kwargs)
    return torch.func.functional_call(module, synced, args, kwargs)
