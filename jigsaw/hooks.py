"""Gradient hooks."""

from collections.abc import Sequence
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.nn.parameter import Parameter


def make_grad_sync_hook(
    group: "dist.ProcessGroup | Sequence[dist.ProcessGroup]",
    op: Any = dist.ReduceOp.SUM,
    async_op: bool = True,
) -> Callable:
    """Create a hook that all-reduces parameter gradients.

    With ``async_op=True`` the all-reduce overlaps with the backward pass and
    completes before the optimizer step. A sequence of groups is reduced
    sequentially and blocking.

    Parameters
    ----------
    group : dist.ProcessGroup | Sequence[dist.ProcessGroup]
        Process group(s) the gradients should be synchronized over.
    op : dist.ReduceOp
        Reduction operation.
    async_op : bool
        Whether to communicate asynchronously (single group only).

    Returns
    -------
    Callable
        Post-accumulate-grad hook.
    """
    groups = [group] if isinstance(group, dist.ProcessGroup) else list(group)
    handles: list[torch.distributed.Work] = []

    def _finalize() -> None:
        """Wait for all queued gradient communications to complete."""
        for h in handles:
            h.wait()
        handles.clear()

    def hook(param: Parameter) -> None:
        """Synchronize or scale a parameter gradient.

        Parameters
        ----------
        param : Parameter
            Parameter whose gradient is processed.
        """
        if param.grad is None:
            return
        if async_op and len(groups) == 1:
            if not handles:
                # first grad of this backward → queue the wait to run when backward finishes
                Variable._execution_engine.queue_callback(_finalize)
            handles.append(
                dist.all_reduce(param.grad, op=op, group=groups[0], async_op=True)
            )
        else:
            for g in groups:
                dist.all_reduce(param.grad, op=op, group=g, async_op=False)

    return hook


def make_grad_scaling_hook(parallelism: int) -> Callable:
    """Create a hook that divides parameter gradients by the parallelism.

    Matches the non-parallel result when the loss uses ``reduction='mean'``.

    Parameters
    ----------
    parallelism : int
        Scaling factor for the gradient.

    Returns
    -------
    Callable
        Post-accumulate-grad hook.
    """

    def hook(param: Parameter) -> None:
        """Synchronize or scale a parameter gradient.

        Parameters
        ----------
        param : Parameter
            Parameter whose gradient is processed.
        """
        if param.grad is not None:
            param.grad = param.grad / parallelism

    return hook


def make_mp_dp_comm_hook(
    mp_group_for: Callable[[Parameter], "dist.ProcessGroup | None"],
    dp_group: "dist.ProcessGroup | None" = None,
    mp_op: Any = dist.ReduceOp.SUM,
) -> Callable:
    """Create a DDP communication hook that reduces model-parallel, then data-parallel.

    A post-accumulate-grad hook cannot be used for the model-parallel reduction:
    DDP's reducer consumes ``param.grad`` in the same autograd node that
    accumulates it, so deferring the wait until the end of the backward pass
    lets DDP reduce a gradient the model-parallel all-reduce has not finished
    writing. Doing both reductions here instead makes the order explicit -- the
    data-parallel all-reduce is *chained* on the model-parallel ones rather than
    merely issued after them -- while keeping both off the critical path: DDP
    launches each bucket as it fills during the backward pass and only joins the
    returned future in ``finalize_backward``.

    Parameters
    ----------
    mp_group_for : Callable[[Parameter], dist.ProcessGroup | None]
        Maps a parameter to the model-parallel group its gradient must be
        reduced over, or ``None`` to leave that gradient alone. Buckets do not
        respect parallel-group boundaries, so the group is looked up per
        parameter.
    dp_group : dist.ProcessGroup | None
        Data-parallel group. Defaults to ``dist.group.WORLD``; it must be the
        group DDP itself was constructed with.
    mp_op : dist.ReduceOp
        Reduction operation for the model-parallel all-reduce. ``SUM`` commutes
        with gradient accumulation, so reducing here matches reducing
        ``param.grad`` after accumulation.

    Returns
    -------
    Callable
        Communication hook for ``DistributedDataParallel.register_comm_hook``.
    """

    def comm_hook(
        state: Any, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        """Reduce one gradient bucket over the model- then data-parallel group.

        Parameters
        ----------
        state : Any
            Unused; the groups are captured by the enclosing factory.
        bucket : dist.GradBucket
            Bucket of gradients ready for reduction.

        Returns
        -------
        torch.futures.Future[torch.Tensor]
            Future resolving to the reduced flat buffer.
        """
        buffer = bucket.buffer()
        group = dp_group if dp_group is not None else dist.group.WORLD

        def _reduce_dp(_: Any = None) -> torch.Tensor:
            """Average the bucket over the data-parallel group."""
            # Divide first, as the default DDP hook does, to limit overflow.
            buffer.div_(group.size())
            # ``wait`` runs on the future executor thread, not the backward
            # thread, so it does not stall the rest of the backward pass.
            return (
                dist.all_reduce(buffer, group=group, async_op=True)
                .get_future()
                .wait()[0]
            )

        mp_groups = [mp_group_for(param) for param in bucket.parameters()]

        if mp_groups and all(g is not None and g is mp_groups[0] for g in mp_groups):
            # Every parameter in the bucket shares one model-parallel group, so
            # reduce the flat buffer in a single collective and chain the
            # data-parallel reduction onto its future. The future comes from the
            # process group itself, so it carries the CUDA event the callback
            # needs; ``torch.futures.collect_all`` would drop it and leave the
            # data-parallel all-reduce unordered against this one.
            return (
                dist.all_reduce(buffer, op=mp_op, group=mp_groups[0], async_op=True)
                .get_future()
                .then(_reduce_dp)
            )

        # Mixed or partial groups: reduce per parameter. ``bucket.gradients()``
        # are views into ``buffer``, so these write through. ``wait`` on a NCCL
        # collective orders the current stream against it without blocking the
        # host, which is what the data-parallel all-reduce below then inherits.
        for param, grad in zip(bucket.parameters(), bucket.gradients()):
            mp_group = mp_group_for(param)
            if mp_group is not None:
                dist.all_reduce(grad, op=mp_op, group=mp_group, async_op=True).wait()

        buffer.div_(group.size())
        return (
            dist.all_reduce(buffer, group=group, async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    return comm_hook
