"""Gradient hooks."""

from collections.abc import Sequence
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from . import parallel


def make_grad_sync_hook(
    group: "dist.ProcessGroup | Sequence[dist.ProcessGroup]",
    op: Any = dist.ReduceOp.SUM,
    async_op: bool = True,
) -> Callable:
    """Create a hook that all-reduces parameter gradients.

    With ``async_op=True`` the all-reduce overlaps with the backward pass and
    completes before the optimizer step. A sequence of groups is reduced
    sequentially and blocking.

    .. warning::
        Prefer :func:`jigsaw.parallel.mark_replicated` with
        :func:`jigsaw.parallel.sync_grad`, which reduce inside the autograd
        graph. This hook is wrong in two common setups:

        * Under ``DistributedDataParallel`` the asynchronous all-reduce races
          DDP's reducer on the same parameter, which then overwrites the grad
          with the un-summed local share; replicas diverge from the first step.
          :func:`jigsaw.parallel.check_ddp` warns about it.
        * Under gradient accumulation the hook reduces the *running*
          ``p.grad``, so earlier microbatches are summed again on every later
          one.

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

    setattr(hook, parallel.HOOK_TAG, True)  # check_ddp warns about it
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
