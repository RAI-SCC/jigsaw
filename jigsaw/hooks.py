"""Gradient hook definitions."""

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
    """
    Define hook for grad_weight synchronization.

    If async_op=True, the all_reduction will overlap with the backward pass.
    All collective operations will occur before optimizers step.

    Otherwise, all_reduction is blocking.

    ``group`` may be a single process group or a sequence of them. A sequence is
    used when the domain is split along several spatial axes: a SUM all-reduce
    over each axis group in turn is equivalent to one all-reduce over the whole
    spatial grid. Multiple groups are always reduced sequentially (blocking), so
    the in-place ``param.grad`` buffer is never touched by two collectives at
    once.

    Parameters
    ----------
    group: dist.ProcessGroup | Sequence[dist.ProcessGroup]
        Process group(s) the gradients should be synchronized over.
    op: dist.ReduceOp
        Type of reduction.
    async_op: bool
        Flag for whether communication is blocking or not (single group only).

    Returns
    -------
    hook: Callable
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
    """
    Scale the gradient with parallelism.

    Required if we want strict equivalence with non-parallel case when reduction
    in the loss function = 'mean'. Implemented for correctness.

    Another alternative is to use:
    loss = F.loss_function(y, y', reduction='mean') / parallelism

    Parameters
    ----------
    parallelism: int
        Scaling factor for grad.

    Returns
    -------
    hook: Callable
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
