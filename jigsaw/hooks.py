"""Gradient hook definitions."""

from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.nn.parameter import Parameter


def make_grad_sync_hook(
    group: dist.ProcessGroup,
    op: Any = dist.ReduceOp.SUM,
    async_op: bool = True,
) -> Callable:
    """
    Define hook for grad_weight synchronization.

    If async_op=True, the all_reduction will overlap with the backward pass.
    All collective operations will occur before optimizers step.

    Otherwise, all_reduction is blocking.

    Parameters
    ----------
    group: dist.ProcessGroup
        Process group where the weights should be synchronized.
    op: dist.ReduceOp
        Type of reduction.
    async_op: bool
        Flag for whether communication is blocking or not.

    Returns
    -------
    hook: Callable
    """
    handles: list[torch.distributed.Work] = []

    def _finalize() -> None:
        for h in handles:
            h.wait()
        handles.clear()

    def hook(param: Parameter) -> None:
        if param.grad is None:
            return
        if async_op:
            if not handles:
                # first grad of this backward → queue the wait to run when backward finishes
                Variable._execution_engine.queue_callback(_finalize)
            handles.append(
                dist.all_reduce(param.grad, op=op, group=group, async_op=True)
            )
        else:
            dist.all_reduce(param.grad, op=op, group=group, async_op=False)

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
        if param.grad is not None:
            param.grad = param.grad / parallelism

    return hook
