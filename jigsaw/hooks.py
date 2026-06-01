import torch.distributed as dist
from torch.autograd import Variable


def make_grad_sync_hook(group, op=dist.ReduceOp.SUM, async_op=True):
    handles = []

    def _finalize():
        for h in handles:
            h.wait()
        handles.clear()

    def hook(param):
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


def make_grad_scaling_hook(parallelism):
    """
    Scale the gradient with parallelism.

    Required if we want strict equivalence with non-parallel case when reduction
    in the loss function = 'mean'. Implemented for correctness.

    Another alternative is to use:
    loss = F.loss_function(y, y', reduction='mean') / parallelism
    """

    def hook(param):
        param.grad = param.grad / parallelism

    return hook
