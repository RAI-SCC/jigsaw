import torch.distributed as dist


def make_grad_sync_hook(group, op=dist.ReduceOp.SUM, async_op=True):
    """All-reduce weight gradient over domain-parallel processes."""

    def hook(param):
        if async_op:
            handle = dist.all_reduce(param.grad, op=op, group=group, async_op=True)
            return handle
        else:
            dist.all_reduce(param.grad, op=op, group=group, async_op=False)
            return

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
