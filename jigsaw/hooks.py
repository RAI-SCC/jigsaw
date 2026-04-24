import torch.distributed as dist


def make_grad_sync_hook(group):
    """All-reduce gradients over domain-parallel processes"""
    def hook(param):
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)

    return hook