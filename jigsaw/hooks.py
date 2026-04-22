import torch.distributed as dist


def make_grad_sync_hook(group):
    def hook(param):
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)

    return hook


def make_init_param_sync_hook(group):
    def hook(param):
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)

    return hook
