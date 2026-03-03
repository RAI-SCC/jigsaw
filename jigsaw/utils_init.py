import torch
import torch.distributed as dist

@torch.no_grad()
def average_weights(model: torch.nn.Module, spatial_group: dist.ProcessGroup, bias_only: bool = False):
    for name, param in model.named_parameters():
        if bias_only and not 'bias' in name:
            continue
        else:
            param.requires_grad_(False)
            # dist.all_reduce(param, group=spatial_group, op=dist.ReduceOp.AVG)
            param.requires_grad_(True)
