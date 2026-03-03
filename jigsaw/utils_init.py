import torch
import torch.distributed as dist


@torch.no_grad()
def average_weights(
    model: torch.nn.Module, spatial_group: dist.ProcessGroup, bias_only: bool = False
):
    """
    Average model weights upon initialization.

    model: torch.nn.Module
    spatial_group: dist.ProcessGroup
        process group that the parameters should be averaged over.
    bias_only: bool
        boolean flag for whether the terms should only be averaged over the bias terms.
    """
    for name, param in model.named_parameters():
        if bias_only and "bias" not in name:
            continue
        else:
            param.requires_grad_(False)
            dist.all_reduce(param, group=spatial_group, op=dist.ReduceOp.AVG)
            param.requires_grad_(True)
