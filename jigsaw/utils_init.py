import torch
import torch.distributed as dist


@torch.no_grad()
def average_weights(params: list[torch.Tensor], spatial_group: dist.ProcessGroup):
    """
    Average parameters of an entire module upon initialization.

    params: list[torch.Tensor]
        list of tensors that need to be averaged
    spatial_group: dist.ProcessGroup
        process group that the parameters should be averaged over.
    """
    for param in params:
        if param is not None:
            dist.all_reduce(param, group=spatial_group, op=dist.ReduceOp.AVG)


@torch.no_grad()
def average_weights_module(
    model: torch.nn.Module, spatial_group: dist.ProcessGroup, bias_only: bool = False
):
    """
    Average model weights of an entire module upon initialization.

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
