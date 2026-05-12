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
            dist.all_reduce(param, group=spatial_group, op=dist.ReduceOp.AVG)

def gather_results(x: torch.Tensor, process_mesh: dist.device_mesh = None):
    """Gather tensors split across arbitrary processes."""
    row_group, col_group = process_mesh["row"], process_mesh["col"]
    n_rows = row_group.size()
    n_cols = col_group.size()
    tensor_list = [
        torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for _ in range(n_rows * n_cols)
    ]
    torch.distributed.all_gather(tensor_list, x, group=process_mesh["model"])
    batch_dims, h, w = x.shape[:-2], x.shape[-2], x.shape[-1]
    ndim = len(batch_dims)
    perm = list(range(2, 2 + ndim)) + [0, 2 + ndim, 1, 2 + ndim + 1]
    result = torch.stack(tensor_list)
    result = (
        result.reshape(n_rows, n_cols, *x.shape)
        .permute(*perm)
        .reshape(*batch_dims, n_rows * h, n_cols * w)
    )
    return result

def gather_1d_results(
    x: torch.Tensor, process_mesh: dist.device_mesh = None, dim: int = -1
):
    """Gather tensor distributed across process mesh rows along any dimension."""
    n_rows = process_mesh["row"].size()
    tensor_list = [
        torch.zeros(x.shape, dtype=x.dtype, device=x.device) for _ in range(n_rows)
    ]
    torch.distributed.all_gather(
        tensor_list, x.contiguous(), group=process_mesh["model"]
    )

    ndim = x.dim()
    if dim < 0:
        dim = ndim + dim
    result = torch.cat(tensor_list, dim=dim)
    return result