"""Helpers for averaging and gathering distributed tensors."""

import torch
import torch.distributed as dist

__all__ = [
    "average_weights",
    "average_weights_module",
    "gather_results",
    "gather_results_as_list",
    "gather_1d_results",
    "gather_1d_results_list",
]


@torch.no_grad()
def average_weights(
    params: list[torch.Tensor], spatial_group: dist.ProcessGroup
) -> None:
    """Average tensors in place over a process group.

    Parameters
    ----------
    params : list[torch.Tensor]
        Tensors to average.
    spatial_group : dist.ProcessGroup
        Process group to average over.
    """
    for param in params:
        if param is not None:
            dist.all_reduce(param, group=spatial_group, op=dist.ReduceOp.AVG)


@torch.no_grad()
def average_weights_module(
    model: torch.nn.Module, spatial_group: dist.ProcessGroup, bias_only: bool = False
) -> None:
    """Average the parameters of a module in place over a process group.

    Parameters
    ----------
    model : torch.nn.Module
        Module whose parameters are averaged.
    spatial_group : dist.ProcessGroup
        Process group to average over.
    bias_only : bool
        Average only parameters whose name contains ``"bias"``.
    """
    for name, param in model.named_parameters():
        if bias_only and "bias" not in name:
            continue
        else:
            dist.all_reduce(param, group=spatial_group, op=dist.ReduceOp.AVG)


@torch.no_grad()
def gather_results(
    x: torch.Tensor,
    process_mesh_dict: dict[str, dist.ProcessGroup],
    pm_key_row: str = "row",
    pm_key_col: str = "col",
    pm_key_model: str = "model",
) -> torch.Tensor:
    """Gather a 2D-sharded tensor into the full tensor.

    Parameters
    ----------
    x : torch.Tensor
        Local shard, sharded along the last two dimensions.
    process_mesh_dict : dict[str, dist.ProcessGroup]
        Process groups of the mesh.
    pm_key_row : str
        Key of the row group.
    pm_key_col : str
        Key of the column group.
    pm_key_model : str
        Key of the model group.

    Returns
    -------
    torch.Tensor
        Full tensor.
    """
    row_group, col_group = process_mesh_dict[pm_key_row], process_mesh_dict[pm_key_col]
    n_rows = row_group.size()
    n_cols = col_group.size()
    tensor_list = [
        torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for _ in range(n_rows * n_cols)
    ]
    torch.distributed.all_gather(tensor_list, x, group=process_mesh_dict[pm_key_model])
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


@torch.no_grad()
def gather_results_as_list(
    x: torch.Tensor,
    process_mesh: dict[str, dist.ProcessGroup] | None = None,
    pm_key_row: str = "row",
    pm_key_col: str = "col",
    pm_key_model: str = "model",
) -> list[torch.Tensor]:
    """Gather shards of a 2D-sharded tensor into a list.

    Parameters
    ----------
    x : torch.Tensor
        Local shard.
    process_mesh : dict[str, dist.ProcessGroup], optional
        Process groups of the mesh.
    pm_key_row : str
        Key of the row group.
    pm_key_col : str
        Key of the column group.
    pm_key_model : str
        Key of the model group.

    Returns
    -------
    list[torch.Tensor]
        Shards ordered by model rank.
    """
    assert process_mesh is not None
    row_group, col_group = process_mesh[pm_key_row], process_mesh[pm_key_col]
    n_rows = row_group.size()
    n_cols = col_group.size()
    tensor_list = [
        torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for _ in range(n_rows * n_cols)
    ]
    torch.distributed.all_gather(tensor_list, x, group=process_mesh[pm_key_model])
    return tensor_list


@torch.no_grad()
def gather_1d_results(
    x: torch.Tensor,
    process_mesh_dict: dict[str, dist.ProcessGroup],
    dim: int = -1,
    pm_key: str = "model",
) -> torch.Tensor:
    """Gather a tensor sharded along one dimension and concatenate the shards.

    Parameters
    ----------
    x : torch.Tensor
        Local shard.
    process_mesh_dict : dict[str, dist.ProcessGroup]
        Process groups of the mesh.
    dim : int
        Dimension to concatenate along.
    pm_key : str
        Key of the group to gather over.

    Returns
    -------
    torch.Tensor
        Concatenated tensor.
    """
    n_rows = process_mesh_dict[pm_key].size()
    tensor_list = [
        torch.zeros(x.shape, dtype=x.dtype, device=x.device) for _ in range(n_rows)
    ]
    torch.distributed.all_gather(
        tensor_list, x.contiguous(), group=process_mesh_dict[pm_key]
    )

    ndim = x.dim()
    if dim < 0:
        dim = ndim + dim
    result = torch.cat(tensor_list, dim=dim)
    return result


@torch.no_grad()
def gather_1d_results_list(
    x: torch.Tensor,
    process_mesh_dict: dict[str, dist.ProcessGroup],
    pm_key: str = "model",
) -> list[torch.Tensor]:
    """Gather a tensor sharded along one dimension into a list.

    Parameters
    ----------
    x : torch.Tensor
        Local shard.
    process_mesh_dict : dict[str, dist.ProcessGroup]
        Process groups of the mesh.
    pm_key : str
        Key of the group to gather over.

    Returns
    -------
    list[torch.Tensor]
        Shards ordered by rank.
    """
    n_rows = process_mesh_dict[pm_key].size()
    tensor_list = [
        torch.zeros(x.shape, dtype=x.dtype, device=x.device) for _ in range(n_rows)
    ]
    torch.distributed.all_gather(
        tensor_list, x.contiguous(), group=process_mesh_dict[pm_key]
    )
    return tensor_list
