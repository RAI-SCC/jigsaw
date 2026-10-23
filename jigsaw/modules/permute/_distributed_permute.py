"""Distributed permutation kernels."""

from collections.abc import Callable

import torch
import torch.distributed as dist

from jigsaw._core.buffers import get_preallocated_buffer
from jigsaw._core.comm import CommOp, all_to_all_async, resolve_comm_op
from jigsaw.modules.convolution._common.channel_comm import gather_full_channels


def _exchange(
    recv_list: list[torch.Tensor],
    send_list: list[torch.Tensor],
    group: dist.ProcessGroup,
    comm: CommOp,
) -> list[dist.Work]:
    """Start the exchange of one tensor per peer and return its work handles.

    Parameters
    ----------
    recv_list : list[torch.Tensor]
        Buffers receiving the data of each rank, empty for no data.
    send_list : list[torch.Tensor]
        Data sent to each rank, empty for no data.
    group : dist.ProcessGroup
        Process group of the exchange.
    comm : CommOp
        Resolved transport.

    Returns
    -------
    list[dist.Work]
        Handles to wait on.
    """
    if comm is CommOp.ALL_TO_ALL:
        return [all_to_all_async(recv_list, send_list, group=group)]

    # Talk to each peer directly; empty entries carry no data.
    sends = [(peer, t) for peer, t in enumerate(send_list) if t.numel()]
    recvs = [(peer, t) for peer, t in enumerate(recv_list) if t.numel()]
    if comm is CommOp.BATCH_ISEND_IRECV:
        ops = [dist.P2POp(dist.isend, t, group_peer=p, group=group) for p, t in sends]
        ops += [dist.P2POp(dist.irecv, t, group_peer=p, group=group) for p, t in recvs]
        return dist.batch_isend_irecv(ops) if ops else []
    works = [dist.isend(t, group_dst=p, group=group) for p, t in sends]
    works += [dist.irecv(t, group_src=p, group=group) for p, t in recvs]
    return [work for work in works if work is not None]


def _wait(works: list[dist.Work]) -> None:
    """Block until every exchange in ``works`` has completed."""
    for work in works:
        work.wait()


def x_transpose_twoway(
    x: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
    comm: CommOp,
) -> torch.Tensor:
    """Transpose a batched matrix distributed across two processes.

    Parameters
    ----------
    x : torch.Tensor
        Local input.
    local_rank : int
        Rank within the model group.
    device : torch.device
        Device of the computation.
    model_group : dist.ProcessGroup
        Process group of one model-parallel instance.
    comm : CommOp
        Resolved transport.

    Returns
    -------
    torch.Tensor
        Local transposed output.
    """
    mod = local_rank % 2

    x0, x1 = torch.split(x, x.shape[-2] // 2, dim=-2)
    x0, x1 = x0.contiguous(), x1.contiguous()
    buffer_x = get_preallocated_buffer(x1.shape, x1.dtype, local_rank, device)
    empty = torch.empty(0, device=device, dtype=x.dtype)

    match mod:
        case 0:
            x_send_list = [empty, x1]
            x_recv_list = [empty, buffer_x]
        case 1:
            x_send_list = [x0, empty]
            x_recv_list = [buffer_x, empty]

    xt_works = _exchange(x_recv_list, x_send_list, model_group, comm)

    match mod:
        case 0:
            x0 = x0.mT
            x_transpose = torch.empty(
                *x0.shape[:-2],
                x0.shape[-2] * 2,
                x0.shape[-1],
                dtype=x.dtype,
                device=device,
            )
            x_transpose[..., : x0.shape[-2], :] = x0
            _wait(xt_works)
            x_transpose[..., x0.shape[-2] :, :] = buffer_x.mT
        case 1:
            x1 = x1.mT
            x_transpose = torch.empty(
                *x1.shape[:-2],
                x1.shape[-2] * 2,
                x1.shape[-1],
                dtype=x.dtype,
                device=device,
            )
            x_transpose[..., x1.shape[-2] :, :] = x1
            _wait(xt_works)
            x_transpose[..., : x1.shape[-2], :] = buffer_x.mT
    return x_transpose


def x_transpose_fourway(
    x: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
    comm: CommOp,
) -> torch.Tensor:
    """Transpose a batched matrix distributed across four processes.

    Parameters
    ----------
    x : torch.Tensor
        Local input.
    local_rank : int
        Rank within the model group.
    device : torch.device
        Device of the computation.
    model_group : dist.ProcessGroup
        Process group of one model-parallel instance.
    comm : CommOp
        Resolved transport.

    Returns
    -------
    torch.Tensor
        Local transposed output.
    """
    mod = local_rank % 4
    empty = torch.empty(0, device=device, dtype=x.dtype)

    x = x.contiguous()
    match mod:
        case 0:
            x_send_list = [empty, empty, empty, empty]
            x_recv_list = [empty, empty, empty, empty]
        case 1:
            x_send_list = [empty, empty, x, empty]
            x_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(x.shape, x.dtype, local_rank, device, 0),
                empty,
            ]
        case 2:
            x_send_list = [empty, x, empty, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(x.shape, x.dtype, local_rank, device, 0),
                empty,
                empty,
            ]
        case 3:
            x_send_list = [empty, empty, empty, empty]
            x_recv_list = [empty, empty, empty, empty]

    _wait(_exchange(x_recv_list, x_send_list, model_group, comm))

    match mod:
        case 0:
            return x.mT
        case 1:
            return x_recv_list[2].mT
        case 2:
            return x_recv_list[1].mT
        case 3:
            return x.mT
        case _:
            raise ValueError(f"Invalid mod value: {mod}")


def x_transpose_eightway(
    xij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
    comm: CommOp,
) -> torch.Tensor:
    """Transpose a batched matrix distributed across eight processes.

    Parameters
    ----------
    xij : torch.Tensor
        Local input.
    local_rank : int
        Rank within the model group.
    device : torch.device
        Device of the computation.
    model_group : dist.ProcessGroup
        Process group of one model-parallel instance.
    comm : CommOp
        Resolved transport.

    Returns
    -------
    torch.Tensor
        Local transposed output.
    """
    mod = local_rank % 8
    empty = torch.empty(0, device=device, dtype=xij.dtype)

    # split upon last dimension
    x0, x1 = torch.split(xij, xij.shape[-1] // 2, dim=-1)
    x0, x1 = x0.contiguous(), x1.contiguous()
    cij0 = get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, index=0)
    cij1 = get_preallocated_buffer(x1.shape, xij.dtype, local_rank, device, index=0)

    match mod:
        case 0:
            x_send_list = [empty, empty, x1, empty, empty, empty, empty, empty]
            cij0.copy_(x0)
            x_recv_list = [empty, empty, cij1, empty, empty, empty, empty, empty]
        case 1:
            x_send_list = [empty, empty, empty, empty, x0, empty, x1, empty]
            x_recv_list = [empty, empty, empty, empty, cij0, empty, cij1, empty]
        case 2:
            cij1.copy_(x1)
            x_send_list = [x0, empty, empty, empty, empty, empty, empty, empty]
            x_recv_list = [cij0, empty, empty, empty, empty, empty, empty, empty]
        case 3:
            x_send_list = [empty, empty, empty, empty, x0, empty, x1, empty]
            x_recv_list = [empty, empty, empty, empty, cij0, empty, cij1, empty]
        case 4:
            x_send_list = [empty, x0, empty, x1, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                cij0,
                empty,
                cij1,
                empty,
                empty,
                empty,
                empty,
            ]
        case 5:
            x_send_list = [empty, empty, empty, empty, empty, empty, empty, x1]
            cij0.copy_(x0)
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                cij1,
            ]
        case 6:
            x_send_list = [empty, x0, empty, x1, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                cij0,
                empty,
                cij1,
                empty,
                empty,
                empty,
                empty,
            ]
        case 7:
            x_send_list = [empty, empty, empty, empty, empty, x0, empty, empty]
            cij1.copy_(x1)
            x_recv_list = [empty, empty, empty, empty, empty, cij0, empty, empty]

    _wait(_exchange(x_recv_list, x_send_list, model_group, comm))
    cij = torch.concat([cij0, cij1], dim=-2).mT.contiguous()
    return cij


# Fast path for `distributed_permute`: swapping the last two axes on these mesh shapes
# uses direct peer exchange instead of an all-gather.
_SWAP_KERNELS: dict[tuple[int, int], Callable[..., torch.Tensor]] = {
    (1, 2): x_transpose_twoway,
    (2, 2): x_transpose_fourway,
    (4, 2): x_transpose_eightway,
}


def distributed_permute(
    x: torch.Tensor,
    dims: tuple[int, ...],
    row_dim: int,
    col_dim: int,
    local_rank: int,
    device: torch.device,
    mesh: dict[str, dist.ProcessGroup],
    comm: CommOp | str = CommOp.AUTO,
) -> torch.Tensor:
    """Permute ``x`` by ``dims``, resharding ``row_dim``/``col_dim`` if ``dims`` swaps them.

    ``mesh["row"]`` shards axis position ``row_dim`` and ``mesh["col"]`` shards
    ``col_dim``, in both the input and the output. Every other axis must not be
    sharded over ``mesh["row"]`` or ``mesh["col"]``; sharding over another mesh
    axis (e.g. data-parallel) is fine.

    Parameters
    ----------
    x : torch.Tensor
        Local input.
    dims : tuple[int, ...]
        Target axis order, as in ``torch.permute``. Must feed ``row_dim`` and ``col_dim``
        from ``{row_dim, col_dim}``, either unchanged or swapped with each other.
    row_dim : int
        Axis of ``x`` sharded by ``mesh["row"]``.
    col_dim : int
        Axis of ``x`` sharded by ``mesh["col"]``.
    local_rank : int
        Rank within the model group. Only used by the fast-path kernels.
    device : torch.device
        Device of the computation.
    mesh : dict[str, dist.ProcessGroup]
        Needs ``"row"`` and ``"col"``; the fast path also needs ``"model"``.
    comm : CommOp or str
        Transport of the fast-path kernels. ``"auto"`` picks all-to-all on CUDA
        and ``batch_isend_irecv`` on CPU.

    Returns
    -------
    torch.Tensor
        Local output, permuted and resharded.
    """
    comm = resolve_comm_op(comm, device)
    n = x.ndim
    if sorted(dims) != list(range(n)):
        raise ValueError(f"dims {dims} is not a permutation of range({n})")
    if row_dim == col_dim or not 0 <= row_dim < n or not 0 <= col_dim < n:
        raise ValueError(
            f"row_dim {row_dim} and col_dim {col_dim} must be distinct axes of a "
            f"{n}-d tensor"
        )
    if {dims[row_dim], dims[col_dim]} != {row_dim, col_dim}:
        raise ValueError(
            f"dims {dims} moves an axis other than row_dim={row_dim}/col_dim={col_dim} "
            "into one of those two positions (or moves one of them out); "
            "distributed_permute only knows how to reshard those two axes; the others "
            "may be sharded by a mesh it has no knowledge of"
        )

    if dims[row_dim] == row_dim:
        # row_dim and col_dim keep their roles; every axis is just repositioned.
        return x.permute(*dims).contiguous()

    row_group, col_group = mesh["row"], mesh["col"]
    n_rows, n_cols = row_group.size(), col_group.size()

    if row_dim == n - 2 and col_dim == n - 1 and (n_rows, n_cols) in _SWAP_KERNELS:
        # Reorder the batch axes first (free); the kernel below leaves them untouched
        # and swaps the trailing two, so the result already matches `dims`.
        x = x.permute(*dims[:-2], row_dim, col_dim).contiguous()
        return _SWAP_KERNELS[(n_rows, n_cols)](
            x, local_rank, device, mesh["model"], comm
        )

    # General path: reconstruct each tracked axis in full over the group that currently
    # shards it, then narrow it into the other axis's role.
    x = gather_full_channels(x.contiguous(), col_group, col_dim)
    x = gather_full_channels(x.contiguous(), row_group, row_dim)

    row_rank, col_rank = row_group.rank(), col_group.rank()

    size_col = x.shape[col_dim]
    if size_col % n_rows:
        raise ValueError(
            f"axis {col_dim} (size {size_col}) doesn't divide evenly over "
            f"{n_rows} row ranks"
        )
    chunk_col = size_col // n_rows
    x = x.narrow(col_dim, row_rank * chunk_col, chunk_col).contiguous()

    size_row = x.shape[row_dim]
    if size_row % n_cols:
        raise ValueError(
            f"axis {row_dim} (size {size_row}) doesn't divide evenly over "
            f"{n_cols} col ranks"
        )
    chunk_row = size_row // n_cols
    x = x.narrow(row_dim, col_rank * chunk_row, chunk_row).contiguous()

    return x.permute(*dims).contiguous()
