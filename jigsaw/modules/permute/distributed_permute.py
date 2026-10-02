"""Distributed permutation kernels.

The kernels exchange unequal blocks between the ranks: with ``dist.all_to_all``
where the backend supports it, and point-to-point on gloo, which does not.
"""

from typing import Optional

import torch
import torch.distributed as dist

from jigsaw.comm import all_to_all_async
from jigsaw.modules.buffers import get_preallocated_buffer


def _exchange(
    recv_list: list[torch.Tensor],
    send_list: list[torch.Tensor],
    group: dist.ProcessGroup,
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

    Returns
    -------
    list[dist.Work]
        Handles to wait on.
    """
    if dist.get_backend(group) != "gloo":
        return [all_to_all_async(recv_list, send_list, group=group)]

    # gloo's all-to-all needs equally sized tensors, so talk to each peer directly.
    works: list[Optional[dist.Work]] = []
    for peer, data in enumerate(send_list):
        if data.numel() > 0:
            works.append(dist.isend(data, group_dst=peer, group=group))
    for peer, buffer in enumerate(recv_list):
        if buffer.numel() > 0:
            works.append(dist.irecv(buffer, group_src=peer, group=group))
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

    xt_works = _exchange(x_recv_list, x_send_list, model_group)

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

    _wait(_exchange(x_recv_list, x_send_list, model_group))

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

    _wait(_exchange(x_recv_list, x_send_list, model_group))
    cij = torch.concat([cij0, cij1], dim=-2).mT.contiguous()
    return cij
