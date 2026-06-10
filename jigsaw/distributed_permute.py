"""Implementation of distributed permutation functions."""

import torch
import torch.distributed as dist

from .distributed_matmul import _get_preallocated_buffer


def x_transpose_twoway(
    x: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Batched transpose (x_transpose) operation distributed across two processes.

    Parameters.
    ----------
    x : torch.Tensor
        The input tensor.
    local_rank : int
        Local model rank.
    device : torch.device
        Device to perform computation on.
    model_group : dist.ProcessGroup, optional
        Process group for an individual model-parallel instance.

    Returns
    -------
    x_transpose: torch.Tensor
        Local transposed tensor.
    """
    mod = local_rank % 2

    x0, x1 = torch.split(x, x.shape[-2] // 2, dim=-2)
    x0, x1 = x0.contiguous(), x1.contiguous()
    buffer_x = _get_preallocated_buffer(x1.shape, x1.dtype, local_rank, device)
    empty = torch.empty(0, device=device, dtype=x.dtype)

    match mod:
        case 0:
            x_send_list = [empty, x1]
            x_recv_list = [empty, buffer_x]
        case 1:
            x_send_list = [x0, empty]
            x_recv_list = [buffer_x, empty]

    xt_handle = dist.all_to_all(
        x_recv_list, x_send_list, group=model_group, async_op=True
    )

    match mod:
        case 0:
            x0 = x0.mT

            x_shape = x0.shape
            if len(x_shape) >= 3:
                x_transpose = torch.empty(
                    *x_shape[:-2], x0.shape[-2] * 2, x0.shape[-1], device=device
                )
                x_transpose[..., : x0.shape[-2], :] = x0
            else:
                x_transpose = torch.empty(x0.shape[0] * 2, x0.shape[1], device=device)
                x_transpose[: x0.shape[1]] = x0
            xt_handle.wait()
            if len(x_shape) >= 3:
                x_transpose[..., x0.shape[-2] :, :] = buffer_x.mT
            else:
                x_transpose[x0.shape[1] :, :] = buffer_x.mT
        case 1:
            x1 = x1.mT
            x_shape = x1.shape
            if len(x_shape) >= 3:
                x_transpose = torch.empty(
                    *x_shape[:-2], x1.shape[-2] * 2, x1.shape[-1], device=device
                )
                x_transpose[..., x1.shape[-2] :, :] = x1
            else:
                x_transpose = torch.empty(x1.shape[1] * 2, x1.shape[2], device=device)
                x_transpose[x1.shape[1] :] = x1
            xt_handle.wait()
            if len(x_shape) >= 3:
                x_transpose[..., : x1.shape[-2], :] = buffer_x.mT
            else:
                x_transpose[: x1.shape[1], :] = buffer_x.mT
    return x_transpose


def x_transpose_fourway(
    x: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Batched transpose (x_transpose) operation distributed across four processes.

    Parameters.
    ----------
    X : torch.Tensor
        Local input tensor to be transposed.
    local_rank : int
        Local model rank.
    device : torch.device
        Device to perform computation on.

    Returns
    -------
    X : torch.Tensor
        The transposed tensor.
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
                _get_preallocated_buffer(x.shape, x.dtype, local_rank, device, 0),
                empty,
            ]
        case 2:
            x_send_list = [empty, x, empty, empty]
            x_recv_list = [
                empty,
                _get_preallocated_buffer(x.shape, x.dtype, local_rank, device, 0),
                empty,
                empty,
            ]
        case 3:
            x_send_list = [empty, empty, empty, empty]
            x_recv_list = [empty, empty, empty, empty]

    dist.all_to_all(x_recv_list, x_send_list, group=model_group)

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
    """
    Batched transpose (x_transpose) operation distributed across eight processes.

    Parameters.
    ----------
    X : torch.Tensor
        Local input tensor to be transposed.
    local_rank : int
        Local model rank.
    device : torch.device
        Device to perform computation on.

    Returns
    -------
    X : torch.Tensor
        The transposed tensor.
    """
    mod = local_rank % 8
    empty = torch.empty(0, device=device, dtype=torch.float32)

    # split upon last dimension
    x0, x1 = torch.split(xij, xij.shape[-1] // 2, dim=-1)
    x0, x1 = x0.contiguous(), x1.contiguous()
    cij0 = _get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, index=0)
    cij1 = _get_preallocated_buffer(x1.shape, xij.dtype, local_rank, device, index=0)

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

    dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=False)
    cij = torch.concat([cij0, cij1], dim=-2).mT.contiguous()
    return cij
