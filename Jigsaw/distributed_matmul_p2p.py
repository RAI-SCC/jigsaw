import torch
import torch.distributed as dist

from .distributed_matmul import _get_preallocated_buffer


def matrix_transpose_twoway_(x, rank, device, model_group):
    """
    Batched transpose (XT) operation distributed across two processes.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.
    rank : int
        Global rank.
    device : torch.device
        Device to perform computation on.
    model_group : torch.distributed.ProcessGroup, optional
        Process group for an individual model-parallel instance.

    Returns
    -------
    XT: torch.Tensor
        Local transposed tensor.
    """
    mod = rank % 2

    x_0, x_1 = torch.split(x, x.shape[-2] // 2, dim=-2)
    buffer_x = _get_preallocated_buffer(x_1.shape, torch.float32, rank, device)

    if mod == 0:
        send_b_1 = torch.distributed.isend(x_1, dst=rank + 1, group=model_group)
        recv_b_1 = torch.distributed.irecv(buffer_x, src=rank + 1, group=model_group)
        x_0 = x_0.mT
        if len(x_0.shape) == 3:
            x_t = torch.empty(
                x_0.shape[0], x_0.shape[1] * 2, x_0.shape[2], device=device
            )
            x_t[:, : x_0.shape[1]] = x_0
        else:
            x_t = torch.empty(x_0.shape[0] * 2, x_0.shape[1], device=device)
            x_t[: x_0.shape[1]] = x_0

        send_b_1.wait()
        recv_b_1.wait()

    elif mod == 1:
        recv_transpose = torch.distributed.irecv(
            buffer_x, src=rank - 1, group=model_group
        )
        send_transpose = torch.distributed.isend(x_0, dst=rank - 1, group=model_group)
        x_1 = x_1.mT
        if len(x_0.shape) == 3:
            x_t = torch.empty(
                x_1.shape[0], x_1.shape[1] * 2, x_1.shape[2], device=device
            )
            x_t[:, x_1.shape[1] :] = x_1
        else:
            x_t = torch.empty(x_1.shape[1] * 2, x_1.shape[2], device=device)
            x_t[x_1.shape[1] :] = x_1

        send_transpose.wait()
        recv_transpose.wait()

    if mod == 0:
        if len(x_0.shape) == 3:
            x_t[:, x_0.shape[1] :, :] = buffer_x.mT
        else:
            x_t[x_0.shape[1] :, :] = buffer_x.mT
    elif mod == 1:
        if len(x_1.shape) == 3:
            x_t[:, : x_1.shape[1], :] = buffer_x.mT
        else:
            x_t[: x_1.shape[1], :] = buffer_x.mT
    return x_t


def xw_twoway_(xij, wij, rank, device, model_group):
    """
    Batched matrix-matrix (XW) multiplication distributed across two processes.

    Parameters
    ----------
    Xij : torch.Tensor
        Local input matrix.
    Wij : torch.Tensor
        Local weight matrix.
    device : torch.device
        Device to perform computation on.
    rank : int
        Global rank.
    model_group : torch.distributed.ProcessGroup, optional
        Process group for an individual model-parallel instance.

    Returns
    -------
    Cij : torch.Tensor
        Local result.
    """
    mod = rank % 2
    w0, w1 = torch.split(wij, wij.shape[-2] // 2, dim=-2)
    buffer_x = _get_preallocated_buffer(xij.shape, torch.float32, rank, device)
    if mod == 0:
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_w_1 = torch.distributed.isend(xij, dst=rank + 1, group=None)
        recv_w_1 = torch.distributed.irecv(buffer_x, src=rank + 1, group=None)

        cij = torch.matmul(xij, w0)
        send_w_1.wait()
        recv_w_1.wait()
    elif mod == 1:
        recv_w_0 = torch.distributed.irecv(buffer_x, src=rank - 1, group=None)
        send_w_0 = torch.distributed.isend(xij, dst=rank - 1, group=None)

        cij = torch.matmul(xij, w1)
        send_w_0.wait()
        recv_w_0.wait()

    if mod == 0:
        cij = cij + torch.matmul(buffer_x, w1)
    elif mod == 1:
        cij = cij + torch.matmul(buffer_x, w0)
    return cij


def xw_eightway_(xij, wij, rank, device, model_group):
    """
    Batched matrix-matrix (XW) multiplication distributed across eight processes.

    Parameters
    ----------
    Xij : torch.Tensor
        Local input matrix.
    Wij : torch.Tensor
        Local weight matrix.
    rank : int
        Global rank.
    device : torch.device
        Device to perform computation on.
    model_group : torch.distributed.ProcessGroup, optional
        Process group for an individual model-parallel instance.

    Returns
    -------
    Cij : torch.Tensor
        Local result.
    """
    mod = rank % 8

    empty = torch.empty(0, device=device, dtype=xij.dtype)

    if mod == 0:
        x_send_list = [empty, xij, empty, empty, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            _get_preallocated_buffer(xij.shape, xij.dtype, rank, device, 0),
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        ]
        w_send_list = [empty, empty, wij, empty, wij, empty, wij, empty]
        w_recv_list = [
            empty,
            empty,
            _get_preallocated_buffer(wij.shape, xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(wij.shape, xij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(wij.shape, xij.dtype, rank, device, 3),
            empty,
        ]
    elif mod == 1:
        x_send_list = [xij, empty, empty, empty, empty, empty, empty, empty]
        x_recv_list = [
            _get_preallocated_buffer(xij.shape, xij.dtype, rank, device, 0),
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        ]
        w_send_list = [empty, empty, empty, wij, empty, wij, empty, wij]
        w_recv_list = [
            empty,
            empty,
            empty,
            _get_preallocated_buffer(wij.shape, wij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(wij.shape, wij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(wij.shape, wij.dtype, rank, device, 3),
        ]
    elif mod == 2:
        x_send_list = [empty, empty, empty, xij, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            empty,
            empty,
            _get_preallocated_buffer(xij.shape, xij.dtype, rank, device, 0),
            empty,
            empty,
            empty,
            empty,
        ]
        w_send_list = [wij, empty, empty, empty, wij, empty, wij, empty]
        w_recv_list = [
            _get_preallocated_buffer(wij.shape, xij.dtype, rank, device, 1),
            empty,
            empty,
            empty,
            _get_preallocated_buffer(wij.shape, xij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(wij.shape, xij.dtype, rank, device, 3),
            empty,
        ]
    elif mod == 3:
        x_send_list = [empty, empty, xij, empty, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            empty,
            _get_preallocated_buffer(xij.shape, xij.dtype, rank, device, 0),
            empty,
            empty,
            empty,
            empty,
            empty,
        ]
        w_send_list = [empty, wij, empty, empty, empty, wij, empty, wij]
        w_recv_list = [
            empty,
            _get_preallocated_buffer(wij.shape, wij.dtype, rank, device, 1),
            empty,
            empty,
            empty,
            _get_preallocated_buffer(wij.shape, wij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(wij.shape, wij.dtype, rank, device, 3),
        ]
    elif mod == 4:
        x_send_list = [empty, empty, empty, empty, empty, xij, empty, empty]
        x_recv_list = [
            empty,
            empty,
            empty,
            empty,
            empty,
            _get_preallocated_buffer(xij.shape, xij.dtype, rank, device, 0),
            empty,
            empty,
        ]
        w_send_list = [wij, empty, wij, empty, empty, empty, wij, empty]
        w_recv_list = [
            _get_preallocated_buffer(wij.shape, xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(wij.shape, xij.dtype, rank, device, 2),
            empty,
            empty,
            empty,
            _get_preallocated_buffer(wij.shape, xij.dtype, rank, device, 3),
            empty,
        ]
    elif mod == 5:
        x_send_list = [empty, empty, empty, empty, xij, empty, empty, empty]
        x_recv_list = [
            empty,
            empty,
            empty,
            empty,
            _get_preallocated_buffer(xij.shape, xij.dtype, rank, device, 0),
            empty,
            empty,
            empty,
        ]
        w_send_list = [empty, wij, empty, wij, empty, empty, empty, wij]
        w_recv_list = [
            empty,
            _get_preallocated_buffer(wij.shape, wij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(wij.shape, wij.dtype, rank, device, 2),
            empty,
            empty,
            empty,
            _get_preallocated_buffer(wij.shape, wij.dtype, rank, device, 3),
        ]
    elif mod == 6:
        x_send_list = [empty, empty, empty, empty, empty, empty, empty, xij]
        x_recv_list = [
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            _get_preallocated_buffer(xij.shape, xij.dtype, rank, device, 0),
        ]
        w_send_list = [wij, empty, wij, empty, wij, empty, empty, empty]
        w_recv_list = [
            _get_preallocated_buffer(wij.shape, xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(wij.shape, xij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(wij.shape, xij.dtype, rank, device, 3),
            empty,
            empty,
            empty,
        ]
    elif mod == 7:
        x_send_list = [empty, empty, empty, empty, empty, empty, xij, empty]
        x_recv_list = [
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            _get_preallocated_buffer(xij.shape, xij.dtype, rank, device, 0),
            empty,
        ]
        w_send_list = [empty, wij, empty, wij, empty, wij, empty, empty]
        w_recv_list = [
            empty,
            _get_preallocated_buffer(wij.shape, wij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(wij.shape, wij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(wij.shape, wij.dtype, rank, device, 3),
            empty,
            empty,
        ]

    handle_w = dist.all_to_all(
        w_recv_list, w_send_list, group=model_group, async_op=True
    )
    handle_x = dist.all_to_all(
        x_recv_list, x_send_list, group=model_group, async_op=True
    )

    # X0, X1 = torch.split(Xij, Xij.shape[-1]//2, dim=-1) # EXAMPLE SPLITTING CODE

    handle_w.wait()

    if mod == 0:
        ws_stacked = torch.concatenate([wij, w_recv_list[2]], dim=0).contiguous()
        # Cij = torch.matmul(X0, Wij) + torch.matmul(X1, w_recv_list[2])
    elif mod == 1:
        ws_stacked = torch.concatenate(
            [w_recv_list[5], w_recv_list[7]], dim=0
        ).contiguous()
        # Cij = torch.matmul(X0, w_recv_list[5]) + torch.matmul(X1, w_recv_list[7])
    elif mod == 2:
        ws_stacked = torch.concatenate([w_recv_list[0], wij], dim=0).contiguous()
        # Cij = torch.matmul(X0, w_recv_list[0]) + torch.matmul(X1, Wij)
    elif mod == 3:
        ws_stacked = torch.concatenate(
            [w_recv_list[5], w_recv_list[7]], dim=0
        ).contiguous()
        # Cij = torch.matmul(X0, w_recv_list[5]) + torch.matmul(X1, w_recv_list[7])
    elif mod == 4:
        ws_stacked = torch.concatenate(
            [w_recv_list[0], w_recv_list[2]], dim=0
        ).contiguous()
        # Cij = torch.matmul(X0, w_recv_list[0]) + torch.matmul(X1, w_recv_list[2])
    elif mod == 5:
        ws_stacked = torch.concatenate([wij, w_recv_list[7]], dim=0).contiguous()
        # Cij = torch.matmul(X0, Wij) + torch.matmul(X1, w_recv_list[7])
    elif mod == 6:
        ws_stacked = torch.concatenate(
            [w_recv_list[0], w_recv_list[2]], dim=0
        ).contiguous()
        # Cij = torch.matmul(X0, w_recv_list[0]) + torch.matmul(X1, w_recv_list[2])
    elif mod == 7:
        ws_stacked = torch.concatenate([wij, wij], dim=0).contiguous()
        # Cij = torch.matmul(X0, w_recv_list[5]) + torch.matmul(X1, Wij)

    cij = torch.matmul(xij, ws_stacked)
    handle_x.wait()

    if mod == 0:
        x0, x1 = torch.split(
            x_recv_list[1], x_recv_list[1].shape[-1] // 2, dim=-1
        )  # split upon last dimension
        cij = cij + torch.matmul(x0, w_recv_list[4]) + torch.matmul(x1, w_recv_list[6])
    elif mod == 1:
        x0, x1 = torch.split(
            x_recv_list[0], x_recv_list[0].shape[-1] // 2, dim=-1
        )  # split upon last dimension
        cij = cij + torch.matmul(x0, wij) + torch.matmul(x1, w_recv_list[3])
    elif mod == 2:
        x0, x1 = torch.split(
            x_recv_list[3], x_recv_list[3].shape[-1] // 2, dim=-1
        )  # split upon last dimension
        cij = cij + torch.matmul(x0, w_recv_list[4]) + torch.matmul(x1, w_recv_list[6])
    elif mod == 3:
        x0, x1 = torch.split(
            x_recv_list[2], x_recv_list[2].shape[-1] // 2, dim=-1
        )  # split upon last dimension
        cij = cij + torch.matmul(x0, w_recv_list[1]) + torch.matmul(x1, wij)
    elif mod == 4:
        x0, x1 = torch.split(
            x_recv_list[5], x_recv_list[5].shape[-1] // 2, dim=-1
        )  # split upon last dimension
        cij = cij + torch.matmul(x0, wij) + torch.matmul(x1, w_recv_list[6])
    elif mod == 5:
        x0, x1 = torch.split(
            x_recv_list[4], x_recv_list[4].shape[-1] // 2, dim=-1
        )  # split upon last dimension
        cij = cij + torch.matmul(x0, w_recv_list[1]) + torch.matmul(x1, w_recv_list[3])
    elif mod == 6:
        x0, x1 = torch.split(
            x_recv_list[7], x_recv_list[7].shape[-1] // 2, dim=-1
        )  # split upon last dimension
        cij = cij + torch.matmul(x0, w_recv_list[4]) + torch.matmul(x1, wij)
    elif mod == 7:
        x0, x1 = torch.split(
            x_recv_list[6], x_recv_list[6].shape[-1] // 2, dim=-1
        )  # split upon last dimension
        cij = cij + torch.matmul(x0, w_recv_list[1]) + torch.matmul(x1, w_recv_list[3])

    return cij
