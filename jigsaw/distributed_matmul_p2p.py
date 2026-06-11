"""Distributed matrix multiplication implementation with Point-to-point operations."""

import torch
import torch.distributed as dist

from .distributed_matmul import _get_preallocated_buffer


def fused_xw_xtw_twoway(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    input: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused calculation to calculate grad_input and grad_weight in backward pass of a linear layer.

    Parameters
    ----------
    grad_output: torch.Tensor
        Output with respect to the gradient.
    weight: torch.Tensor
        Model weights.
    input: torch.Tensor
        Model input.
    local_rank: int
        Local model rank.
    device: torch.device
        Device upon which computations occur.
    model_group: dist.ProcessGroup
        Process group of the domain- and tensor-parallel group of this linear layer.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
    Gradient with respect to the input and gradient of the weights.
    """
    mod = local_rank % 2
    w0, w1 = torch.split(weight, weight.shape[-2] // 2, dim=-2)
    buffer_x = _get_preallocated_buffer(
        grad_output.shape, grad_output.dtype, local_rank, device
    )
    match mod:
        case 0:
            send = dist.isend(grad_output, group_dst=local_rank + 1, group=model_group)
            recv = dist.irecv(buffer_x, group_src=local_rank + 1, group=model_group)
        case 1:
            recv = dist.irecv(buffer_x, group_src=local_rank - 1, group=model_group)
            send = dist.isend(grad_output, group_dst=local_rank - 1, group=model_group)

    match mod:
        case 0:
            grad_input = torch.matmul(grad_output, w0)
            grad_weight_top = torch.matmul(grad_output.mT, input)
            send.wait()
            recv.wait()
            grad_input.add_(torch.matmul(buffer_x, w1))
            grad_weight_bottom = torch.matmul(buffer_x.mT, input)
        case 1:
            grad_input = torch.matmul(grad_output, w1)
            grad_weight_bottom = torch.matmul(grad_output.mT, input)
            recv.wait()
            send.wait()
            grad_input.add_(torch.matmul(buffer_x, w0))
            grad_weight_top = torch.matmul(buffer_x.mT, input)

    grad_weight = torch.concat([grad_weight_top, grad_weight_bottom], dim=-2)
    return grad_input, grad_weight


def xw_transpose_twoway(
    xij: torch.tensor,
    wij: torch.tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.tensor:
    """
    Batched matrix-matrix (XWT) multiplication distributed across two processes.

    Parameters
    ----------
    xij : torch.tensor
        Local input matrix.
    wij : torch.tensor
        Local weight matrix.
    device : torch.device
        Device to perform computation on.
    local_rank : int
        Local local_rank.
    model_group : dist.ProcessGroup, optional
        Process group for an individual model-parallel instance.

    Returns
    -------
    cij : torch.tensor
        Local result.
    """
    mod = local_rank % 2

    w0, w1 = torch.split(wij, wij.shape[-2] // 2, dim=-2)
    buffer_xw = (
        torch.zeros((*xij.shape[:-1], w0.shape[-2])).to(torch.float32).to(device)
    )
    match mod:
        case 0:
            xw = torch.matmul(xij, w1.mT)
            send = dist.isend(xw, group_dst=local_rank + 1, group=model_group)
            recv = dist.irecv(buffer_xw, group_src=local_rank + 1, group=model_group)
        case 1:
            xw = torch.matmul(xij, w0.mT)
            recv = dist.irecv(buffer_xw, group_src=local_rank - 1, group=model_group)
            send = dist.isend(xw, group_dst=local_rank - 1, group=model_group)

    match mod:
        case 0:
            cij = torch.matmul(xij, w0.mT)
            send.wait()
            recv.wait()
            cij = cij + buffer_xw
        case 1:
            cij = torch.matmul(xij, w1.mT)
            recv.wait()
            send.wait()
            cij = cij + buffer_xw

    return cij


def xw_twoway(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Batched matrix-matrix (XW) multiplication distributed across two processes.

    Parameters
    ----------
    xij : torch.Tensor
        Local input matrix.
    wij : torch.Tensor
        Local weight matrix.
    device : torch.device
        Device to perform computation on.
    local_rank : int
        Global local_rank.
    model_group : dist.ProcessGroup, optional
        Process group for an individual model-parallel instance.

    Returns
    -------
    cij : torch.Tensor
        Local result.
    """
    mod = local_rank % 2
    w0, w1 = torch.split(wij, wij.shape[-2] // 2, dim=-2)
    buffer_x = _get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device)

    match mod:
        case 0:
            send = dist.isend(xij, group_dst=local_rank + 1, group=model_group)
            recv = dist.irecv(buffer_x, group_src=local_rank + 1, group=model_group)
        case 1:
            recv = dist.irecv(buffer_x, group_src=local_rank - 1, group=model_group)
            send = dist.isend(xij, group_dst=local_rank - 1, group=model_group)

    match mod:
        case 0:
            cij = torch.matmul(xij, w0)
            send.wait()
            recv.wait()
            cij = cij + torch.matmul(buffer_x, w1)
        case 1:
            cij = torch.matmul(xij, w1)
            recv.wait()
            send.wait()
            cij = cij + torch.matmul(buffer_x, w0)

    return cij


def x_transpose_w_twoway(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Batched matrix-matrix (x_transposeW) multiplication distributed across two processes.

    Parameters
    ----------
    xij : torch.Tensor
        Local input matrix.
    wij : torch.Tensor
        Local weight matrix.
    device : torch.device
        The device on which to perform the operations.
    local_rank : int
        Local model rank.
    model_group : dist.ProcessGroup, optional
        Process group for an individual model-parallel instance.

    Returns
    -------
    cij : torch.Tensor
        Local result.
    """
    mod = local_rank % 2

    buffer_x = _get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device)

    match mod:
        case 0:
            send = dist.isend(xij, group_dst=local_rank + 1, group=model_group)
            recv = dist.irecv(buffer_x, group_src=local_rank + 1, group=model_group)
        case 1:
            recv = dist.irecv(buffer_x, group_src=local_rank - 1, group=model_group)
            send = dist.isend(xij, group_dst=local_rank - 1, group=model_group)

    match mod:
        case 0:
            cij_top = torch.matmul(xij.mT, wij)
            send.wait()
            recv.wait()
            cij_bottom = torch.matmul(buffer_x.mT, wij)
        case 1:
            cij_bottom = torch.matmul(xij.mT, wij)
            recv.wait()
            send.wait()
            cij_top = torch.matmul(buffer_x.mT, wij)

    cij = torch.concat([cij_top, cij_bottom], dim=-2)

    return cij


def xw_transpose_fourway(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Batched matrix-matrix (XWT) multiplication distributed across four processes.

    Parameters
    ----------
    xij : torch.Tensor
        Local input matrix.
    wij : torch.Tensor
        Local weight matrix.
    local_rank : int
        Local model rank.
    device : torch.device
        Device to perform computation on.
    model_group : dist.ProcessGroup, optional
        Process group for an individual model-parallel instance.

    Returns
    -------
    cij : torch.Tensor
        Local result.
    """
    mod = local_rank % 4
    buffer_x = _get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device)
    match mod:
        case 0:
            w_buffer = _get_preallocated_buffer(
                wij.shape, wij.dtype, local_rank, device
            )
            send_x = dist.isend(xij, dst=local_rank + 1, group=model_group)
            recv_x = dist.irecv(buffer_x, src=local_rank + 1, group=model_group)
            send_w_2 = dist.isend(wij, dst=local_rank + 2, group=model_group)
            recv_w_1 = dist.irecv(w_buffer, src=local_rank + 1, group=model_group)
            # send w to rank + 2, receive w from rank + 1

        case 1:
            w_buffer_2 = _get_preallocated_buffer(
                wij.shape, wij.dtype, local_rank, device
            )
            w_buffer_3 = _get_preallocated_buffer(
                wij.shape, wij.dtype, local_rank, device
            )
            recv_x = dist.irecv(buffer_x, src=local_rank - 1, group=model_group)
            send_x = dist.isend(xij, dst=local_rank - 1, group=model_group)
            send_w_0 = dist.isend(wij, dst=local_rank + 1, group=model_group)
            recv_w_3 = dist.irecv(w_buffer_3, src=local_rank + 1, group=model_group)
            send_w_2 = dist.isend(wij, dst=local_rank + 2, group=model_group)
            # send w to 0 & 2, receive w from 2 & 3
            recv_w_2 = dist.irecv(w_buffer_2, src=local_rank - 1, group=model_group)
        case 2:
            send_x = dist.isend(xij, dst=local_rank + 1, group=model_group)
            recv_x = dist.irecv(buffer_x, src=local_rank + 1, group=model_group)
            w_buffer_0 = _get_preallocated_buffer(
                wij.shape, wij.dtype, local_rank, device
            )
            w_buffer_1 = _get_preallocated_buffer(
                wij.shape, wij.dtype, local_rank, device
            )
            recv_w_0 = dist.irecv(w_buffer_0, src=local_rank - 2, group=model_group)
            recv_w_1 = dist.irecv(w_buffer_1, src=local_rank - 1, group=model_group)
            send_w_1 = dist.isend(wij, dst=local_rank - 1, group=model_group)
            send_w_3 = dist.isend(wij, dst=local_rank + 1, group=model_group)
            # send w to 1 & 3, receive w from 0 & 1
        case 3:
            recv_x = dist.irecv(buffer_x, src=local_rank - 1, group=model_group)
            send_x = dist.isend(xij, dst=local_rank - 1, group=model_group)
            w_buffer_2 = _get_preallocated_buffer(
                wij.shape, wij.dtype, local_rank, device
            )
            # send w to 1, receive w from 2
            send_w_1 = dist.isend(wij, dst=local_rank - 2, group=model_group)
            recv_w_1 = dist.irecv(w_buffer_2, src=local_rank - 2, group=model_group)

    match mod:
        case 0:
            send_w_2.wait()
            cij = torch.matmul(xij, wij.mT)
        case 1:
            recv_w_3.wait()
            cij = torch.matmul(xij, w_buffer_3.mT)
        case 2:
            recv_w_0.wait()
            cij = torch.matmul(xij, w_buffer_0.mT)
        case 3:
            send_w_1.wait()
            cij = torch.matmul(xij, wij.mT)

    match mod:
        case 0:
            recv_x.wait()
            send_x.wait()
            recv_w_1.wait()
            cij = cij + torch.matmul(buffer_x, w_buffer_1.mT)
        case 1:
            send_x.wait()
            recv_x.wait()
            send_w_0.wait()
            recv_w_2.wait()
            send_w_1.wait()
            cij = cij + torch.matmul(buffer_x, w_buffer_2.mT)
        case 2:
            recv_x.wait()
            send_x.wait()
            send_w_1.wait()
            recv_w_1.wait()
            send_w_3.wait()
            cij = cij + torch.matmul(buffer_x, w_buffer_1.mT)
        case 3:
            send_x.wait()
            recv_x.wait()
            recv_w_2.wait()
            cij = cij + torch.matmul(buffer_x, w_buffer_2.mT)

    return cij


def xw_fourway(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Batched matrix-matrix (XW) multiplication distributed across four processes.

    Parameters
    ----------
    xij : torch.Tensor
        Local input matrix.
    wij : torch.Tensor
        Local weight matrix.
    local_rank : int
        Local model rank.
    device : torch.device
        Device to perform computation on.
    model_group : dist.ProcessGroup, optional
        Process group for an individual model-parallel instance.

    Returns
    -------
    torch.Tensor
        Local result.
    """
    mod = local_rank % 4

    buffer_x = _get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device)
    buffer_w = _get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device)
    xij = xij.contiguous()
    match mod:
        case 0:
            send_x = dist.isend(xij, group_dst=local_rank + 1, group=model_group)
            recv_x = dist.irecv(buffer_x, group_src=local_rank + 1, group=model_group)

            send_w_2 = dist.isend(wij, group_dst=local_rank + 2, group=model_group)
            recv_w_2 = dist.irecv(buffer_w, group_src=local_rank + 2, group=model_group)

        case 1:
            recv_x = dist.irecv(buffer_x, group_src=local_rank - 1, group=model_group)
            send_x = dist.isend(xij, group_dst=local_rank - 1, group=model_group)

            send_w = dist.isend(wij, group_dst=local_rank + 2, group=model_group)
            recv_w = dist.irecv(buffer_w, group_src=local_rank + 2, group=model_group)

        case 2:
            send_x = dist.isend(xij, group_dst=local_rank + 1, group=model_group)
            recv_x = dist.irecv(buffer_x, group_src=local_rank + 1, group=model_group)

            recv_w = dist.irecv(buffer_w, group_src=local_rank - 2, group=model_group)
            send_w = dist.isend(wij, group_dst=local_rank - 2, group=model_group)

        case 3:
            recv_x = dist.irecv(buffer_x, group_src=local_rank - 1, group=model_group)
            send_x = dist.isend(xij, group_dst=local_rank - 1, group=model_group)

            recv_w = dist.irecv(buffer_w, group_src=local_rank - 2, group=model_group)
            send_w = dist.isend(wij, group_dst=local_rank - 2, group=model_group)

    match mod:
        case 0:
            send_w.wait()
            cij = torch.matmul(xij, wij)
        case 1:
            recv_w.wait()
            cij = torch.matmul(xij, buffer_w)
        case 2:
            recv_w.wait()
            cij = torch.matmul(xij, buffer_w)
        case 3:
            send_w.wait()
            cij = torch.matmul(xij, wij)

    match mod:
        case 0:
            recv_w.wait()
            recv_x.wait()
            send_x.wait()
        case 1:
            send_w.wait()
            send_x.wait()
            recv_x.wait()
        case 2:
            send_w.wait()
            recv_x.wait()
            send_x.wait()
        case 3:
            recv_w.wait()
            send_x.wait()
            recv_x.wait()

    cij = cij + torch.matmul(buffer_x, buffer_w)

    return cij


def fused_xw_xtw_fourway(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    input: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused calculation to calculate grad_input and grad_weight in backward pass of a linear layer.

    Parameters
    ----------
    grad_output: torch.Tensor
        Output with respect to the gradient.
    weight: torch.Tensor
        Model weights.
    input: torch.Tensor
        Model input.
    local_rank: int
        Local model rank.
    device: torch.device
        Device upon which computations occur.
    model_group: dist.ProcessGroup
        Process group of the domain- and tensor-parallel group of this linear layer.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
    Gradient with respect to the input and gradient of the weights.
    """
    mod = local_rank % 4

    grad_output = grad_output.contiguous()
    xdtype, wdtype, idtype = grad_output.dtype, weight.dtype, input.dtype

    match mod:
        case 0:
            buffer_grad_1 = _get_preallocated_buffer(
                grad_output.shape, xdtype, local_rank, device
            )
            buffer_grad_2 = _get_preallocated_buffer(
                grad_output.shape, xdtype, local_rank, device
            )
            # send to 1, receive from 1 & 2
            send_grad_1 = dist.isend(grad_output, dst=local_rank + 1, group=model_group)
            recv_grad_1 = dist.irecv(
                buffer_grad_1, src=local_rank + 1, group=model_group
            )
            recv_grad_2 = dist.irecv(
                buffer_grad_2, src=local_rank + 2, group=model_group
            )

            # w: send to 2, recv from 2
            buffer_w_2 = _get_preallocated_buffer(
                weight.shape, wdtype, local_rank, device
            )
            send_w_2 = dist.isend(weight, dst=local_rank + 2, group=model_group)
            recv_w_2 = dist.irecv(buffer_w_2, src=local_rank + 2, group=model_group)

            # input: send to 2, recv from 2
            buffer_in_2 = _get_preallocated_buffer(
                input.shape, idtype, local_rank, device
            )
            send_in_2 = dist.isend(input, dst=local_rank + 2, group=model_group)
            recv_in_2 = dist.irecv(buffer_in_2, src=local_rank + 2, group=model_group)
        case 1:
            # send to 0, 2, 3, recv from 0,2
            buffer_grad_0 = _get_preallocated_buffer(
                grad_output.shape, xdtype, local_rank, device
            )
            buffer_grad_2 = _get_preallocated_buffer(
                grad_output.shape, xdtype, local_rank, device
            )
            recv_grad_0 = dist.irecv(
                buffer_grad_0, src=local_rank - 1, group=model_group
            )
            send_grad_0 = dist.isend(grad_output, dst=local_rank - 1, group=model_group)
            recv_grad_2 = dist.irecv(
                buffer_grad_2, src=local_rank + 1, group=model_group
            )
            send_grad_2 = dist.isend(grad_output, dst=local_rank + 1, group=model_group)
            send_grad_3 = dist.isend(grad_output, dst=local_rank + 2, group=model_group)

            # w: send to 3, recv from 3
            buffer_w_3 = _get_preallocated_buffer(
                weight.shape, wdtype, local_rank, device
            )
            send_w_3 = dist.isend(weight, dst=local_rank + 2, group=model_group)
            recv_w_3 = dist.irecv(w_buffer_1, src=local_rank + 2, group=model_group)

            # input: send to 3, recv from 3
            buffer_in_3 = _get_preallocated_buffer(
                input.shape, idtype, local_rank, device
            )
            recv_in_3 = dist.irecv(buffer_in_3, src=local_rank + 2, group=model_group)
            send_in_3 = dist.isend(input, dst=local_rank + 2, group=model_group)
        case 2:
            # send to 0, 1, 3, recv from 1, 3
            buffer_grad_1 = _get_preallocated_buffer(
                grad_output.shape, xdtype, local_rank, device
            )
            buffer_grad_3 = _get_preallocated_buffer(
                grad_output.shape, xdtype, local_rank, device
            )
            send_grad_0 = dist.isend(grad_output, dst=local_rank - 2, group=model_group)
            send_grad_1 = dist.isend(grad_output, dst=local_rank - 1, group=model_group)
            recv_grad_1 = dist.irecv(
                buffer_grad_1, src=local_rank - 1, group=model_group
            )
            send_grad_3 = dist.isend(grad_output, dst=local_rank + 1, group=model_group)
            recv_grad_3 = dist.irecv(
                buffer_grad_3, src=local_rank + 1, group=model_group
            )

            # w: send to 0, recv from 0
            buffer_w_0 = _get_preallocated_buffer(
                weight.shape, wdtype, local_rank, device
            )
            recv_w_0 = dist.irecv(buffer_w_0, src=local_rank - 2, group=model_group)
            send_w_0 = dist.isend(weight, dst=local_rank - 2, group=model_group)

            # input: send to 0, recv from 0
            buffer_in_0 = _get_preallocated_buffer(
                input.shape, idtype, local_rank, device
            )
            recv_in_0 = dist.irecv(buffer_in_0, src=local_rank - 2, group=model_group)
            send_in_0 = dist.isend(input, dst=local_rank - 2, group=model_group)

        case 3:
            # send to 2, recv from 1, 2
            buffer_grad_1 = _get_preallocated_buffer(
                grad_output.shape, xdtype, local_rank, device
            )
            buffer_grad_2 = _get_preallocated_buffer(
                grad_output.shape, xdtype, local_rank, device
            )
            recv_grad_2 = dist.irecv(
                buffer_grad_2, src=local_rank - 1, group=model_group
            )
            send_grad_2 = dist.isend(grad_output, dst=local_rank - 1, group=model_group)
            recv_grad_1 = dist.irecv(
                buffer_grad_1, src=local_rank - 2, group=model_group
            )

            # w: send to 1, recv from 1
            buffer_w_1 = _get_preallocated_buffer(
                weight.shape, wdtype, local_rank, device
            )
            recv_w_1 = dist.irecv(buffer_w_1, src=local_rank - 2, group=model_group)
            send_w_1 = dist.isend(weight, dst=local_rank - 2, group=model_group)

            # input: send to 1, recv from 1
            buffer_in_1 = _get_preallocated_buffer(
                input.shape, idtype, local_rank, device
            )
            send_in_1 = dist.isend(input, dst=local_rank - 2, group=model_group)
            recv_in_1 = dist.irecv(buffer_in_1, src=local_rank - 2, group=model_group)

    match mod:
        case 0:
            send_grad_1.wait()
            grad_input = torch.matmul(grad_output, weight)
            grad_weight = torch.matmul(grad_output.mT, input)
        case 1:
            recv_grad_0.wait()
            grad_input = torch.matmul(buffer_grad_0, weight)
            grad_weight = torch.matmul(buffer_grad_0.mT, input)
        case 2:
            recv_grad_3.wait()
            grad_input = torch.matmul(buffer_grad_3, weight)
            grad_weight = torch.matmul(buffer_grad_3.mT, input)
        case 3:
            send_grad_2.wait()
            grad_input = torch.matmul(grad_output, weight)
            grad_weight = torch.matmul(grad_output.mT, input)

    match mod:
        case 0:
            recv_grad_2.wait()
            send_grad_2.wait()
            recv_grad_1.wait()
            recv_w_2.wait()
            send_w_2.wait()
            grad_input = grad_input + torch.matmul(buffer_grad_1, buffer_w_2)
        case 1:
            recv_w_3.wait()
            send_w_3.wait()
            send_grad_0.wait()
            send_w_3.wait()
            recv_w_3.wait()
            grad_input = grad_input + torch.matmul(grad_output, buffer_w_3)
        case 2:
            send_grad_0.wait()
            recv_grad_0.wait()
            send_grad_3.wait()
            send_w_0.wait()
            recv_w_0.wait()
            grad_input = grad_input + torch.matmul(grad_output, buffer_w_0)
        case 3:
            send_w_1.wait()
            recv_w_1.wait()
            recv_grad_2.wait()

            grad_input = grad_input + torch.matmul(buffer_grad_2, buffer_w_1)

    match mod:
        case 0:
            recv_grad_2.wait()
            recv_in_2.wait()
            send_in_0.wait()
            grad_weight = grad_weight + torch.matmul(buffer_grad_2.mT, buffer_in_2)
        case 1:
            send_grad_2.wait()
            send_grad_3.wait()
            recv_grad_2.wait()
            recv_in_3.wait()
            send_in_3.wait()
            grad_weight = grad_weight + torch.matmul(buffer_grad_2.mT, buffer_in_3)
        case 2:
            send_grad_0.wait()
            recv_grad_1.wait()
            send_grad_2.wait()
            send_in_0.wait()
            recv_in_0.wait()
            grad_weight = grad_weight + torch.matmul(buffer_grad_1.mT, buffer_in_0)
        case 3:
            recv_grad_1.wait()
            send_in_1.wait()
            recv_in_1.wait()
            grad_weight = grad_weight + torch.matmul(buffer_grad_1.mT, buffer_in_1)

    return grad_input, grad_weight


def x_transpose_w_fourway(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Batched matrix-matrix (x_transposeW) multiplication distributed across four processes.

    Parameters
    ----------
    xij : torch.Tensor
        Local input matrix.
    wij : torch.Tensor
        Local weight matrix.
    local_rank : int
        Local model rank.
    device : torch.device
        Device to perform computation on.
    model_group : dist.ProcessGroup
        Process group for an individual model-parallel instance.

    Returns
    -------
    cij : torch.Tensor
        Local result.
    """
    mod = local_rank % 4

    xdtype, wdtype = xij.dtype, wij.dtype

    xij = xij.contiguous()
    match mod:
        case 0:
            # x: send to 1, recv from 2
            buffer_x_2 = _get_preallocated_buffer(xij.shape, xdtype, mod, device)
            send_x_1 = dist.isend(xij, dst=local_rank + 1, group=model_group)
            recv_x_2 = dist.irecv(buffer_x_2, src=local_rank + 2, group=model_group)
            # w: send to 2, recv from 2
            buffer_w_2 = _get_preallocated_buffer(wij.shape, wdtype, mod, device)
            send_w_2 = dist.isend(wij, dst=local_rank + 2, group=model_group)
            recv_w_2 = dist.irecv(buffer_w_2, src=local_rank + 2, group=model_group)
        case 1:
            # send to 2,3 recv from 0,2
            buffer_x_0 = _get_preallocated_buffer(xij.shape, xdtype, mod, device)
            buffer_x_2 = _get_preallocated_buffer(xij.shape, xdtype, mod, device)
            recv_x_0 = dist.irecv(buffer_x_0, src=local_rank - 1, group=model_group)
            send_x_2 = dist.isend(xij, dst=local_rank + 1, group=model_group)
            send_x_3 = dist.isend(xij, dst=local_rank + 2, group=model_group)
            recv_x_2 = dist.irecv(buffer_x_2, src=local_rank + 1, group=model_group)
            # send to 3, recv from 3
            buffer_w_3 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            send_w_3 = dist.isend(wij, dst=local_rank + 2, group=model_group)
            recv_w_3 = dist.irecv(buffer_w_3, src=local_rank + 2, group=model_group)
        case 2:
            # x: send to 0,1 recv from 1,3
            buffer_x_1 = _get_preallocated_buffer(xij.shape, xdtype, mod, device)
            buffer_x_3 = _get_preallocated_buffer(xij.shape, xdtype, mod, device)
            recv_x_1 = dist.irecv(buffer_x_1, src=local_rank - 1, group=model_group)
            send_x_0 = dist.isend(xij, dst=local_rank - 2, group=model_group)
            recv_x_3 = dist.irecv(buffer_x_3, src=local_rank + 1, group=model_group)
            send_x_1 = dist.isend(xij, dst=local_rank - 1, group=model_group)
            # w: send to 0, recv from 0
            buffer_w_0 = _get_preallocated_buffer(wij.shape, wdtype, mod, device)
            recv_w_0 = dist.irecv(buffer_w_0, src=local_rank - 2, group=model_group)
            send_w_0 = dist.isend(wij, dst=local_rank - 2, group=model_group)
        case 3:
            # x: send to 2, recv from 1
            buffer_x_1 = _get_preallocated_buffer(xij.shape, xdtype, mod, device)
            recv_x_1 = dist.irecv(buffer_x_1, src=local_rank - 2, group=model_group)
            send_x_2 = dist.isend(xij, dst=local_rank - 1, group=model_group)
            # w: send to 1, recv from 1
            buffer_w_1 = _get_preallocated_buffer(wij.shape, wdtype, mod, device)
            recv_w_1 = dist.irecv(buffer_w_1, src=local_rank - 2, group=model_group)
            send_w_1 = dist.isend(wij, dst=local_rank - 2, group=model_group)

    match mod:
        case 0:
            send_x_0.wait()
            cij = torch.matmul(xij.mT, wij)
        case 1:
            recv_x_0.wait()
            cij = torch.matmul(buffer_x_0.mT, wij)
        case 2:
            recv_x_3.wait()
            cij = torch.matmul(buffer_x_3.mT, wij)
        case 3:
            send_x_2.wait()
            cij = torch.matmul(xij.mT, wij)

    match mod:
        case 0:
            recv_x_2.wait()
            recv_w_2.wait()
            send_w_0.wait()
            cij = cij + torch.matmul(buffer_x_2.mT, buffer_w_2)
        case 1:
            send_x_1.wait()
            recv_x_2.wait()
            recv_w_3.wait()
            send_x_3.wait()
            send_w_3.wait()
            cij = cij + torch.matmul(buffer_x_2.mT, buffer_w_3)
        case 2:
            send_x_0.wait()
            send_w_2.wait()
            recv_x_1.wait()
            recv_w_0.wait()
            send_x_2.wait()
            cij = cij + torch.matmul(buffer_x_1.mT, buffer_w_0)
        case 3:
            send_w_1.wait()
            recv_x_1.wait()
            recv_w_1.wait()
            cij = cij + torch.matmul(buffer_x_1.mT, buffer_w_1)

    return cij


def xw_transpose_eightway(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Batched matrix-matrix (XWT) multiplication distributed across four processes.

    Parameters.
    ----------
    xij : torch.Tensor
        Local input matrix.
    wij : torch.Tensor
        Local weight matrix.
    local_rank : int
        Local model rank.
    device : torch.device
        Device to perform computation on.
    model_group : dist.ProcessGroup, optional
        Process group for an individual model-parallel instance.

    Returns
    -------
    cij : torch.Tensor
        Local result.
    """
    mod = local_rank % 8
    xdtype, wdtype = xij.dtype, wij.dtype

    match mod:
        case 0:
            # x: send to 1, recv from 1
            buffer_x_1 = _get_preallocated_buffer(xij.shape, xdtype, local_rank, device)
            # w: send to 2, 4, 6, recv from 1,2,3
            buffer_w_1 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_2 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_3 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)

        case 1:
            # x: send to 0, recv from 0
            buffer_x_0 = _get_preallocated_buffer(xij.shape, xdtype, local_rank, device)
            # w: send to 0, 2, 4, 6, recv from 4, 5, 6, 7
            buffer_w_4 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_5 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_6 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_7 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)

        case 2:
            # x: send to 3, recv from 3
            buffer_x_3 = _get_preallocated_buffer(xij.shape, xdtype, local_rank, device)
            # w: send to 0, 4, 6, recv from 0,1,3
            buffer_w_0 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_1 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_3 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)

        case 3:
            # x: send to 2, recv from 2
            buffer_x_2 = _get_preallocated_buffer(xij.shape, xdtype, local_rank, device)
            # w: send to 0, 2, 4, 6, recv from 4, 5, 6, 7
            buffer_w_4 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_5 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_6 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_7 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)

        case 4:
            # x: send to 5, recv from 5
            buffer_x_5 = _get_preallocated_buffer(xij.shape, xdtype, local_rank, device)
            # w: send to 1, 3, 5, 7, recv from 0,1,2,3
            buffer_w_0 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_1 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_2 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_3 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)

        case 5:
            # x: send to 4, recv from 4
            buffer_x_4 = _get_preallocated_buffer(xij.shape, xdtype, local_rank, device)
            # w: send to 1, 3, 7, recv from 4, 6, 7
            buffer_w_4 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_6 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_7 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)

        case 6:
            # x: send to 7, recv from 7
            buffer_x_7 = _get_preallocated_buffer(xij.shape, xdtype, local_rank, device)
            # w: send to 1, 3, 5, 7 recv from 0,1,2,3
            buffer_w_0 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_1 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_2 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_3 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)

        case 7:
            # x: send to 6, recv from 6
            buffer_x_6 = _get_preallocated_buffer(xij.shape, xdtype, local_rank, device)
            # w: send to 1, 3, 5, recv from 4,5,6
            buffer_w_5 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_6 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)
            buffer_w_7 = _get_preallocated_buffer(wij.shape, wdtype, local_rank, device)

    handle_w = dist.all_to_all(
        w_recv_list, w_send_list, group=model_group, async_op=True
    )
    handle_x = dist.all_to_all(
        x_recv_list, x_send_list, group=model_group, async_op=True
    )

    with torch.cuda.stream(streams[0]):
        handle_w.wait()
    with torch.cuda.stream(streams[1]):
        handle_w.wait()

    match mod:
        case 0:
            with torch.cuda.stream(streams[0]):
                cij0 = torch.matmul(xij, wij.mT)
            with torch.cuda.stream(streams[1]):
                cij1 = torch.matmul(xij, w_recv_list[2].mT)
        case 1:
            with torch.cuda.stream(streams[0]):
                cij0 = torch.matmul(xij, w_recv_list[5].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = torch.matmul(xij, w_recv_list[7].mT)
        case 2:
            with torch.cuda.stream(streams[0]):
                cij0 = torch.matmul(xij, w_recv_list[0].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = torch.matmul(xij, wij.mT)
        case 3:
            with torch.cuda.stream(streams[0]):
                cij0 = torch.matmul(xij, w_recv_list[5].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = torch.matmul(xij, w_recv_list[7].mT)
        case 4:
            with torch.cuda.stream(streams[0]):
                cij0 = torch.matmul(xij, w_recv_list[0].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = torch.matmul(xij, w_recv_list[2].mT)
        case 5:
            with torch.cuda.stream(streams[0]):
                cij0 = torch.matmul(xij, wij.mT)
            with torch.cuda.stream(streams[1]):
                cij1 = torch.matmul(xij, w_recv_list[7].mT)
        case 6:
            with torch.cuda.stream(streams[0]):
                cij0 = torch.matmul(xij, w_recv_list[0].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = torch.matmul(xij, w_recv_list[2].mT)
        case 7:
            with torch.cuda.stream(streams[0]):
                cij0 = torch.matmul(xij, w_recv_list[5].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = torch.matmul(xij, wij.mT)

    with torch.cuda.stream(streams[0]):
        handle_x.wait()
    with torch.cuda.stream(streams[1]):
        handle_x.wait()

    match mod:
        case 0:
            with torch.cuda.stream(streams[0]):
                cij0 = cij0 + torch.matmul(x_recv_list[1], w_recv_list[1].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = cij1 + torch.matmul(x_recv_list[1], w_recv_list[3].mT)
        case 1:
            with torch.cuda.stream(streams[0]):
                cij0 = cij0 + torch.matmul(x_recv_list[0], w_recv_list[4].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = cij1 + torch.matmul(x_recv_list[0], w_recv_list[6].mT)
        case 2:
            with torch.cuda.stream(streams[0]):
                cij0 = cij0 + torch.matmul(x_recv_list[3], w_recv_list[1].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = cij1 + torch.matmul(x_recv_list[3], w_recv_list[3].mT)
        case 3:
            with torch.cuda.stream(streams[0]):
                cij0 = cij0 + torch.matmul(x_recv_list[2], w_recv_list[4].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = cij1 + torch.matmul(x_recv_list[2], w_recv_list[6].mT)
        case 4:
            with torch.cuda.stream(streams[0]):
                cij0 = cij0 + torch.matmul(x_recv_list[5], w_recv_list[1].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = cij1 + torch.matmul(x_recv_list[5], w_recv_list[3].mT)
        case 5:
            with torch.cuda.stream(streams[0]):
                cij0 = cij0 + torch.matmul(x_recv_list[4], w_recv_list[4].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = cij1 + torch.matmul(x_recv_list[4], w_recv_list[6].mT)
        case 6:
            with torch.cuda.stream(streams[0]):
                cij0 = cij0 + torch.matmul(x_recv_list[7], w_recv_list[1].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = cij1 + torch.matmul(x_recv_list[7], w_recv_list[3].mT)
        case 7:
            with torch.cuda.stream(streams[0]):
                cij0 = cij0 + torch.matmul(x_recv_list[6], w_recv_list[4].mT)
            with torch.cuda.stream(streams[1]):
                cij1 = cij1 + torch.matmul(x_recv_list[6], w_recv_list[6].mT)

    for s in streams:
        s.synchronize()
    cij = torch.concat([cij0, cij1], dim=-1)
    return cij
