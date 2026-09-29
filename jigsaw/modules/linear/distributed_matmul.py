"""Distributed matrix multiplication implementation."""

from typing import Any, cast

import torch
import torch.distributed as dist

from jigsaw.comm import all_to_all_async, get_empty, resolve_matmul_dtype
from jigsaw.modules.buffers import get_preallocated_buffer

streams = None
streamz = None


def fused_xw_xtw_oneway(
    grad_output: torch.Tensor, weight: torch.Tensor, input: torch.Tensor, **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Trivially fused calculation to calculate grad_input and grad_weight in backward pass of a linear layer.

    Parameters
    ----------
    grad_output: torch.Tensor
        Output with respect to the gradient.
    weight: torch.Tensor
        Model weights.
    input: torch.Tensor
        Model input.
    kwargs: Any
        Additional inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
    Gradient with respect to the input and gradient of the weights.
    """
    grad_input = grad_output @ weight
    grad_weight = grad_output.mT @ input
    return grad_input, grad_weight


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
    # ``grad_output`` is sent verbatim through ``all_to_all`` below, which
    # rejects non-contiguous tensors; autograd routinely hands us a strided
    # gradient (e.g. through a preceding transpose / rearrange).
    grad_output = grad_output.contiguous()
    buffer_x = get_preallocated_buffer(
        grad_output.shape, grad_output.dtype, local_rank, device
    )
    empty = get_empty(device, grad_output.dtype)

    match mod:
        case 0:
            xw_send_list = [empty, grad_output]
            xw_recv_list = [empty, buffer_x]
        case 1:
            xw_send_list = [grad_output, empty]
            xw_recv_list = [buffer_x, empty]

    xw_handle = all_to_all_async(xw_recv_list, xw_send_list, group=model_group)

    match mod:
        case 0:
            grad_input = torch.matmul(grad_output, w0)
            grad_weight_top = torch.matmul(grad_output.mT, input)
            xw_handle.wait()
            grad_input.add_(torch.matmul(buffer_x, w1))
            grad_weight_bottom = torch.matmul(buffer_x.mT, input)
        case 1:
            grad_input = torch.matmul(grad_output, w1)
            grad_weight_bottom = torch.matmul(grad_output.mT, input)
            xw_handle.wait()
            grad_input.add_(torch.matmul(buffer_x, w0))
            grad_weight_top = torch.matmul(buffer_x.mT, input)

    grad_weight = torch.concat([grad_weight_top, grad_weight_bottom], dim=-2)
    return grad_input, grad_weight


def xw_transpose_twoway(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Batched matrix-matrix (XWT) multiplication distributed across two processes.

    Parameters
    ----------
    xij : torch.Tensor
        Local input matrix.
    wij : torch.Tensor
        Local weight matrix.
    device : torch.device
        Device to perform computation on.
    local_rank : int
        Local local_rank.
    model_group : dist.ProcessGroup, optional
        Process group for an individual model-parallel instance.

    Returns
    -------
    cij : torch.Tensor
        Local result.
    """
    mod = local_rank % 2
    wij = wij.to(xij.device)
    w0, w1 = torch.split(wij, wij.shape[-2] // 2, dim=-2)

    # ``xw`` is a matmul result, so under autocast its dtype is not knowable
    # from the inputs alone. Resolve it up front so the receive buffer and the
    # placeholder can be allocated before the ``xw`` matmul runs.
    xw_dtype = resolve_matmul_dtype(device, xij, wij)
    buffer_xw = get_preallocated_buffer(
        cast(torch.Size, (*xij.shape[:-1], w0.shape[-2])),
        xw_dtype,
        local_rank,
        device,
    )
    empty = get_empty(device, xw_dtype)

    match mod:
        case 0:
            xw = torch.matmul(xij, w1.mT)
            xw_send_list = [empty, xw]
            xw_recv_list = [empty, buffer_xw]
        case 1:
            xw = torch.matmul(xij, w0.mT)
            xw_send_list = [xw, empty]
            xw_recv_list = [buffer_xw, empty]

    xw_handle = all_to_all_async(xw_recv_list, xw_send_list, group=model_group)

    match mod:
        case 0:
            cij = torch.matmul(xij, w0.mT)
            xw_handle.wait()
            cij.add_(xw_recv_list[1])
        case 1:
            cij = torch.matmul(xij, w1.mT)
            xw_handle.wait()
            cij.add_(xw_recv_list[0])

    return cij


@torch.compile
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
    buffer_x = get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device)
    empty = get_empty(device, xij.dtype)

    match mod:
        case 0:
            xw_send_list = [empty, xij]
            xw_recv_list = [empty, buffer_x]
        case 1:
            xw_send_list = [xij, empty]
            xw_recv_list = [buffer_x, empty]

    xw_handle = all_to_all_async(xw_recv_list, xw_send_list, group=model_group)

    match mod:
        case 0:
            cij = torch.matmul(xij, w0)
            xw_handle.wait()
            cij = cij + torch.matmul(xw_recv_list[1], w1)
        case 1:
            cij = torch.matmul(xij, w1)
            xw_handle.wait()
            cij = cij + torch.matmul(xw_recv_list[0], w0)

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
    empty = get_empty(device, xij.dtype)

    buffer_x = get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device)

    match mod:
        case 0:
            xtw_send_list = [empty, xij]
            xtw_recv_list = [empty, buffer_x]
        case 1:
            xtw_send_list = [xij, empty]
            xtw_recv_list = [buffer_x, empty]

    xtw_handle = all_to_all_async(xtw_recv_list, xtw_send_list, group=model_group)

    match mod:
        case 0:
            cij_top = torch.matmul(xij.mT, wij)
            xtw_handle.wait()
            cij_bottom = torch.matmul(buffer_x.mT, wij)
        case 1:
            cij_bottom = torch.matmul(xij.mT, wij)
            xtw_handle.wait()
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
    empty = get_empty(device, xij.dtype)

    match mod:
        case 0:
            x_send_list = [empty, xij, empty, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
            ]
            w_send_list = [empty, empty, wij, empty]
            w_recv_list = [
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 1),
                empty,
                empty,
            ]
        case 1:
            x_send_list = [xij, empty, empty, empty]
            x_recv_list = [
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
            ]
            w_send_list = [wij, empty, wij, empty]
            w_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
            ]
        case 2:
            x_send_list = [empty, empty, empty, xij]
            x_recv_list = [
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
            ]
            w_send_list = [empty, wij, empty, wij]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                empty,
                empty,
            ]
        case 3:
            x_send_list = [empty, empty, xij, empty]
            x_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
            ]
            w_send_list = [empty, wij, empty, empty]
            w_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 1),
                empty,
            ]

    handle_1 = all_to_all_async(x_recv_list, x_send_list, group=model_group)
    handle_2 = all_to_all_async(w_recv_list, w_send_list, group=model_group)

    handle_2.wait()

    match mod:
        case 0:
            cij = torch.matmul(xij, wij.mT)
        case 1:
            cij = torch.matmul(xij, w_recv_list[3].mT)
        case 2:
            cij = torch.matmul(xij, w_recv_list[0].mT)
        case 3:
            cij = torch.matmul(xij, wij.mT)

    handle_1.wait()

    match mod:
        case 0:
            cij = cij + torch.matmul(x_recv_list[1], w_recv_list[1].mT)
        case 1:
            cij = cij + torch.matmul(x_recv_list[0], w_recv_list[2].mT)
        case 2:
            cij = cij + torch.matmul(x_recv_list[3], w_recv_list[1].mT)
        case 3:
            cij = cij + torch.matmul(x_recv_list[2], w_recv_list[2].mT)

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
    empty = get_empty(device, xij.dtype)

    xdtype, wdtype = xij.dtype, wij.dtype
    xij = xij.contiguous()
    match mod:
        case 0:
            x_send_list = [empty, xij, empty, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(xij.shape, xdtype, local_rank, device, 0),
                empty,
                empty,
            ]
            w_send_list = [empty, empty, wij, empty]
            w_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wdtype, local_rank, device, 1),
                empty,
            ]
        case 1:
            x_send_list = [xij, empty, empty, empty]
            x_recv_list = [
                get_preallocated_buffer(xij.shape, xdtype, local_rank, device, 0),
                empty,
                empty,
                empty,
            ]
            w_send_list = [empty, empty, empty, wij]
            w_recv_list = [
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wdtype, local_rank, device, 1),
            ]
        case 2:
            x_send_list = [empty, empty, empty, xij]
            x_recv_list = [
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xdtype, local_rank, device, 0),
            ]
            w_send_list = [wij, empty, empty, empty]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, wdtype, local_rank, device, 1),
                empty,
                empty,
                empty,
            ]
        case 3:
            x_send_list = [empty, empty, xij, empty]
            x_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xdtype, local_rank, device, 0),
                empty,
            ]
            w_send_list = [empty, wij, empty, empty]
            w_recv_list = [
                empty,
                get_preallocated_buffer(wij.shape, wdtype, local_rank, device, 1),
                empty,
                empty,
            ]

    w_handle = all_to_all_async(w_recv_list, w_send_list, group=model_group)
    x_handle = all_to_all_async(x_recv_list, x_send_list, group=model_group)

    w_handle.wait()

    match mod:
        case 0:
            cij = torch.matmul(xij, wij)
        case 1:
            cij = torch.matmul(xij, w_recv_list[3])
        case 2:
            cij = torch.matmul(xij, w_recv_list[0])
        case 3:
            cij = torch.matmul(xij, wij)

    x_handle.wait()

    match mod:
        case 0:
            cij = cij + torch.matmul(x_recv_list[1], w_recv_list[2])
        case 1:
            cij = cij + torch.matmul(x_recv_list[0], wij)
        case 2:
            cij = cij + torch.matmul(x_recv_list[3], wij)
        case 3:
            cij = cij + torch.matmul(x_recv_list[2], w_recv_list[1])

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
    empty = get_empty(device, grad_output.dtype)

    grad_output = grad_output.contiguous()
    xdtype, wdtype, idtype = grad_output.dtype, weight.dtype, input.dtype

    match mod:
        case 0:
            grad_send_list = [empty, grad_output, empty, empty]
            grad_recv_list = [
                empty,
                get_preallocated_buffer(
                    grad_output.shape, xdtype, local_rank, device, 0
                ),
                get_preallocated_buffer(
                    grad_output.shape, xdtype, local_rank, device, 0
                ),
                empty,
            ]
            w_send_list = [empty, empty, weight, empty]
            w_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(weight.shape, wdtype, local_rank, device, 1),
                empty,
            ]
            i_send_list = [empty, empty, input, empty]
            i_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(input.shape, idtype, local_rank, device, 1),
                empty,
            ]
        case 1:
            grad_send_list = [grad_output, empty, grad_output, grad_output]
            grad_recv_list = [
                get_preallocated_buffer(
                    grad_output.shape, xdtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    grad_output.shape, xdtype, local_rank, device, 0
                ),
                empty,
            ]
            w_send_list = [empty, empty, empty, weight]
            w_recv_list = [
                empty,
                empty,
                empty,
                get_preallocated_buffer(weight.shape, wdtype, local_rank, device, 1),
            ]
            i_send_list = [empty, empty, empty, input]
            i_recv_list = [
                empty,
                empty,
                empty,
                get_preallocated_buffer(input.shape, idtype, local_rank, device, 1),
            ]
        case 2:
            grad_send_list = [grad_output, grad_output, empty, grad_output]
            grad_recv_list = [
                empty,
                get_preallocated_buffer(
                    grad_output.shape, xdtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    grad_output.shape, xdtype, local_rank, device, 0
                ),
            ]
            w_send_list = [weight, empty, empty, empty]
            w_recv_list = [
                get_preallocated_buffer(weight.shape, wdtype, local_rank, device, 1),
                empty,
                empty,
                empty,
            ]
            i_send_list = [input, empty, empty, empty]
            i_recv_list = [
                get_preallocated_buffer(input.shape, idtype, local_rank, device, 1),
                empty,
                empty,
                empty,
            ]
        case 3:
            grad_send_list = [empty, empty, grad_output, empty]
            grad_recv_list = [
                empty,
                get_preallocated_buffer(
                    grad_output.shape, xdtype, local_rank, device, 0
                ),
                get_preallocated_buffer(
                    grad_output.shape, xdtype, local_rank, device, 0
                ),
                empty,
            ]
            w_send_list = [empty, weight, empty, empty]
            w_recv_list = [
                empty,
                get_preallocated_buffer(weight.shape, wdtype, local_rank, device, 1),
                empty,
                empty,
            ]
            i_send_list = [empty, input, empty, empty]
            i_recv_list = [
                empty,
                get_preallocated_buffer(input.shape, idtype, local_rank, device, 1),
                empty,
                empty,
            ]

    x_handle = all_to_all_async(grad_recv_list, grad_send_list, group=model_group)
    w_handle = all_to_all_async(w_recv_list, w_send_list, group=model_group)
    i_handle = all_to_all_async(i_recv_list, i_send_list, group=model_group)

    x_handle.wait()

    match mod:
        case 0:
            grad_input = torch.matmul(grad_output, weight)
            grad_weight = torch.matmul(grad_output.mT, input)
        case 1:
            grad_input = torch.matmul(grad_recv_list[0], weight)
            grad_weight = torch.matmul(grad_recv_list[0].mT, input)
        case 2:
            grad_input = torch.matmul(grad_recv_list[3], weight)
            grad_weight = torch.matmul(grad_recv_list[3].mT, input)
        case 3:
            grad_input = torch.matmul(grad_output, weight)
            grad_weight = torch.matmul(grad_output.mT, input)

    w_handle.wait()
    match mod:
        case 0:
            grad_input = grad_input + torch.matmul(grad_recv_list[1], w_recv_list[2])
        case 1:
            grad_input = grad_input + torch.matmul(grad_output, w_recv_list[3])
        case 2:
            grad_input = grad_input + torch.matmul(grad_output, w_recv_list[0])
        case 3:
            grad_input = grad_input + torch.matmul(grad_recv_list[2], w_recv_list[1])

    i_handle.wait()
    match mod:
        case 0:
            grad_weight = grad_weight + torch.matmul(
                grad_recv_list[2].mT, i_recv_list[2]
            )
        case 1:
            grad_weight = grad_weight + torch.matmul(
                grad_recv_list[2].mT, i_recv_list[3]
            )
        case 2:
            grad_weight = grad_weight + torch.matmul(
                grad_recv_list[1].mT, i_recv_list[0]
            )
        case 3:
            grad_weight = grad_weight + torch.matmul(
                grad_recv_list[1].mT, i_recv_list[1]
            )

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
    empty = get_empty(device, xij.dtype)

    xdtype, wdtype = xij.dtype, wij.dtype

    xij = xij.contiguous()
    match mod:
        case 0:
            x_send_list = [empty, xij, empty, empty]
            x_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xdtype, local_rank, device, 0),
                empty,
            ]
            w_send_list = [empty, empty, wij, empty]
            w_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wdtype, local_rank, device, 1),
                empty,
            ]
        case 1:
            x_send_list = [empty, empty, xij, xij]
            x_recv_list = [
                get_preallocated_buffer(xij.shape, xdtype, local_rank, device, 0),
                empty,
                get_preallocated_buffer(xij.shape, xdtype, local_rank, device, 1),
                empty,
            ]
            w_send_list = [empty, empty, empty, wij]
            w_recv_list = [
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wdtype, local_rank, device, 2),
            ]
        case 2:
            x_send_list = [xij, xij, empty, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(xij.shape, xdtype, local_rank, device, 0),
                empty,
                get_preallocated_buffer(xij.shape, xdtype, local_rank, device, 1),
            ]
            w_send_list = [wij, empty, empty, empty]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, wdtype, local_rank, device, 2),
                empty,
                empty,
                empty,
            ]
        case 3:
            x_send_list = [empty, empty, xij, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(xij.shape, xdtype, local_rank, device, 0),
                empty,
                empty,
            ]
            w_send_list = [empty, wij, empty, empty]
            w_recv_list = [
                empty,
                get_preallocated_buffer(wij.shape, wdtype, local_rank, device, 1),
                empty,
                empty,
            ]

    x_handle = all_to_all_async(x_recv_list, x_send_list, group=model_group)
    w_handle = all_to_all_async(w_recv_list, w_send_list, group=model_group)

    x_handle.wait()

    match mod:
        case 0:
            cij = torch.matmul(xij.mT, wij)
        case 1:
            cij = torch.matmul(x_recv_list[0].mT, wij)
        case 2:
            cij = torch.matmul(x_recv_list[3].mT, wij)
        case 3:
            cij = torch.matmul(xij.mT, wij)

    w_handle.wait()

    match mod:
        case 0:
            cij = cij + torch.matmul(x_recv_list[2].mT, w_recv_list[2])
        case 1:
            cij = cij + torch.matmul(x_recv_list[2].mT, w_recv_list[3])
        case 2:
            cij = cij + torch.matmul(x_recv_list[1].mT, w_recv_list[0])
        case 3:
            cij = cij + torch.matmul(x_recv_list[1].mT, w_recv_list[1])

    return cij


def xw_transpose_eightway_(
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
    mod = local_rank % 8
    empty = get_empty(device, xij.dtype)

    match mod:
        case 0:
            x_send_list = [empty, xij, empty, empty, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
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
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 3),
                empty,
                empty,
                empty,
                empty,
            ]
        case 1:
            x_send_list = [xij, empty, empty, empty, empty, empty, empty, empty]
            x_recv_list = [
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
            ]
            w_send_list = [wij, empty, wij, empty, wij, empty, wij, empty]
            w_recv_list = [
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 4),
            ]
        case 2:
            x_send_list = [empty, empty, empty, xij, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
                empty,
            ]
            w_send_list = [wij, empty, empty, empty, wij, empty, wij, empty]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 2),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 3),
                empty,
                empty,
                empty,
                empty,
            ]
        case 3:
            x_send_list = [empty, empty, xij, empty, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
                empty,
                empty,
            ]
            w_send_list = [wij, empty, wij, empty, wij, empty, wij, empty]
            w_recv_list = [
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 4),
            ]
        case 4:
            x_send_list = [empty, empty, empty, empty, empty, xij, empty, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
            ]
            w_send_list = [empty, wij, empty, wij, empty, wij, empty, wij]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 3),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 4),
                empty,
                empty,
                empty,
                empty,
            ]
        case 5:
            x_send_list = [empty, empty, empty, empty, xij, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
            ]
            w_send_list = [empty, wij, empty, wij, empty, empty, empty, wij]
            w_recv_list = [
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
            ]
        case 6:
            x_send_list = [empty, empty, empty, empty, empty, empty, empty, xij]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
            ]
            w_send_list = [empty, wij, empty, wij, empty, wij, empty, wij]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 3),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 4),
                empty,
                empty,
                empty,
                empty,
            ]
        case 7:
            x_send_list = [empty, empty, empty, empty, empty, empty, xij, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
            ]
            w_send_list = [empty, wij, empty, wij, empty, wij, empty, empty]
            w_recv_list = [
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                empty,
            ]

    handle_w = all_to_all_async(w_recv_list, w_send_list, group=model_group)
    handle_x = all_to_all_async(x_recv_list, x_send_list, group=model_group)

    cij = get_preallocated_buffer(
        cast(torch.Size, (*xij.shape[:-1], 2 * wij.shape[-2])),
        dtype=xij.dtype,
        local_rank=local_rank,
        device=device,
        index=10,
    )  # new buffer...

    w_dim = wij.shape[-2]

    handle_w.wait()

    match mod:
        case 0:
            cij[:, :, :w_dim] = torch.matmul(xij, wij.mT)
            cij[:, :, w_dim:] = torch.matmul(xij, w_recv_list[2].mT)
        case 1:
            cij[:, :, :w_dim] = torch.matmul(xij, w_recv_list[5].mT)
            cij[:, :, w_dim:] = torch.matmul(xij, w_recv_list[7].mT)
        case 2:
            cij[:, :, :w_dim] = torch.matmul(xij, w_recv_list[0].mT)
            cij[:, :, w_dim:] = torch.matmul(xij, wij.mT)
        case 3:
            cij[:, :, :w_dim] = torch.matmul(xij, w_recv_list[5].mT)
            cij[:, :, w_dim:] = torch.matmul(xij, w_recv_list[7].mT)
        case 4:
            cij[:, :, :w_dim] = torch.matmul(xij, w_recv_list[0].mT)
            cij[:, :, w_dim:] = torch.matmul(xij, w_recv_list[2].mT)
        case 5:
            cij[:, :, :w_dim] = torch.matmul(xij, wij.mT)
            cij[:, :, w_dim:] = torch.matmul(xij, w_recv_list[7].mT)
        case 6:
            cij[:, :, :w_dim] = torch.matmul(xij, w_recv_list[0].mT)
            cij[:, :, w_dim:] = torch.matmul(xij, w_recv_list[2].mT)
        case 7:
            cij[:, :, :w_dim] = torch.matmul(xij, w_recv_list[5].mT)
            cij[:, :, w_dim:] = torch.matmul(xij, wij.mT)

    handle_x.wait()

    match mod:
        case 0:
            cij[:, :, :w_dim] = cij[:, :, :w_dim] + torch.matmul(
                x_recv_list[1], w_recv_list[1].mT
            )
            cij[:, :, w_dim:] = cij[:, :, w_dim:] + torch.matmul(
                x_recv_list[1], w_recv_list[3].mT
            )
        case 1:
            cij[:, :, :w_dim] = cij[:, :, :w_dim] + torch.matmul(
                x_recv_list[0], w_recv_list[4].mT
            )
            cij[:, :, w_dim:] = cij[:, :, w_dim:] + torch.matmul(
                x_recv_list[0], w_recv_list[6].mT
            )
        case 2:
            cij[:, :, :w_dim] = cij[:, :, :w_dim] + torch.matmul(
                x_recv_list[3], w_recv_list[1].mT
            )
            cij[:, :, w_dim:] = cij[:, :, w_dim:] + torch.matmul(
                x_recv_list[3], w_recv_list[3].mT
            )
        case 3:
            cij[:, :, :w_dim] = cij[:, :, :w_dim] + torch.matmul(
                x_recv_list[2], w_recv_list[4].mT
            )
            cij[:, :, w_dim:] = cij[:, :, w_dim:] + torch.matmul(
                x_recv_list[2], w_recv_list[6].mT
            )
        case 4:
            cij[:, :, :w_dim] = cij[:, :, :w_dim] + torch.matmul(
                x_recv_list[5], w_recv_list[1].mT
            )
            cij[:, :, w_dim:] = cij[:, :, w_dim:] + torch.matmul(
                x_recv_list[5], w_recv_list[3].mT
            )
        case 5:
            cij[:, :, :w_dim] = cij[:, :, :w_dim] + torch.matmul(
                x_recv_list[4], w_recv_list[4].mT
            )
            cij[:, :, w_dim:] = cij[:, :, w_dim:] + torch.matmul(
                x_recv_list[4], w_recv_list[6].mT
            )
        case 6:
            cij[:, :, :w_dim] = cij[:, :, :w_dim] + torch.matmul(
                x_recv_list[7], w_recv_list[1].mT
            )
            cij[:, :, w_dim:] = cij[:, :, w_dim:] + torch.matmul(
                x_recv_list[7], w_recv_list[3].mT
            )
        case 7:
            cij[:, :, :w_dim] = cij[:, :, :w_dim] + torch.matmul(
                x_recv_list[6], w_recv_list[4].mT
            )
            cij[:, :, w_dim:] = cij[:, :, w_dim:] + torch.matmul(
                x_recv_list[6], w_recv_list[6].mT
            )

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
    empty = get_empty(device, xij.dtype)
    streams: list[torch.cuda.Stream] = [torch.cuda.Stream() for _ in range(2)]
    match mod:
        case 0:
            x_send_list = [empty, xij, empty, empty, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
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
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 3),
                empty,
                empty,
                empty,
                empty,
            ]
        case 1:
            x_send_list = [xij, empty, empty, empty, empty, empty, empty, empty]
            x_recv_list = [
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
            ]
            w_send_list = [wij, empty, wij, empty, wij, empty, wij, empty]
            w_recv_list = [
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 4),
            ]
        case 2:
            x_send_list = [empty, empty, empty, xij, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
                empty,
            ]
            w_send_list = [wij, empty, empty, empty, wij, empty, wij, empty]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 2),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 3),
                empty,
                empty,
                empty,
                empty,
            ]
        case 3:
            x_send_list = [empty, empty, xij, empty, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
                empty,
                empty,
            ]
            w_send_list = [wij, empty, wij, empty, wij, empty, wij, empty]
            w_recv_list = [
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 4),
            ]
        case 4:
            x_send_list = [empty, empty, empty, empty, empty, xij, empty, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
            ]
            w_send_list = [empty, wij, empty, wij, empty, wij, empty, wij]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 3),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 4),
                empty,
                empty,
                empty,
                empty,
            ]
        case 5:
            x_send_list = [empty, empty, empty, empty, xij, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
            ]
            w_send_list = [empty, wij, empty, wij, empty, empty, empty, wij]
            w_recv_list = [
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
            ]
        case 6:
            x_send_list = [empty, empty, empty, empty, empty, empty, empty, xij]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
            ]
            w_send_list = [empty, wij, empty, wij, empty, wij, empty, wij]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 3),
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 4),
                empty,
                empty,
                empty,
                empty,
            ]
        case 7:
            x_send_list = [empty, empty, empty, empty, empty, empty, xij, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
            ]
            w_send_list = [empty, wij, empty, wij, empty, wij, empty, empty]
            w_recv_list = [
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                empty,
            ]

    handle_w = all_to_all_async(w_recv_list, w_send_list, group=model_group)
    handle_x = all_to_all_async(x_recv_list, x_send_list, group=model_group)

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


def xw_eightway_(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Batched matrix-matrix (XW) multiplication distributed across eight processes.

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
    # mod = rank % 8
    mod = local_rank
    empty = get_empty(device, xij.dtype)
    match mod:
        case 0:
            x_send_list = [empty, xij, empty, empty, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
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
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                empty,
            ]
        case 1:
            x_send_list = [xij, empty, empty, empty, empty, empty, empty, empty]
            x_recv_list = [
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
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
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
            ]
        case 2:
            x_send_list = [empty, empty, xij, xij, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
                empty,
            ]
            w_send_list = [wij, empty, empty, empty, wij, empty, wij, empty]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                empty,
            ]
        case 3:
            x_send_list = [empty, empty, xij, xij, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
                empty,
            ]
            w_send_list = [empty, wij, empty, empty, empty, wij, empty, wij]
            w_recv_list = [
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
            ]
        case 4:
            x_send_list = [empty, empty, empty, empty, empty, xij, empty, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
            ]
            w_send_list = [wij, empty, wij, empty, empty, empty, wij, empty]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                empty,
            ]
        case 5:
            x_send_list = [empty, empty, empty, empty, xij, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
            ]
            w_send_list = [empty, wij, empty, wij, empty, empty, empty, wij]
            w_recv_list = [
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
            ]
        case 6:
            x_send_list = [empty, empty, empty, empty, empty, empty, empty, xij]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
            ]
            w_send_list = [wij, empty, wij, empty, wij, empty, empty, empty]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                empty,
                empty,
                empty,
            ]
        case 7:
            x_send_list = [empty, empty, empty, empty, empty, empty, xij, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
            ]
            w_send_list = [empty, wij, empty, wij, empty, wij, empty, empty]
            w_recv_list = [
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 2),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                empty,
                empty,
            ]

    handle_w = all_to_all_async(w_recv_list, w_send_list, group=model_group)
    handle_x = all_to_all_async(x_recv_list, x_send_list, group=model_group)

    handle_w.wait()
    x0, x1 = torch.split(xij, xij.shape[-1] // 2, dim=-1)
    x0 = x0.contiguous()
    x1 = x1.contiguous()
    match mod:
        case 0:
            # ws_stacked = torch.concatenate([wij, w_recv_list[2]], dim=-2).contiguous()
            cij = torch.matmul(x0, wij) + torch.matmul(x1, w_recv_list[2])
        case 1:
            # ws_stacked = torch.concatenate(
            #    [w_recv_list[5], w_recv_list[7]], dim=-2
            # ).contiguous()
            cij = torch.matmul(x0, w_recv_list[5]) + torch.matmul(x1, w_recv_list[7])
        case 2:
            # ws_stacked = torch.concatenate([w_recv_list[0], wij], dim=-2).contiguous()
            cij = torch.matmul(x0, w_recv_list[0]) + torch.matmul(x1, wij)
        case 3:
            # ws_stacked = torch.concatenate(
            #    [w_recv_list[5], w_recv_list[7]], dim=-2
            # ).contiguous()
            cij = torch.matmul(x0, w_recv_list[5]) + torch.matmul(x1, w_recv_list[7])
        case 4:
            # ws_stacked = torch.concatenate(
            #    [w_recv_list[0], w_recv_list[2]], dim=-2
            # ).contiguous()
            cij = torch.matmul(x0, w_recv_list[0]) + torch.matmul(x1, w_recv_list[2])
        case 5:
            # ws_stacked = torch.concatenate([wij, w_recv_list[7]], dim=-2).contiguous()
            cij = torch.matmul(x0, wij) + torch.matmul(x1, w_recv_list[7])
        case 6:
            # ws_stacked = torch.concatenate(
            #    [w_recv_list[0], w_recv_list[2]], dim=-2
            # ).contiguous()
            cij = torch.matmul(x0, w_recv_list[0]) + torch.matmul(x1, w_recv_list[2])
        case 7:
            # ws_stacked = torch.concatenate([w_recv_list[5], wij], dim=-2).contiguous()
            cij = torch.matmul(x0, w_recv_list[5]) + torch.matmul(x1, wij)

    handle_x.wait()

    match mod:
        case 0:
            x0, x1 = torch.split(
                x_recv_list[1], x_recv_list[1].shape[-1] // 2, dim=-1
            )  # split upon last dimension
            cij = (
                cij
                + torch.matmul(x0, w_recv_list[4])
                + torch.matmul(x1, w_recv_list[6])
            )
        case 1:
            x0, x1 = torch.split(
                x_recv_list[0], x_recv_list[0].shape[-1] // 2, dim=-1
            )  # split upon last dimension
            cij = cij + torch.matmul(x0, wij) + torch.matmul(x1, w_recv_list[3])
        case 2:
            x0, x1 = torch.split(
                x_recv_list[3], x_recv_list[3].shape[-1] // 2, dim=-1
            )  # split upon last dimension
            cij = (
                cij
                + torch.matmul(x0, w_recv_list[4])
                + torch.matmul(x1, w_recv_list[6])
            )
        case 3:
            x0, x1 = torch.split(
                x_recv_list[2], x_recv_list[2].shape[-1] // 2, dim=-1
            )  # split upon last dimension
            cij = cij + torch.matmul(x0, w_recv_list[1]) + torch.matmul(x1, wij)
        case 4:
            x0, x1 = torch.split(
                x_recv_list[5], x_recv_list[5].shape[-1] // 2, dim=-1
            )  # split upon last dimension
            cij = cij + torch.matmul(x0, wij) + torch.matmul(x1, w_recv_list[6])
        case 5:
            x0, x1 = torch.split(
                x_recv_list[4], x_recv_list[4].shape[-1] // 2, dim=-1
            )  # split upon last dimension
            cij = (
                cij
                + torch.matmul(x0, w_recv_list[1])
                + torch.matmul(x1, w_recv_list[3])
            )
        case 6:
            x0, x1 = torch.split(
                x_recv_list[7], x_recv_list[7].shape[-1] // 2, dim=-1
            )  # split upon last dimension
            cij = cij + torch.matmul(x0, w_recv_list[4]) + torch.matmul(x1, wij)
        case 7:
            x0, x1 = torch.split(
                x_recv_list[6], x_recv_list[6].shape[-1] // 2, dim=-1
            )  # split upon last dimension
            cij = (
                cij
                + torch.matmul(x0, w_recv_list[1])
                + torch.matmul(x1, w_recv_list[3])
            )

    return cij


def xw_eightway(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Batched matrix-matrix (XW) multiplication distributed across eight processes.

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
    empty = get_empty(device, torch.float32)
    streams: list[torch.cuda.Stream] = [torch.cuda.Stream() for _ in range(2)]

    w0 = get_preallocated_buffer(
        cast(torch.Size, (*wij.shape[:-2], wij.shape[-2] * 2, wij.shape[-1])),
        dtype=wij.dtype,
        local_rank=local_rank,
        device=device,
        index=1,
    )
    w1 = get_preallocated_buffer(
        cast(torch.Size, (*wij.shape[:-2], wij.shape[-2] * 2, wij.shape[-1])),
        dtype=wij.dtype,
        local_rank=local_rank,
        device=device,
        index=2,
    )

    w0_shape = w0.shape
    split_size = w0_shape[-2] // 2
    empty = get_empty(device, torch.float32)

    match mod:
        case 0:
            x_send_list = [empty, xij, empty, empty, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
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
                w0[..., split_size:, :].contiguous(),
                empty,
                w1[..., :split_size, :].contiguous(),
                empty,
                w1[..., split_size:, :].contiguous(),
                empty,
            ]
            handle_w = all_to_all_async(w_recv_list, w_send_list, group=model_group)
            handle_x = all_to_all_async(x_recv_list, x_send_list, group=model_group)
            w0[..., :split_size, :].copy_(wij)
        case 1:
            x_send_list = [xij, empty, empty, empty, empty, empty, empty, empty]
            x_recv_list = [
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
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
                w1[..., split_size:, :].contiguous(),
                empty,
                w0[..., :split_size, :].contiguous(),
                empty,
                w0[..., split_size:, :].contiguous(),
            ]
            handle_w = all_to_all_async(w_recv_list, w_send_list, group=model_group)
            handle_x = all_to_all_async(x_recv_list, x_send_list, group=model_group)
            w1[..., :split_size, :].copy_(wij)
        case 2:
            x_send_list = [empty, empty, empty, xij, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
                empty,
            ]
            w_send_list = [wij, empty, empty, empty, wij, empty, wij, empty]
            w_recv_list = [
                w0[..., :split_size, :].contiguous(),
                empty,
                empty,
                empty,
                w1[..., :split_size, :].contiguous(),
                empty,
                w1[..., split_size:, :].contiguous(),
                empty,
            ]
            handle_w = all_to_all_async(w_recv_list, w_send_list, group=model_group)
            handle_x = all_to_all_async(x_recv_list, x_send_list, group=model_group)
            w0[..., split_size:, :].copy_(wij)
        case 3:
            x_send_list = [empty, empty, xij, empty, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
                empty,
                empty,
            ]
            w_send_list = [empty, wij, empty, empty, empty, wij, empty, wij]
            w_recv_list = [
                empty,
                w1[..., :split_size, :],
                empty,
                empty,
                empty,
                w0[..., :split_size, :],
                empty,
                w0[..., split_size:, :],
            ]
            handle_w = all_to_all_async(w_recv_list, w_send_list, group=model_group)
            handle_x = all_to_all_async(x_recv_list, x_send_list, group=model_group)
            w0[..., split_size:, :].copy_(wij)
        case 4:
            x_send_list = [empty, empty, empty, empty, empty, xij, empty, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
            ]
            w_send_list = [wij, empty, wij, empty, empty, empty, wij, empty]
            w_recv_list = [
                w0[..., :split_size, :],
                empty,
                w0[..., split_size:, :],
                empty,
                empty,
                empty,
                w1[..., split_size:, :],
                empty,
            ]
            handle_w = all_to_all_async(w_recv_list, w_send_list, group=model_group)
            handle_x = all_to_all_async(x_recv_list, x_send_list, group=model_group)
            w1[..., :split_size, :].copy_(wij)
        case 5:
            x_send_list = [empty, empty, empty, empty, xij, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
            ]
            w_send_list = [empty, wij, empty, wij, empty, empty, empty, wij]
            w_recv_list = [
                empty,
                w1[..., :split_size, :],
                empty,
                w1[..., split_size:, :],
                empty,
                empty,
                empty,
                w0[..., split_size:, :],
            ]
            handle_w = all_to_all_async(w_recv_list, w_send_list, group=model_group)
            handle_x = all_to_all_async(x_recv_list, x_send_list, group=model_group)
            w0[..., :split_size, :].copy_(wij)
        case 6:
            x_send_list = [empty, empty, empty, empty, empty, empty, empty, xij]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
            ]
            w_send_list = [wij, empty, wij, empty, wij, empty, empty, empty]
            w_recv_list = [
                w0[..., :split_size, :],
                empty,
                w0[..., split_size:, :],
                empty,
                w1[..., :split_size, :],
                empty,
                empty,
                empty,
            ]
            handle_w = all_to_all_async(w_recv_list, w_send_list, group=model_group)
            handle_x = all_to_all_async(x_recv_list, x_send_list, group=model_group)
            w1[..., split_size:, :].copy_(wij)
        case 7:
            x_send_list = [empty, empty, empty, empty, empty, empty, xij, empty]
            x_recv_list = [
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(xij.shape, xij.dtype, local_rank, device, 0),
                empty,
            ]
            w_send_list = [empty, wij, empty, wij, empty, wij, empty, empty]
            w_recv_list = [
                empty,
                w1[..., :split_size, :],
                empty,
                w1[..., split_size:, :],
                empty,
                w0[..., :split_size, :],
                empty,
                empty,
            ]
            handle_w = all_to_all_async(w_recv_list, w_send_list, group=model_group)
            handle_x = all_to_all_async(x_recv_list, x_send_list, group=model_group)
            w0[..., split_size:, :].copy_(wij)

    with torch.cuda.stream(streams[0]):
        handle_w.wait()
        cij0 = torch.matmul(xij, w0)
    match mod:
        case 0:
            with torch.cuda.stream(streams[1]):
                handle_w.wait()
                handle_x.wait()
                cij1 = torch.matmul(x_recv_list[1], w1)
        case 1:
            with torch.cuda.stream(streams[1]):
                handle_w.wait()
                handle_x.wait()
                cij1 = torch.matmul(x_recv_list[0], w1)
        case 2:
            with torch.cuda.stream(streams[1]):
                handle_w.wait()
                handle_x.wait()
                cij1 = torch.matmul(x_recv_list[3], w1)
        case 3:
            with torch.cuda.stream(streams[1]):
                handle_w.wait()
                handle_x.wait()
                cij1 = torch.matmul(x_recv_list[2], w1)
        case 4:
            with torch.cuda.stream(streams[1]):
                handle_w.wait()
                handle_x.wait()
                cij1 = torch.matmul(x_recv_list[5], w1)
        case 5:
            with torch.cuda.stream(streams[1]):
                handle_w.wait()
                handle_x.wait()
                cij1 = torch.matmul(x_recv_list[4], w1)
        case 6:
            with torch.cuda.stream(streams[1]):
                handle_w.wait()
                handle_x.wait()
                cij1 = torch.matmul(x_recv_list[7], w1)
        case 7:
            with torch.cuda.stream(streams[1]):
                handle_w.wait()
                handle_x.wait()
                cij1 = torch.matmul(x_recv_list[6], w1)

    for s in streams:
        s.synchronize()

    cij = cij0 + cij1
    return cij


def x_transpose_w_eightway_(
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
    empty = get_empty(device, torch.float32)

    # split upon last dimension
    x0, x1 = torch.split(xij, xij.shape[-1] // 2, dim=-1)
    x0, x1 = x0.contiguous(), x1.contiguous()

    match mod:
        case 0:
            x_send_list = [empty, x0, x1, x1, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 0),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 2),
                empty,
            ]
            w_send_list = [empty, empty, wij, empty, wij, empty, wij, empty]
            w_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 3),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 4),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 5),
                empty,
            ]
        case 1:
            x_send_list = [empty, empty, empty, empty, x0, x0, x1, x1]
            x_recv_list = [
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 0),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 2),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 3),
                empty,
            ]
            w_send_list = [empty, empty, empty, wij, empty, wij, empty, wij]
            w_recv_list = [
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 4),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 5),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 6),
            ]
        case 2:
            x_send_list = [x0, x0, empty, x1, empty, empty, empty, empty]
            x_recv_list = [
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 0),
                empty,
                empty,
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 2),
                empty,
            ]
            w_send_list = [wij, empty, empty, empty, wij, empty, wij, empty]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 3),
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 4),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 5),
                empty,
            ]
        case 3:
            x_send_list = [empty, empty, empty, empty, x0, x0, x1, x1]
            x_recv_list = [
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 0),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 2),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 3),
                empty,
            ]
            w_send_list = [empty, wij, empty, empty, empty, wij, empty, wij]
            w_recv_list = [
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 4),
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 5),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 6),
            ]
        case 4:
            x_send_list = [x0, x0, x1, x1, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 0),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 2),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 3),
            ]
            w_send_list = [wij, empty, wij, empty, empty, empty, wij, empty]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 4),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 5),
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 6),
                empty,
            ]
        case 5:
            x_send_list = [empty, empty, empty, empty, x0, empty, x1, x1]
            x_recv_list = [
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 0),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 1),
                empty,
                empty,
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 2),
            ]
            w_send_list = [empty, wij, empty, wij, empty, empty, empty, wij]
            w_recv_list = [
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 4),
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 5),
            ]
        case 6:
            x_send_list = [x0, x0, x1, x1, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 0),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 2),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 3),
            ]
            w_send_list = [wij, empty, wij, empty, wij, empty, empty, empty]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 4),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 5),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 6),
                empty,
                empty,
                empty,
            ]
        case 7:
            x_send_list = [empty, empty, empty, empty, x0, x0, x1, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 0),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 1),
                empty,
                get_preallocated_buffer(x0.shape, xij.dtype, local_rank, device, 2),
                empty,
                empty,
            ]
            w_send_list = [empty, wij, empty, wij, empty, wij, empty, empty]
            w_recv_list = [
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 4),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 5),
                empty,
                empty,
            ]

    x_handle = all_to_all_async(x_recv_list, x_send_list, group=model_group)
    w_handle = all_to_all_async(w_recv_list, w_send_list, group=model_group)

    x_handle.wait()

    match mod:
        case 0:
            cij = torch.matmul(x0.mT, wij)
        case 1:
            cij = torch.matmul(x_recv_list[0].mT, wij)
        case 2:
            cij = torch.matmul(x1.mT, wij)
        case 3:
            cij = torch.matmul(x_recv_list[2].mT, wij)
        case 4:
            cij = torch.matmul(x_recv_list[5].mT, wij)
        case 5:
            cij = torch.matmul(x0.mT, wij)
        case 6:
            cij = torch.matmul(x_recv_list[7].mT, wij)
        case 7:
            cij = torch.matmul(x1.mT, wij)

    w_handle.wait()

    match mod:
        case 0:
            cij = (
                cij
                + torch.matmul(x_recv_list[2].mT, w_recv_list[2])
                + torch.matmul(x_recv_list[4].mT, w_recv_list[4])
                + torch.matmul(x_recv_list[6].mT, w_recv_list[6])
            )
        case 1:
            cij = (
                cij
                + torch.matmul(x_recv_list[2].mT, w_recv_list[3])
                + torch.matmul(x_recv_list[4].mT, w_recv_list[5])
                + torch.matmul(x_recv_list[6].mT, w_recv_list[7])
            )
        case 2:
            cij = (
                cij
                + torch.matmul(x_recv_list[0].mT, w_recv_list[0])
                + torch.matmul(x_recv_list[4].mT, w_recv_list[4])
                + torch.matmul(x_recv_list[6].mT, w_recv_list[6])
            )
        case 3:
            cij = (
                cij
                + torch.matmul(x_recv_list[0].mT, w_recv_list[1])
                + torch.matmul(x_recv_list[4].mT, w_recv_list[5])
                + torch.matmul(x_recv_list[6].mT, w_recv_list[7])
            )
        case 4:
            cij = (
                cij
                + torch.matmul(x_recv_list[1].mT, w_recv_list[0])
                + torch.matmul(x_recv_list[3].mT, w_recv_list[2])
                + torch.matmul(x_recv_list[7].mT, w_recv_list[6])
            )
        case 5:
            cij = (
                cij
                + torch.matmul(x_recv_list[1].mT, w_recv_list[1])
                + torch.matmul(x_recv_list[3].mT, w_recv_list[3])
                + torch.matmul(x_recv_list[7].mT, w_recv_list[7])
            )
        case 6:
            cij = (
                cij
                + torch.matmul(x_recv_list[1].mT, w_recv_list[0])
                + torch.matmul(x_recv_list[3].mT, w_recv_list[2])
                + torch.matmul(x_recv_list[5].mT, w_recv_list[4])
            )
        case 7:
            cij = (
                cij
                + torch.matmul(x_recv_list[1].mT, w_recv_list[1])
                + torch.matmul(x_recv_list[3].mT, w_recv_list[3])
                + torch.matmul(x_recv_list[5].mT, w_recv_list[5])
            )

    return cij


def x_transpose_w_eightway(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Batched matrix-matrix (x_transposeW) multiplication distributed across eight processes.

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
    mod = local_rank % 8
    empty = get_empty(device, torch.float32)

    x0, x1 = torch.split(xij.mT, xij.shape[-1] // 2, dim=-2)
    x0, x1 = x0.contiguous(), x1.contiguous()

    x_transpose_shape = x0.shape
    streamz: list[torch.cuda.Stream] = [torch.cuda.Stream() for _ in range(2)]

    match mod:
        case 0:
            x_send_list = [empty, x0, x1, x1, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 2
                ),
                empty,
            ]
            w_send_list = [empty, empty, wij, empty, wij, empty, wij, empty]
            w_recv_list = [
                empty,
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 3),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 4),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 5),
                empty,
            ]
        case 1:
            x_send_list = [empty, empty, empty, empty, x0, x0, x1, x1]
            x_recv_list = [
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 2
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 3
                ),
                empty,
            ]
            w_send_list = [empty, empty, empty, wij, empty, wij, empty, wij]
            w_recv_list = [
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 4),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 5),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 6),
            ]
        case 2:
            x_send_list = [x0, x0, empty, x1, empty, empty, empty, empty]
            x_recv_list = [
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 0
                ),
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 2
                ),
                empty,
            ]
            w_send_list = [wij, empty, empty, empty, wij, empty, wij, empty]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 3),
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 4),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 5),
                empty,
            ]
        case 3:
            x_send_list = [empty, empty, empty, empty, x0, x0, x1, x1]
            x_recv_list = [
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 2
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 3
                ),
                empty,
            ]
            w_send_list = [empty, wij, empty, empty, empty, wij, empty, wij]
            w_recv_list = [
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 4),
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 5),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 6),
            ]
        case 4:
            x_send_list = [x0, x0, x1, x1, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 2
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 3
                ),
            ]
            w_send_list = [wij, empty, wij, empty, empty, empty, wij, empty]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 4),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 5),
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 6),
                empty,
            ]
        case 5:
            x_send_list = [empty, empty, empty, empty, x0, empty, x1, x1]
            x_recv_list = [
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 1
                ),
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 2
                ),
            ]
            w_send_list = [empty, wij, empty, wij, empty, empty, empty, wij]
            w_recv_list = [
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 4),
                empty,
                empty,
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 5),
            ]
        case 6:
            x_send_list = [x0, x0, x1, x1, empty, empty, empty, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 2
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 3
                ),
            ]
            w_send_list = [wij, empty, wij, empty, wij, empty, empty, empty]
            w_recv_list = [
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 4),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 5),
                empty,
                get_preallocated_buffer(wij.shape, xij.dtype, local_rank, device, 6),
                empty,
                empty,
                empty,
            ]
        case 7:
            x_send_list = [empty, empty, empty, empty, x0, x0, x1, empty]
            x_recv_list = [
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    x_transpose_shape, xij.dtype, local_rank, device, 2
                ),
                empty,
                empty,
            ]
            w_send_list = [empty, wij, empty, wij, empty, wij, empty, empty]
            w_recv_list = [
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 3),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 4),
                empty,
                get_preallocated_buffer(wij.shape, wij.dtype, local_rank, device, 5),
                empty,
                empty,
            ]

    x_handle = all_to_all_async(x_recv_list, x_send_list, group=model_group)
    w_handle = all_to_all_async(w_recv_list, w_send_list, group=model_group)

    match mod:
        case 0:
            with torch.cuda.stream(streamz[0]):
                x_handle.wait()
                cij0 = torch.matmul(x0, wij)
        case 1:
            with torch.cuda.stream(streamz[0]):
                x_handle.wait()
                cij0 = torch.matmul(x_recv_list[0], wij)
        case 2:
            with torch.cuda.stream(streamz[0]):
                x_handle.wait()
                cij0 = torch.matmul(x1, wij)
        case 3:
            with torch.cuda.stream(streamz[0]):
                x_handle.wait()
                cij0 = torch.matmul(x_recv_list[2], wij)
        case 4:
            with torch.cuda.stream(streamz[0]):
                x_handle.wait()
                cij0 = torch.matmul(x_recv_list[5], wij)
        case 5:
            with torch.cuda.stream(streamz[0]):
                x_handle.wait()
                cij0 = torch.matmul(x0, wij)
        case 6:
            with torch.cuda.stream(streamz[0]):
                x_handle.wait()
                cij0 = torch.matmul(x_recv_list[7], wij)
        case 7:
            with torch.cuda.stream(streamz[0]):
                x_handle.wait()
                cij0 = torch.matmul(x1, wij)

    for stream in streamz[1:]:
        with torch.cuda.stream(stream):
            w_handle.wait()

    match mod:
        case 0:
            with torch.cuda.stream(streamz[1]):
                cij1 = torch.matmul(x_recv_list[2], w_recv_list[2])
            with torch.cuda.stream(streamz[2]):
                cij2 = torch.matmul(x_recv_list[4], w_recv_list[4])
            with torch.cuda.stream(streamz[3]):
                cij3 = torch.matmul(x_recv_list[6], w_recv_list[6])
        case 1:
            with torch.cuda.stream(streamz[1]):
                cij1 = torch.matmul(x_recv_list[2], w_recv_list[3])
            with torch.cuda.stream(streamz[2]):
                cij2 = torch.matmul(x_recv_list[4], w_recv_list[5])
            with torch.cuda.stream(streamz[3]):
                cij3 = torch.matmul(x_recv_list[6], w_recv_list[7])
        case 2:
            with torch.cuda.stream(streamz[1]):
                cij1 = torch.matmul(x_recv_list[0], w_recv_list[0])
            with torch.cuda.stream(streamz[2]):
                cij2 = torch.matmul(x_recv_list[4], w_recv_list[4])
            with torch.cuda.stream(streamz[3]):
                cij3 = torch.matmul(x_recv_list[6], w_recv_list[6])
        case 3:
            with torch.cuda.stream(streamz[1]):
                cij1 = torch.matmul(x_recv_list[0], w_recv_list[1])
            with torch.cuda.stream(streamz[2]):
                cij2 = torch.matmul(x_recv_list[4], w_recv_list[5])
            with torch.cuda.stream(streamz[3]):
                cij3 = torch.matmul(x_recv_list[6], w_recv_list[7])
        case 4:
            with torch.cuda.stream(streamz[1]):
                cij1 = torch.matmul(x_recv_list[1], w_recv_list[0])
            with torch.cuda.stream(streamz[2]):
                cij2 = torch.matmul(x_recv_list[3], w_recv_list[2])
            with torch.cuda.stream(streamz[3]):
                cij3 = torch.matmul(x_recv_list[7], w_recv_list[6])
        case 5:
            with torch.cuda.stream(streamz[1]):
                cij1 = torch.matmul(x_recv_list[1], w_recv_list[1])
            with torch.cuda.stream(streamz[2]):
                cij2 = torch.matmul(x_recv_list[3], w_recv_list[3])
            with torch.cuda.stream(streamz[3]):
                cij3 = torch.matmul(x_recv_list[7], w_recv_list[7])
        case 6:
            with torch.cuda.stream(streamz[1]):
                cij1 = torch.matmul(x_recv_list[1], w_recv_list[0])
            with torch.cuda.stream(streamz[2]):
                cij2 = torch.matmul(x_recv_list[3], w_recv_list[2])
            with torch.cuda.stream(streamz[3]):
                cij3 = torch.matmul(x_recv_list[5], w_recv_list[4])
        case 7:
            with torch.cuda.stream(streamz[1]):
                cij1 = torch.matmul(x_recv_list[1], w_recv_list[1])
            with torch.cuda.stream(streamz[2]):
                cij2 = torch.matmul(x_recv_list[3], w_recv_list[3])
            with torch.cuda.stream(streamz[3]):
                cij3 = torch.matmul(x_recv_list[5], w_recv_list[5])

    for s in streamz:
        s.synchronize()

    cij = cij0 + cij1 + cij2 + cij3

    return cij


def fused_xw_xtw_eightway(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    input: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused calculation of grad_input and grad_weight in the backward pass of a linear layer.

    Computes ``grad_input = grad_output @ weight`` (as in :func:`xw_eightway`) and
    ``grad_weight = grad_output.mT @ input`` (as in :func:`x_transpose_w_eightway`)
    across eight processes using the ``all_to_all`` collective. Rather than running
    the two collective kernels back to back, this hand-merged kernel posts the
    ``all_to_all`` exchanges of *both* operations up front, so the communication of
    one overlaps the computation of the other. The per-rank exchange patterns and
    the multi-stream compute schedules are lifted verbatim from the two standalone
    kernels; only the interleaving is new.

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
    mod = local_rank % 8
    empty = get_empty(device, torch.float32)

    # ---- grad_input = grad_output @ weight setup (see xw_eightway) ----
    xw_streams: list[torch.cuda.Stream] = [torch.cuda.Stream() for _ in range(2)]
    w0 = get_preallocated_buffer(
        cast(
            torch.Size,
            (*weight.shape[:-2], weight.shape[-2] * 2, weight.shape[-1]),
        ),
        dtype=weight.dtype,
        local_rank=local_rank,
        device=device,
        index=1,
    )
    w1 = get_preallocated_buffer(
        cast(
            torch.Size,
            (*weight.shape[:-2], weight.shape[-2] * 2, weight.shape[-1]),
        ),
        dtype=weight.dtype,
        local_rank=local_rank,
        device=device,
        index=2,
    )
    split_size = w0.shape[-2] // 2

    # ---- grad_weight = grad_output.mT @ input setup (see x_transpose_w_eightway) ----
    xtw_streamz: list[torch.cuda.Stream] = [torch.cuda.Stream() for _ in range(4)]
    xt0, xt1 = torch.split(grad_output.mT, grad_output.shape[-1] // 2, dim=-2)
    xt0, xt1 = xt0.contiguous(), xt1.contiguous()
    xt_shape = xt0.shape

    match mod:
        case 0:
            xw_x_send = [empty, grad_output, empty, empty, empty, empty, empty, empty]
            xw_x_recv = [
                empty,
                get_preallocated_buffer(
                    grad_output.shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
            ]
            xw_w_send = [empty, empty, weight, empty, weight, empty, weight, empty]
            xw_w_recv = [
                empty,
                empty,
                w0[..., split_size:, :].contiguous(),
                empty,
                w1[..., :split_size, :].contiguous(),
                empty,
                w1[..., split_size:, :].contiguous(),
                empty,
            ]
            xtw_x_send = [empty, xt0, xt1, xt1, empty, empty, empty, empty]
            xtw_x_recv = [
                empty,
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 2
                ),
                empty,
            ]
            xtw_w_send = [empty, empty, input, empty, input, empty, input, empty]
            xtw_w_recv = [
                empty,
                empty,
                get_preallocated_buffer(
                    input.shape, grad_output.dtype, local_rank, device, 3
                ),
                empty,
                get_preallocated_buffer(
                    input.shape, grad_output.dtype, local_rank, device, 4
                ),
                empty,
                get_preallocated_buffer(
                    input.shape, grad_output.dtype, local_rank, device, 5
                ),
                empty,
            ]
        case 1:
            xw_x_send = [grad_output, empty, empty, empty, empty, empty, empty, empty]
            xw_x_recv = [
                get_preallocated_buffer(
                    grad_output.shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
            ]
            xw_w_send = [empty, empty, empty, weight, empty, weight, empty, weight]
            xw_w_recv = [
                empty,
                empty,
                empty,
                w1[..., split_size:, :].contiguous(),
                empty,
                w0[..., :split_size, :].contiguous(),
                empty,
                w0[..., split_size:, :].contiguous(),
            ]
            xtw_x_send = [empty, empty, empty, empty, xt0, xt0, xt1, xt1]
            xtw_x_recv = [
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 2
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 3
                ),
                empty,
            ]
            xtw_w_send = [empty, empty, empty, input, empty, input, empty, input]
            xtw_w_recv = [
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    input.shape, input.dtype, local_rank, device, 4
                ),
                empty,
                get_preallocated_buffer(
                    input.shape, input.dtype, local_rank, device, 5
                ),
                empty,
                get_preallocated_buffer(
                    input.shape, input.dtype, local_rank, device, 6
                ),
            ]
        case 2:
            xw_x_send = [empty, empty, empty, grad_output, empty, empty, empty, empty]
            xw_x_recv = [
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    grad_output.shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                empty,
                empty,
                empty,
            ]
            xw_w_send = [weight, empty, empty, empty, weight, empty, weight, empty]
            xw_w_recv = [
                w0[..., :split_size, :].contiguous(),
                empty,
                empty,
                empty,
                w1[..., :split_size, :].contiguous(),
                empty,
                w1[..., split_size:, :].contiguous(),
                empty,
            ]
            xtw_x_send = [xt0, xt0, empty, xt1, empty, empty, empty, empty]
            xtw_x_recv = [
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 2
                ),
                empty,
            ]
            xtw_w_send = [input, empty, empty, empty, input, empty, input, empty]
            xtw_w_recv = [
                get_preallocated_buffer(
                    input.shape, grad_output.dtype, local_rank, device, 3
                ),
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    input.shape, grad_output.dtype, local_rank, device, 4
                ),
                empty,
                get_preallocated_buffer(
                    input.shape, grad_output.dtype, local_rank, device, 5
                ),
                empty,
            ]
        case 3:
            xw_x_send = [empty, empty, grad_output, empty, empty, empty, empty, empty]
            xw_x_recv = [
                empty,
                empty,
                get_preallocated_buffer(
                    grad_output.shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                empty,
                empty,
                empty,
                empty,
            ]
            xw_w_send = [empty, weight, empty, empty, empty, weight, empty, weight]
            xw_w_recv = [
                empty,
                w1[..., :split_size, :],
                empty,
                empty,
                empty,
                w0[..., :split_size, :],
                empty,
                w0[..., split_size:, :],
            ]
            xtw_x_send = [empty, empty, empty, empty, xt0, xt0, xt1, xt1]
            xtw_x_recv = [
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 2
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 3
                ),
                empty,
            ]
            xtw_w_send = [empty, input, empty, empty, empty, input, empty, input]
            xtw_w_recv = [
                empty,
                get_preallocated_buffer(
                    input.shape, input.dtype, local_rank, device, 4
                ),
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    input.shape, input.dtype, local_rank, device, 5
                ),
                empty,
                get_preallocated_buffer(
                    input.shape, input.dtype, local_rank, device, 6
                ),
            ]
        case 4:
            xw_x_send = [empty, empty, empty, empty, empty, grad_output, empty, empty]
            xw_x_recv = [
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    grad_output.shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                empty,
            ]
            xw_w_send = [weight, empty, weight, empty, empty, empty, weight, empty]
            xw_w_recv = [
                w0[..., :split_size, :],
                empty,
                w0[..., split_size:, :],
                empty,
                empty,
                empty,
                w1[..., split_size:, :],
                empty,
            ]
            xtw_x_send = [xt0, xt0, xt1, xt1, empty, empty, empty, empty]
            xtw_x_recv = [
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 2
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 3
                ),
            ]
            xtw_w_send = [input, empty, input, empty, empty, empty, input, empty]
            xtw_w_recv = [
                get_preallocated_buffer(
                    input.shape, grad_output.dtype, local_rank, device, 4
                ),
                empty,
                get_preallocated_buffer(
                    input.shape, grad_output.dtype, local_rank, device, 5
                ),
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    input.shape, grad_output.dtype, local_rank, device, 6
                ),
                empty,
            ]
        case 5:
            xw_x_send = [empty, empty, empty, empty, grad_output, empty, empty, empty]
            xw_x_recv = [
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    grad_output.shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                empty,
                empty,
            ]
            xw_w_send = [empty, weight, empty, weight, empty, empty, empty, weight]
            xw_w_recv = [
                empty,
                w1[..., :split_size, :],
                empty,
                w1[..., split_size:, :],
                empty,
                empty,
                empty,
                w0[..., split_size:, :],
            ]
            xtw_x_send = [empty, empty, empty, empty, xt0, empty, xt1, xt1]
            xtw_x_recv = [
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 1
                ),
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 2
                ),
            ]
            xtw_w_send = [empty, input, empty, input, empty, empty, empty, input]
            xtw_w_recv = [
                empty,
                get_preallocated_buffer(
                    input.shape, input.dtype, local_rank, device, 3
                ),
                empty,
                get_preallocated_buffer(
                    input.shape, input.dtype, local_rank, device, 4
                ),
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    input.shape, input.dtype, local_rank, device, 5
                ),
            ]
        case 6:
            xw_x_send = [empty, empty, empty, empty, empty, empty, empty, grad_output]
            xw_x_recv = [
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    grad_output.shape, grad_output.dtype, local_rank, device, 0
                ),
            ]
            xw_w_send = [weight, empty, weight, empty, weight, empty, empty, empty]
            xw_w_recv = [
                w0[..., :split_size, :],
                empty,
                w0[..., split_size:, :],
                empty,
                w1[..., :split_size, :],
                empty,
                empty,
                empty,
            ]
            xtw_x_send = [xt0, xt0, xt1, xt1, empty, empty, empty, empty]
            xtw_x_recv = [
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 2
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 3
                ),
            ]
            xtw_w_send = [input, empty, input, empty, input, empty, empty, empty]
            xtw_w_recv = [
                get_preallocated_buffer(
                    input.shape, grad_output.dtype, local_rank, device, 4
                ),
                empty,
                get_preallocated_buffer(
                    input.shape, grad_output.dtype, local_rank, device, 5
                ),
                empty,
                get_preallocated_buffer(
                    input.shape, grad_output.dtype, local_rank, device, 6
                ),
                empty,
                empty,
                empty,
            ]
        case 7:
            xw_x_send = [empty, empty, empty, empty, empty, empty, grad_output, empty]
            xw_x_recv = [
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                get_preallocated_buffer(
                    grad_output.shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
            ]
            xw_w_send = [empty, weight, empty, weight, empty, weight, empty, empty]
            xw_w_recv = [
                empty,
                w1[..., :split_size, :],
                empty,
                w1[..., split_size:, :],
                empty,
                w0[..., :split_size, :],
                empty,
                empty,
            ]
            xtw_x_send = [empty, empty, empty, empty, xt0, xt0, xt1, empty]
            xtw_x_recv = [
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 0
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 1
                ),
                empty,
                get_preallocated_buffer(
                    xt_shape, grad_output.dtype, local_rank, device, 2
                ),
                empty,
                empty,
            ]
            xtw_w_send = [empty, input, empty, input, empty, input, empty, empty]
            xtw_w_recv = [
                empty,
                get_preallocated_buffer(
                    input.shape, input.dtype, local_rank, device, 3
                ),
                empty,
                get_preallocated_buffer(
                    input.shape, input.dtype, local_rank, device, 4
                ),
                empty,
                get_preallocated_buffer(
                    input.shape, input.dtype, local_rank, device, 5
                ),
                empty,
                empty,
            ]

    # Post the exchanges of both operations up front. NCCL serialises the four
    # collectives on its stream in submission order, so ``xw_w_handle`` still
    # completes first and waiting on a later handle transitively covers the
    # earlier ones -- the standalone kernels rely on the same ordering.
    xw_w_handle = all_to_all_async(xw_w_recv, xw_w_send, group=model_group)
    xw_x_handle = all_to_all_async(xw_x_recv, xw_x_send, group=model_group)
    xtw_x_handle = all_to_all_async(xtw_x_recv, xtw_x_send, group=model_group)
    xtw_w_handle = all_to_all_async(xtw_w_recv, xtw_w_send, group=model_group)

    # Local slice of ``weight`` that this rank keeps for its own ``grad_input``
    # contribution (disjoint from the slices filled by ``xw_w_handle``).
    match mod:
        case 0:
            w0[..., :split_size, :].copy_(weight)
        case 1:
            w1[..., :split_size, :].copy_(weight)
        case 2:
            w0[..., split_size:, :].copy_(weight)
        case 3:
            w0[..., split_size:, :].copy_(weight)
        case 4:
            w1[..., :split_size, :].copy_(weight)
        case 5:
            w0[..., :split_size, :].copy_(weight)
        case 6:
            w1[..., split_size:, :].copy_(weight)
        case 7:
            w0[..., split_size:, :].copy_(weight)

    # ---- grad_input (XW) computation ----
    with torch.cuda.stream(xw_streams[0]):
        xw_w_handle.wait()
        gi_cij0 = torch.matmul(grad_output, w0)
    match mod:
        case 0:
            with torch.cuda.stream(xw_streams[1]):
                xw_w_handle.wait()
                xw_x_handle.wait()
                gi_cij1 = torch.matmul(xw_x_recv[1], w1)
        case 1:
            with torch.cuda.stream(xw_streams[1]):
                xw_w_handle.wait()
                xw_x_handle.wait()
                gi_cij1 = torch.matmul(xw_x_recv[0], w1)
        case 2:
            with torch.cuda.stream(xw_streams[1]):
                xw_w_handle.wait()
                xw_x_handle.wait()
                gi_cij1 = torch.matmul(xw_x_recv[3], w1)
        case 3:
            with torch.cuda.stream(xw_streams[1]):
                xw_w_handle.wait()
                xw_x_handle.wait()
                gi_cij1 = torch.matmul(xw_x_recv[2], w1)
        case 4:
            with torch.cuda.stream(xw_streams[1]):
                xw_w_handle.wait()
                xw_x_handle.wait()
                gi_cij1 = torch.matmul(xw_x_recv[5], w1)
        case 5:
            with torch.cuda.stream(xw_streams[1]):
                xw_w_handle.wait()
                xw_x_handle.wait()
                gi_cij1 = torch.matmul(xw_x_recv[4], w1)
        case 6:
            with torch.cuda.stream(xw_streams[1]):
                xw_w_handle.wait()
                xw_x_handle.wait()
                gi_cij1 = torch.matmul(xw_x_recv[7], w1)
        case 7:
            with torch.cuda.stream(xw_streams[1]):
                xw_w_handle.wait()
                xw_x_handle.wait()
                gi_cij1 = torch.matmul(xw_x_recv[6], w1)

    for s in xw_streams:
        s.synchronize()

    grad_input = gi_cij0 + gi_cij1

    # ---- grad_weight (XtW) computation ----
    match mod:
        case 0:
            with torch.cuda.stream(xtw_streamz[0]):
                xtw_x_handle.wait()
                gw_cij0 = torch.matmul(xt0, input)
        case 1:
            with torch.cuda.stream(xtw_streamz[0]):
                xtw_x_handle.wait()
                gw_cij0 = torch.matmul(xtw_x_recv[0], input)
        case 2:
            with torch.cuda.stream(xtw_streamz[0]):
                xtw_x_handle.wait()
                gw_cij0 = torch.matmul(xt1, input)
        case 3:
            with torch.cuda.stream(xtw_streamz[0]):
                xtw_x_handle.wait()
                gw_cij0 = torch.matmul(xtw_x_recv[2], input)
        case 4:
            with torch.cuda.stream(xtw_streamz[0]):
                xtw_x_handle.wait()
                gw_cij0 = torch.matmul(xtw_x_recv[5], input)
        case 5:
            with torch.cuda.stream(xtw_streamz[0]):
                xtw_x_handle.wait()
                gw_cij0 = torch.matmul(xt0, input)
        case 6:
            with torch.cuda.stream(xtw_streamz[0]):
                xtw_x_handle.wait()
                gw_cij0 = torch.matmul(xtw_x_recv[7], input)
        case 7:
            with torch.cuda.stream(xtw_streamz[0]):
                xtw_x_handle.wait()
                gw_cij0 = torch.matmul(xt1, input)

    for stream in xtw_streamz[1:]:
        with torch.cuda.stream(stream):
            xtw_w_handle.wait()

    match mod:
        case 0:
            with torch.cuda.stream(xtw_streamz[1]):
                gw_cij1 = torch.matmul(xtw_x_recv[2], xtw_w_recv[2])
            with torch.cuda.stream(xtw_streamz[2]):
                gw_cij2 = torch.matmul(xtw_x_recv[4], xtw_w_recv[4])
            with torch.cuda.stream(xtw_streamz[3]):
                gw_cij3 = torch.matmul(xtw_x_recv[6], xtw_w_recv[6])
        case 1:
            with torch.cuda.stream(xtw_streamz[1]):
                gw_cij1 = torch.matmul(xtw_x_recv[2], xtw_w_recv[3])
            with torch.cuda.stream(xtw_streamz[2]):
                gw_cij2 = torch.matmul(xtw_x_recv[4], xtw_w_recv[5])
            with torch.cuda.stream(xtw_streamz[3]):
                gw_cij3 = torch.matmul(xtw_x_recv[6], xtw_w_recv[7])
        case 2:
            with torch.cuda.stream(xtw_streamz[1]):
                gw_cij1 = torch.matmul(xtw_x_recv[0], xtw_w_recv[0])
            with torch.cuda.stream(xtw_streamz[2]):
                gw_cij2 = torch.matmul(xtw_x_recv[4], xtw_w_recv[4])
            with torch.cuda.stream(xtw_streamz[3]):
                gw_cij3 = torch.matmul(xtw_x_recv[6], xtw_w_recv[6])
        case 3:
            with torch.cuda.stream(xtw_streamz[1]):
                gw_cij1 = torch.matmul(xtw_x_recv[0], xtw_w_recv[1])
            with torch.cuda.stream(xtw_streamz[2]):
                gw_cij2 = torch.matmul(xtw_x_recv[4], xtw_w_recv[5])
            with torch.cuda.stream(xtw_streamz[3]):
                gw_cij3 = torch.matmul(xtw_x_recv[6], xtw_w_recv[7])
        case 4:
            with torch.cuda.stream(xtw_streamz[1]):
                gw_cij1 = torch.matmul(xtw_x_recv[1], xtw_w_recv[0])
            with torch.cuda.stream(xtw_streamz[2]):
                gw_cij2 = torch.matmul(xtw_x_recv[3], xtw_w_recv[2])
            with torch.cuda.stream(xtw_streamz[3]):
                gw_cij3 = torch.matmul(xtw_x_recv[7], xtw_w_recv[6])
        case 5:
            with torch.cuda.stream(xtw_streamz[1]):
                gw_cij1 = torch.matmul(xtw_x_recv[1], xtw_w_recv[1])
            with torch.cuda.stream(xtw_streamz[2]):
                gw_cij2 = torch.matmul(xtw_x_recv[3], xtw_w_recv[3])
            with torch.cuda.stream(xtw_streamz[3]):
                gw_cij3 = torch.matmul(xtw_x_recv[7], xtw_w_recv[7])
        case 6:
            with torch.cuda.stream(xtw_streamz[1]):
                gw_cij1 = torch.matmul(xtw_x_recv[1], xtw_w_recv[0])
            with torch.cuda.stream(xtw_streamz[2]):
                gw_cij2 = torch.matmul(xtw_x_recv[3], xtw_w_recv[2])
            with torch.cuda.stream(xtw_streamz[3]):
                gw_cij3 = torch.matmul(xtw_x_recv[5], xtw_w_recv[4])
        case 7:
            with torch.cuda.stream(xtw_streamz[1]):
                gw_cij1 = torch.matmul(xtw_x_recv[1], xtw_w_recv[1])
            with torch.cuda.stream(xtw_streamz[2]):
                gw_cij2 = torch.matmul(xtw_x_recv[3], xtw_w_recv[3])
            with torch.cuda.stream(xtw_streamz[3]):
                gw_cij3 = torch.matmul(xtw_x_recv[5], xtw_w_recv[5])

    for s in xtw_streamz:
        s.synchronize()

    grad_weight = gw_cij0 + gw_cij1 + gw_cij2 + gw_cij3

    return grad_input, grad_weight
