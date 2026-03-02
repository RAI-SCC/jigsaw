import torch
import torch.distributed as dist

from . import distributed_matmul as dist_mm

streams = [torch.cuda.Stream() for _ in range(2)]
streamz = [torch.cuda.Stream() for _ in range(4)]

_prealloc_buffers = {}


def xw_transpose_columnwise_twoway(
    xij: torch.tensor,
    wij: torch.tensor,
    rank: int,
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
    rank : int
        Global rank.
    model_group : dist.ProcessGroup, optional
        Process group for an individual model-parallel instance.

    Returns
    -------
    cij : torch.tensor
        Local result.
    """
    mod = rank % 2
    empty = torch.empty(0, device=device, dtype=torch.float32)

    w0, w1 = torch.split(wij, wij.shape[-2] // 2, dim=-2)
    buffer_xw = dist_mm._get_preallocated_buffer(
        (*xij.shape[:-1], w0.shape[-2]), torch.float32, rank, device
    )

    match mod:
        case 0:
            xw = torch.matmul(xij, w1.mT)
            xw_send_list = [empty, xw]
            xw_recv_list = [empty, buffer_xw]
        case 1:
            xw = torch.matmul(xij, w0.mT)
            xw_send_list = [xw, empty]
            xw_recv_list = [buffer_xw, empty]

    xw_handle = dist.all_to_all(
        xw_recv_list, xw_send_list, group=model_group, async_op=True
    )

    match mod:
        case 0:
            cij = torch.matmul(xij, w0.mT)
            xw_handle.wait()
            cij = cij + xw_recv_list[1]
        case 1:
            cij = torch.matmul(xij, w1.mT)
            xw_handle.wait()
            cij = cij + xw_recv_list[0]

    return cij
