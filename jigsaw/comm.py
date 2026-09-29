"""Communication primitives."""

from functools import lru_cache
from typing import Any, Optional

import torch
import torch.distributed as dist


def all_to_all_async(
    output_tensor_list: list[torch.Tensor],
    input_tensor_list: list[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
) -> dist.Work:
    """Issue an asynchronous ``dist.all_to_all`` and return its work handle.

    Parameters
    ----------
    output_tensor_list : list[torch.Tensor]
        Buffers receiving the data of each rank.
    input_tensor_list : list[torch.Tensor]
        Data sent to each rank.
    group : dist.ProcessGroup, optional
        Process group of the exchange.

    Returns
    -------
    dist.Work
        Handle to wait on.
    """
    work: Optional[dist.Work] = dist.all_to_all(
        output_tensor_list, input_tensor_list, group=group, async_op=True
    )
    assert work is not None
    return work


@lru_cache(maxsize=None)
def get_empty(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a cached zero-length tensor for ``(device, dtype)``.

    Used as the placeholder entry for ranks that exchange no data in an
    ``all_to_all``.

    Parameters
    ----------
    device : torch.device
        Device of the tensor.
    dtype : torch.dtype
        Data type of the tensor.

    Returns
    -------
    torch.Tensor
        Shared empty tensor.
    """
    return torch.empty(0, device=device, dtype=dtype)


def resolve_matmul_dtype(device: torch.device, *operands: torch.Tensor) -> torch.dtype:
    """Return the dtype a ``matmul`` over ``operands`` produces on ``device``.

    This is the autocast dtype if autocast is active, otherwise the promoted
    dtype of the operands.

    Parameters
    ----------
    device : torch.device
        Device the matmul runs on.
    *operands : torch.Tensor
        Matmul operands.

    Returns
    -------
    torch.dtype
        Result dtype.
    """
    if torch.is_autocast_enabled(device.type):
        return torch.get_autocast_dtype(device.type)
    dtype = operands[0].dtype
    for operand in operands[1:]:
        dtype = torch.promote_types(dtype, operand.dtype)
    return dtype


def comm_2way_alltoall(
    data: torch.Tensor,
    buffer: torch.Tensor,
    empty: torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = True,
) -> tuple[torch.Tensor, list[Optional[dist.Work]]]:
    """Send data to a neighbor process using all-to-all.

    Parameters
    ----------
    data : torch.Tensor
        Data to send.
    buffer : torch.Tensor
        Buffer to receive.
    empty : torch.Tensor
        Placeholder tensor for ranks that exchange no data.
    group : dist.ProcessGroup
        Process group within which to send data.
    async_op : bool
        Whether to communicate asynchronously.

    Returns
    -------
    buffer : torch.Tensor
        Buffer in which the data is received.
    wait_handle : list[dist.Work | None]
        Wait handles if asynchronous. Always has length 1.
    """
    match group.rank():
        case 0:
            wait_handle = dist.all_to_all(
                [empty, buffer], [empty, data], group=group, async_op=async_op
            )
            return buffer, [wait_handle]
        case 1:
            wait_handle = dist.all_to_all(
                [buffer, empty], [data, empty], group=group, async_op=async_op
            )
            return buffer, [wait_handle]
        case _:
            raise ValueError(
                f"Rank of {group.rank()} not valid for build_halo_comm_42ay."
            )


def comm_2way_p2p(
    data: torch.Tensor,
    buffer: torch.Tensor,
    empty: torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = True,
) -> tuple[torch.Tensor, list[Optional[dist.Work]]]:
    """Send data to a neighbor process using P2P communication.

    Parameters
    ----------
    data : torch.Tensor
        Data to send.
    buffer : torch.Tensor
        Buffer to receive.
    empty : torch.Tensor
        Placeholder tensor for ranks that exchange no data.
    group : dist.ProcessGroup
        Process group within which to send data.
    async_op : bool
        Whether to communicate asynchronously.

    Returns
    -------
    buffer : torch.Tensor
        Buffer in which the data is received.
    wait_handle : list[dist.Work | None]
        Wait handles if asynchronous. Always has length 1.
    """
    match group.rank():
        case 0:
            if async_op:
                send = dist.isend(data, group_dst=1, group=group)
                recv = dist.irecv(buffer, group_src=1, group=group)
                return buffer, [send, recv]
            else:
                dist.send(data, group_dst=1, group=group)
                dist.recv(buffer, group_src=1, group=group)
                return buffer, [None, None]

        case 1:
            if async_op:
                recv = dist.irecv(buffer, group_src=0, group=group)
                send = dist.isend(data, group_dst=0, group=group)
                return buffer, [send, recv]
            else:
                dist.recv(buffer, group_src=0, group=group)
                dist.send(data, group_dst=0, group=group)
                return buffer, [None, None]
        case _:
            raise ValueError(
                f"Rank of {group.rank()} not valid for build_halo_comm_4way."
            )


def comm_1d_alltoall(
    data: torch.Tensor,
    buffer: torch.Tensor,
    direction: Optional[int] = None,
    send_recv_ranks: Optional[list[int]] = None,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = True,
    group_size: Optional[int] = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, Optional[dist.Work]]:
    """Send data to a neighbor process using all-to-all.

    Parameters
    ----------
    data : torch.Tensor
        Data to send.
    buffer : torch.Tensor
        Buffer to receive.
    direction : int, optional
        Positive sends to the right, negative to the left. Required if
        ``send_recv_ranks`` is omitted.
    send_recv_ranks : list[int], optional
        Explicit send and receive rank pair.
    group : dist.ProcessGroup
        Process group within which to send data.
    async_op : bool
        Whether to communicate asynchronously.
    group_size : int, optional
        Size of the process group.
    *args : Any
        Unused.
    **kwargs : Any
        Unused.

    Returns
    -------
    buffer : torch.Tensor
        Buffer in which the data is received.
    handle : dist.Work | None
        Wait handle if asynchronous.
    """
    if group is None:
        raise ValueError("group is required")
    group_size = group.size() if group_size is None else group_size
    empty = get_empty(data.device, data.dtype)

    if send_recv_ranks is None:
        if direction is None:
            raise ValueError("direction is required when send_recv_ranks is omitted")
        send_to = (group.rank() + direction) % group_size
        recv_from = (group.rank() - direction) % group_size
    else:
        send_to, recv_from = send_recv_ranks

    send_list = [empty] * group.size()
    recv_list = [empty] * group.size()
    send_list[send_to] = data
    recv_list[recv_from] = buffer

    handle = dist.all_to_all(recv_list, send_list, group=group, async_op=async_op)
    return buffer, handle


def comm_1d_p2p(
    data: torch.Tensor,
    buffer: torch.Tensor,
    direction: Optional[int] = None,
    send_recv_ranks: Optional[list[int]] = None,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = True,
    tag: int = 0,
    group_size: Optional[int] = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, Optional[dist.Work]]:
    """Send data to a neighbor process using P2P communication.

    Parameters
    ----------
    data : torch.Tensor
        Data to send.
    buffer : torch.Tensor
        Buffer to receive.
    direction : int, optional
        Positive sends to the right, negative to the left. Required if
        ``send_recv_ranks`` is omitted.
    send_recv_ranks : list[int], optional
        Explicit send and receive rank pair.
    group : dist.ProcessGroup
        Process group within which to send data.
    async_op : bool
        Whether to communicate asynchronously.
    tag : int
        Tag for P2P operations.
    group_size : int, optional
        Size of the process group.
    *args : Any
        Unused.
    **kwargs : Any
        Unused.

    Returns
    -------
    buffer : torch.Tensor
        Buffer in which the data is received.
    handle : dist.Work | None
        Wait handle if asynchronous.
    """
    if group is None:
        raise ValueError("group is required")
    group_size = group.size() if group_size is None else group_size

    if send_recv_ranks is None:
        if direction is None:
            raise ValueError("direction is required when send_recv_ranks is omitted")
        send_to = (group.rank() + direction) % group_size
        recv_from = (group.rank() - direction) % group_size
    else:
        send_to, recv_from = send_recv_ranks
    tag = tag * 8

    send_op = dist.P2POp(dist.isend, data, group_peer=send_to, group=group)
    recv_op = dist.P2POp(dist.irecv, buffer, group_peer=recv_from, group=group)
    reqs = dist.batch_isend_irecv([send_op, recv_op])

    if dist.get_backend() == "gloo":
        dist.barrier()

    if not async_op:
        reqs[0].wait()  # type: ignore[union-attr]
        return buffer, None
    else:
        return buffer, reqs[0]


def comm_broadcast(
    data: torch.Tensor,
    src: int,
    group: dist.ProcessGroup,
    async_op: bool = True,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, Optional[dist.Work]]:
    """Broadcast data from a source rank.

    Parameters
    ----------
    data : torch.Tensor
        Data to send.
    src : int
        Source rank for the broadcast.
    group : dist.ProcessGroup
        Process group within which to send data.
    async_op : bool
        Whether to communicate asynchronously.
    *args : Any
        Unused.
    **kwargs : Any
        Unused.

    Returns
    -------
    data : torch.Tensor
        Broadcast tensor.
    handle : dist.Work | None
        Wait handle if asynchronous.
    """
    handle = dist.broadcast(data, src=src, group=group, async_op=async_op)
    return data, handle
