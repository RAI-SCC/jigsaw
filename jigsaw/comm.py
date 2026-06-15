from typing import Any, Optional

import torch
import torch.distributed as dist


def comm_2way_alltoall(
    data: torch.Tensor,
    buffer: torch.Tensor,
    empty: torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = True,
) -> tuple[torch.Tensor, list[Optional[dist.Work]]]:
    """
    Send data to neighbor process using all-to-all.

    Parameters
    ----------
    data: torch.Tensor
        Data to send.
    buffer: torch.Tensor
        Buffer to receive.
    empty: torch.Tensor
        Empty for computational reasons.
    group: dist.ProcessGroup
        Process group within which to send data.
    async_op: bool
        Boolean flag to see if comm should be asynchronous.

    Returns
    -------
    buffer: torch.Tensor
        Buffer in which the data is received.
    wait_handle: list[Optional[dist.Work]]
        If async, a list of wait handles. In this function, the length of
        this list will always be 1. It is a list for compatability and extensibility reasons.
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
    """
    Send data to neighbor process using P2P communication.

    Parameters
    ----------
    data: torch.Tensor
        Data to send.
    buffer: torch.Tensor
        Buffer to receive.
    empty: torch.Tensor
        Empty for computational reasons.
    group: dist.ProcessGroup
        Process group within which to send data.
    async_op: bool
        Boolean flag to see if comm should be asynchronous.

    Returns
    -------
    buffer: torch.Tensor
        Buffer in which the data is received.
    wait_handle: list[Optional[dist.Work]]
        If async, a list of wait handles. In this function, the length of
        this list will always be 1. It is a list for compatability and extensibility reasons.
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
    direction: int,
    group: dist.ProcessGroup,
    async_op: bool,
    empty: torch.Tensor,
) -> tuple[torch.Tensor, Optional[dist.Work]]:
    """
    Send data to neighbor process using all to all communication.

    Parameters
    ----------
    data: torch.Tensor
        Data to send.
    buffer: torch.Tensor
        Buffer to receive.
    direction: int
        positive direction = send to right.
        negative direction = send to left.
    group: dist.ProcessGroup
        Process group within which to send data.
    empty: torch.Tensor
        Empty for computational reasons.

    Returns
    -------
    buffer: torch.Tensor
        Buffer in which the data is received.
    """
    if direction > 0:
        send_to = (group.rank() - 1) % group.size()
        recv_from = (group.rank() + 1) % group.size()
    else:
        send_to = (group.rank() + 1) % group.size()
        recv_from = (group.rank() - 1) % group.size()

    send_list = [empty] * group.size()
    recv_list = [empty] * group.size()
    send_list[send_to] = data
    recv_list[recv_from] = buffer

    handle = dist.all_to_all(recv_list, send_list, group=group, async_op=async_op)
    return buffer, handle


def comm_1d_p2p(
    data: torch.Tensor,
    buffer: torch.Tensor,
    offset: int,
    group: dist.ProcessGroup,
    async_op: bool = True,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, Optional[dist.Work]]:
    """
    Send data to neighbor process using all to all communication.

    Parameters
    ----------
    data: torch.Tensor
        Data to send.
    buffer: torch.Tensor
        Buffer to receive.
    direction: int
        positive direction = send to right.
        negative direction = send to left.
    group: dist.ProcessGroup
        Process group within which to send data.
    *args: Any
    **kwargs: Any

    Returns
    -------
    buffer: torch.Tensor
        Buffer in which the data is received.
    """
    send_to = (group.rank() + offset) % group.size()
    recv_from = (group.rank() - offset) % group.size()

    if group.rank() % 2 == 0:
        send = dist.isend(data, group_dst=send_to, group=group)
        recv = dist.irecv(buffer, group_src=recv_from, group=group)
        send.wait()
        if not async_op:
            recv.wait()
    elif group.rank() % 2 == 1:
        recv = dist.irecv(buffer, group_src=recv_from, group=group)
        send = dist.isend(data, group_dst=send_to, group=group)
        send.wait()
        if not async_op:
            recv.wait()

    return buffer, recv
