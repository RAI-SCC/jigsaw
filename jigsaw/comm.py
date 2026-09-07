"""Module for communication operations."""

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from jigsaw.offload import NVLinkActivationOffloader


@lru_cache(maxsize=None)
def get_empty(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a cached zero-length tensor for ``(device, dtype)``.

    ``dist.all_to_all`` needs a placeholder entry for every rank a given hop
    doesn't actually exchange data with. Those entries carry nothing, so a
    single shared empty tensor per ``(device, dtype)`` suffices; caching it here
    avoids threading one through every ring-attention call site and reallocating
    it on every hop.
    """
    return torch.empty(0, device=device, dtype=dtype)


def resolve_matmul_dtype(device: torch.device, *operands: torch.Tensor) -> torch.dtype:
    """Return the dtype a ``matmul`` over ``operands`` produces on ``device``.

    Mirrors the autocast handling in :mod:`jigsaw.modules.linear.functional`: an
    active autocast forces the op to run (and emit) in the autocast dtype,
    otherwise the result takes the operands' promoted dtype. Resolving it here
    lets a call site size the receive buffer for a matmul result *before*
    running the matmul that would otherwise be the only way to learn the dtype.
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
    direction: Optional[int] = None,
    send_recv_ranks: Optional[list[int]] = None,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = True,
    group_size: Optional[int] = None,
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
    async_op: bool
        Boolean flag for whether communication should be asynchronous or not.
    send_recv_ranks: list[int], optional
        Explicit send and receive rank pair.
    group_size: int, optional
        Size of the process group.
    args: Any
        Additional args
    kwargs: Any
        Additional kwargs

    Returns
    -------
    buffer: torch.Tensor
        Buffer in which the data is received.
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
    async_op: bool
        Boolean flag to see if operation is asynchronous or not.
    tag: int
        Identifier tag for p2p ops.
    send_recv_ranks: list[int], optional
        Explicit send and receive rank pair.
    group_size: int, optional
        Size of the process group.
    args: Any
        Additional args
    kwargs: Any
        Additional kwargs

    Returns
    -------
    buffer: torch.Tensor
        Buffer in which the data is received.
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
    # print(f'R{dist.get_rank()} send to {send_to} recv from {recv_from}. Group size {group_size}')
    tag = tag * 8

    send_op = dist.P2POp(dist.isend, data, group_peer=send_to, group=group)
    recv_op = dist.P2POp(dist.irecv, buffer, group_peer=recv_from, group=group)
    reqs = dist.batch_isend_irecv([send_op, recv_op])

    if dist.get_backend() == "gloo":
        dist.barrier()

    if not async_op:
        # send.wait()  # type: ignore[union-attr]
        reqs[0].wait()  # type: ignore[union-attr]
        return buffer, None
    else:
        # send.wait()  # type: ignore[union-attr]
        return buffer, reqs[0]


class _CpuInterP2PWork:
    """Duck-typed ``dist.Work`` for :func:`cpu_inter_p2p`'s async path.

    Wraps the in-flight Gloo ``isend`` / ``irecv`` plus the pending host->device
    copy of the received chunk. Callers only ever ``.wait()`` and then read the
    GPU ``buffer`` they passed in, so ``wait`` finishes the recv, stages it back
    onto the device (into that same ``buffer``), and orders the current CUDA
    stream after the copy.
    """

    def __init__(
        self,
        send_work: Optional[dist.Work],
        recv_work: Optional[dist.Work],
        recv_host: torch.Tensor,
        buffer: torch.Tensor,
        offloader: Optional["NVLinkActivationOffloader"],
    ) -> None:
        self._send_work = send_work
        self._recv_work = recv_work
        self._recv_host = recv_host
        self._buffer = buffer
        self._offloader = offloader
        self._done = False

    def wait(self, timeout: Optional[Any] = None) -> bool:
        """Complete the transfer and make ``buffer`` readable on the current stream."""
        if self._done:
            return True
        if self._recv_work is not None:
            self._recv_work.wait()
        # Blocking H2D: `.wait()` is called exactly when the caller needs the
        # chunk on-device, and a CPU sync here guarantees the pinned
        # `_recv_host` buffer is safe to drop once this returns (the async
        # call sites hold the work object; the sync ones discard it).
        if self._offloader is not None:
            event = self._offloader.prefetch_into(self._buffer, self._recv_host)
            event.synchronize()
        else:
            self._buffer.copy_(self._recv_host)
        if self._send_work is not None:
            self._send_work.wait()
        self._done = True
        return True

    def is_completed(self) -> bool:
        """Whether :meth:`wait` has already run."""
        return self._done


def cpu_inter_p2p(
    data: torch.Tensor,
    buffer: torch.Tensor,
    direction: Optional[int] = None,
    send_recv_ranks: Optional[list[int]] = None,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = True,
    tag: int = 0,
    group_size: Optional[int] = None,
    offloader: Optional["NVLinkActivationOffloader"] = None,
    **_: Any,
) -> tuple[torch.Tensor, Optional[_CpuInterP2PWork]]:
    """Send ``data`` to a ring peer over **Gloo, from pinned host memory**.

    Drop-in for :func:`comm_1d_p2p` / :func:`comm_1d_alltoall` on the ring's
    *inter-node* hop when ``RingAttentionConfig.inter_node_transport`` is
    ``CPU_GLOO``. ``data`` / ``buffer`` are GPU tensors exactly as the other
    primitives expect; this routine D2H-stages ``data`` to a pinned host buffer,
    exchanges it with the ``(send_to, recv_from)`` peers named in
    ``send_recv_ranks`` (group-relative ring positions) via ordered
    ``isend`` / ``irecv`` -- no ``batch_isend_irecv`` and no ``barrier``, which
    Gloo handles poorly -- and stages the received chunk back into ``buffer``.

    The transfer runs on the CPU + NIC and never touches a CUDA stream, so on an
    inter-node ring it frees the GPU copy engines and the NVLink-C2C link for the
    concurrent activation-offload traffic.

    Parameters
    ----------
    data : torch.Tensor
        GPU tensor to send.
    buffer : torch.Tensor
        GPU tensor the received chunk is written into.
    direction : int, optional
        Unused; accepted for signature parity with :func:`comm_1d_p2p`.
    send_recv_ranks : list[int]
        ``(send_to, recv_from)`` as ranks within ``group``. Required.
    group : dist.ProcessGroup
        The ring process group. Required.
    async_op : bool
        If True, return a :class:`_CpuInterP2PWork` to ``.wait()`` later;
        otherwise block until the exchange and the host->device copy are done.
    tag, group_size : Any
        Unused; accepted for signature parity.
    offloader : NVLinkActivationOffloader, optional
        Supplies the pinned D2H / H2D staging. When ``None`` a plain blocking
        copy is used (test / fallback path).

    Returns
    -------
    tuple[torch.Tensor, _CpuInterP2PWork | None]
        ``buffer`` and, for ``async_op``, its pending-work handle.
    """
    if group is None:
        raise ValueError("group is required")
    if send_recv_ranks is None:
        raise ValueError("send_recv_ranks is required for cpu_inter_p2p")
    send_to, recv_from = send_recv_ranks

    pin = torch.cuda.is_available()
    if offloader is not None:
        send_host = offloader.offload_blocking(data)
    else:
        send_host = torch.empty(
            data.shape, dtype=data.dtype, pin_memory=pin, device="cpu"
        )
        send_host.copy_(data)
    recv_host = torch.empty(
        buffer.shape, dtype=buffer.dtype, pin_memory=pin, device="cpu"
    )

    # Ordered isend/irecv (never batched) with a rank-parity launch order so the
    # exchange can't deadlock -- the same pattern as `comm_2way_p2p`.
    if dist.get_rank(group) % 2 == 0:
        send_work = dist.isend(send_host, group_dst=send_to, group=group)
        recv_work = dist.irecv(recv_host, group_src=recv_from, group=group)
    else:
        recv_work = dist.irecv(recv_host, group_src=recv_from, group=group)
        send_work = dist.isend(send_host, group_dst=send_to, group=group)

    work = _CpuInterP2PWork(send_work, recv_work, recv_host, buffer, offloader)
    if async_op:
        return buffer, work
    work.wait()
    return buffer, None


def comm_broadcast(
    data: torch.Tensor,
    src: int,
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
    src: int
        Source rank for the broadcast.
    group: dist.ProcessGroup
        Process group within which to send data.
    async_op: bool
        Boolean flag to see if operation is asynchronous or not.
    tag: int
        Identifier tag for p2p ops.
    args: Any
        Additional args
    kwargs: Any
        Additional kwargs

    Returns
    -------
    buffer: torch.Tensor
        Buffer in which the data is received.
    """
    handle = dist.broadcast(data, src=src, group=group, async_op=async_op)
    return data, handle
