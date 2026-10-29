"""Communication primitives."""

from collections.abc import Callable, Iterable
from enum import StrEnum
from functools import cache
from typing import Any

import torch
import torch.distributed as dist

# --- Backend checks ---------------------------------------------------------------

_BACKEND_DEVICE_TYPES = {"nccl": "cuda", "gloo": "cpu"}


def check_backend(group: dist.ProcessGroup | None = None) -> set[str]:
    """Return ``group``'s backend names, raising if jigsaw does not support one.

    ``"nccl"`` covers both NVIDIA NCCL and AMD RCCL (ROCm builds of PyTorch
    register RCCL under that name).

    Parameters
    ----------
    group : dist.ProcessGroup, optional
        Process group; the default group if ``None``.

    Returns
    -------
    set[str]
        The backend names, e.g. ``{"gloo", "nccl"}`` for ``"cpu:gloo,cuda:nccl"``.

    Raises
    ------
    RuntimeError
        If the group's backend is not one jigsaw supports.
    """
    backend = str(dist.get_backend(group))
    # A plain "nccl" or a per-device list "cpu:gloo,cuda:nccl".
    names = {entry.rpartition(":")[2] for entry in backend.split(",")}
    unsupported = names - _BACKEND_DEVICE_TYPES.keys()
    if unsupported:
        raise RuntimeError(
            f"unsupported process-group backend {backend!r}; jigsaw supports "
            f"{sorted(_BACKEND_DEVICE_TYPES)}"
        )
    return names


def get_device_of_process_group(
    group: dist.ProcessGroup | None = None,
) -> torch.device:
    """Return the device collectives over ``group`` need their tensors on.

    A group with several backends (e.g. ``"cpu:gloo,cuda:nccl"``) resolves to
    CUDA if any of them is NCCL.

    Parameters
    ----------
    group : dist.ProcessGroup, optional
        Process group; the default group if ``None``.

    Returns
    -------
    torch.device
        The current CUDA device for NCCL, the CPU for Gloo.

    Raises
    ------
    RuntimeError
        If the group's backend is not one jigsaw supports.
    """
    if "nccl" in check_backend(group):
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


# --- Transport selection ----------------------------------------------------------


class CommOp(StrEnum):
    """Transport of the neighbor exchanges of a distributed layer.

    - ``AUTO``: ``ALL_TO_ALL`` on CUDA, ``BATCH_ISEND_IRECV`` on CPU. Default.
    - ``ALL_TO_ALL``: one ``dist.all_to_all`` per exchange. CUDA only, since
      gloo's all-to-all rejects the zero-length entries of the idle peers.
    - ``ISEND``: separate ``dist.isend`` and ``dist.irecv``. CUDA tensors take
      ``BATCH_ISEND_IRECV`` instead, since separate NCCL sends around a ring
      deadlock.
    - ``BATCH_ISEND_IRECV``: one ``dist.batch_isend_irecv`` per exchange, which
      NCCL coalesces into a single launch.

    Strings take the value in any case (``"all_to_all"``).
    """

    AUTO = "auto"
    ALL_TO_ALL = "all_to_all"
    ISEND = "isend"
    BATCH_ISEND_IRECV = "batch_isend_irecv"


def resolve_comm_op(comm: CommOp | str, device: torch.device) -> CommOp:
    """Resolve ``comm`` to a concrete :class:`CommOp` for ``device``.

    Parameters
    ----------
    comm : CommOp or str
        Requested transport; ``AUTO`` picks by device.
    device : torch.device
        Device of the module.

    Returns
    -------
    CommOp
        ``ALL_TO_ALL``, ``ISEND`` or ``BATCH_ISEND_IRECV``.

    Raises
    ------
    ValueError
        If ``comm`` is unknown, ``device`` is neither CUDA nor CPU, or
        ``ALL_TO_ALL`` is requested off CUDA.
    """
    if device.type not in ("cuda", "cpu"):
        raise ValueError(f"Unsupported device {device.type}")
    if not isinstance(comm, CommOp):
        key = str(comm).lower()
        if key not in {m.value for m in CommOp}:
            raise ValueError(
                f"comm must be one of {[m.value for m in CommOp]}, got {comm!r}"
            )
        comm = CommOp(key)
    if comm is CommOp.AUTO:
        return CommOp.ALL_TO_ALL if device.type == "cuda" else CommOp.BATCH_ISEND_IRECV
    if comm is CommOp.ALL_TO_ALL and device.type != "cuda":
        raise ValueError(
            f"comm='all_to_all' needs CUDA (NCCL), got device {device.type!r}; "
            "use 'isend' or 'batch_isend_irecv'"
        )
    return comm


#: Signature shared by the 1-D neighbor exchanges:
#: ``fn(data, buffer, direction=, send_recv_ranks=, group=, async_op=)``.
Comm1dFn = Callable[..., tuple[torch.Tensor, dist.Work | None]]


def select_comm_1d(comm: CommOp) -> Comm1dFn:
    """Return the 1-D neighbor exchange of a resolved ``comm``.

    Parameters
    ----------
    comm : CommOp
        Resolved transport (see :func:`resolve_comm_op`).

    Returns
    -------
    Comm1dFn
        :func:`comm_1d_alltoall`, :func:`comm_1d_isend` or
        :func:`comm_1d_batch_isend_irecv`.
    """
    return _COMM_1D_FNS[comm]


# --- 1-D neighbor exchanges -------------------------------------------------------


@cache
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


def _peers(
    group: dist.ProcessGroup | None,
    direction: int | None,
    send_recv_ranks: list[int] | None,
    group_size: int | None,
) -> tuple[dist.ProcessGroup, int, int]:
    """Return ``(group, send_to, recv_from)`` of a 1-D exchange.

    Raises
    ------
    ValueError
        If ``group`` is missing, or both ``direction`` and ``send_recv_ranks`` are.
    """
    if group is None:
        raise ValueError("group is required")
    if send_recv_ranks is not None:
        send_to, recv_from = send_recv_ranks
        return group, send_to, recv_from
    if direction is None:
        raise ValueError("direction is required when send_recv_ranks is omitted")
    size = group.size() if group_size is None else group_size
    return group, (group.rank() + direction) % size, (group.rank() - direction) % size


class _BatchWork:
    """Wait handle over the works of one send/receive pair."""

    def __init__(self, works: list[dist.Work]) -> None:
        self._works = works

    def wait(self, *args: Any, **kwargs: Any) -> bool:
        """Block until every send and receive of the batch has completed."""
        for work in self._works:
            work.wait(*args, **kwargs)
        return True

    def is_completed(self) -> bool:
        """Return whether every send and receive of the batch has completed."""
        return all(work.is_completed() for work in self._works)


def _finish(
    buffer: torch.Tensor, works: list[dist.Work], async_op: bool
) -> tuple[torch.Tensor, dist.Work | None]:
    """Wrap the p2p ``works`` in one handle, or wait on them if not ``async_op``."""
    handle = _BatchWork(works)
    if not async_op:
        handle.wait()
        return buffer, None
    return buffer, handle  # type: ignore[return-value]


def comm_1d_alltoall(
    data: torch.Tensor,
    buffer: torch.Tensor,
    direction: int | None = None,
    send_recv_ranks: list[int] | None = None,
    group: dist.ProcessGroup | None = None,
    async_op: bool = True,
    group_size: int | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, dist.Work | None]:
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

    Raises
    ------
    ValueError
        If ``group`` is missing, or both ``direction`` and ``send_recv_ranks`` are.
    """
    group, send_to, recv_from = _peers(group, direction, send_recv_ranks, group_size)
    empty = get_empty(data.device, data.dtype)

    # ``dist.all_to_all`` needs contiguous tensors, and callers may pass a
    # strided view; ``.contiguous()`` is a no-op otherwise.
    data = data.contiguous()
    buffer = buffer.contiguous()

    send_list = [empty] * group.size()
    recv_list = [empty] * group.size()
    send_list[send_to] = data
    recv_list[recv_from] = buffer

    handle = dist.all_to_all(recv_list, send_list, group=group, async_op=async_op)
    return buffer, handle


def comm_1d_isend(
    data: torch.Tensor,
    buffer: torch.Tensor,
    direction: int | None = None,
    send_recv_ranks: list[int] | None = None,
    group: dist.ProcessGroup | None = None,
    async_op: bool = True,
    group_size: int | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, dist.Work | None]:
    """Send data to a neighbor process using separate ``isend`` and ``irecv``.

    CUDA tensors go through :func:`comm_1d_batch_isend_irecv`: an NCCL send
    outside a group blocks until its peer joins, so a ring of separate sends
    deadlocks with every rank waiting in its own send.

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

    Raises
    ------
    ValueError
        If ``group`` is missing, or both ``direction`` and ``send_recv_ranks`` are.
    """
    if data.is_cuda:
        return comm_1d_batch_isend_irecv(
            data, buffer, direction, send_recv_ranks, group, async_op, group_size
        )
    group, send_to, recv_from = _peers(group, direction, send_recv_ranks, group_size)
    # ``dist.isend`` needs a contiguous tensor, and callers may pass a strided view.
    send = dist.isend(data.contiguous(), group_dst=send_to, group=group)
    recv = dist.irecv(buffer, group_src=recv_from, group=group)
    return _finish(buffer, [w for w in (send, recv) if w is not None], async_op)


def comm_1d_batch_isend_irecv(
    data: torch.Tensor,
    buffer: torch.Tensor,
    direction: int | None = None,
    send_recv_ranks: list[int] | None = None,
    group: dist.ProcessGroup | None = None,
    async_op: bool = True,
    group_size: int | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, dist.Work | None]:
    """Send data to a neighbor process using one ``dist.batch_isend_irecv``.

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

    Raises
    ------
    ValueError
        If ``group`` is missing, or both ``direction`` and ``send_recv_ranks`` are.
    """
    group, send_to, recv_from = _peers(group, direction, send_recv_ranks, group_size)
    # ``dist.isend`` needs a contiguous tensor, and callers may pass a strided view.
    send_op = dist.P2POp(dist.isend, data.contiguous(), group_peer=send_to, group=group)
    recv_op = dist.P2POp(dist.irecv, buffer, group_peer=recv_from, group=group)
    # NCCL coalesces the batch into one work; gloo returns one per op. Wait on all
    # of them, or the receive may still be in flight when ``buffer`` is read.
    return _finish(buffer, dist.batch_isend_irecv([send_op, recv_op]), async_op)


_COMM_1D_FNS: dict[CommOp, Comm1dFn] = {
    CommOp.ALL_TO_ALL: comm_1d_alltoall,
    CommOp.ISEND: comm_1d_isend,
    CommOp.BATCH_ISEND_IRECV: comm_1d_batch_isend_irecv,
}


# --- Collectives ------------------------------------------------------------------


def all_to_all_async(
    output_tensor_list: list[torch.Tensor],
    input_tensor_list: list[torch.Tensor],
    group: dist.ProcessGroup | None = None,
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
    work: dist.Work | None = dist.all_to_all(
        output_tensor_list, input_tensor_list, group=group, async_op=True
    )
    assert work is not None
    return work


def comm_broadcast(
    data: torch.Tensor,
    src: int,
    group: dist.ProcessGroup,
    async_op: bool = True,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, dist.Work | None]:
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


def all_reduce_max(t: torch.Tensor, groups: Iterable[dist.ProcessGroup]) -> None:
    """All-reduce ``t`` with MAX over each group in turn, in place.

    Parameters
    ----------
    t : torch.Tensor
        Tensor to reduce, in place.
    groups : Iterable[dist.ProcessGroup]
        Groups to reduce over, one after another.
    """
    for g in groups:
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=g)
