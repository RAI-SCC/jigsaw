"""Configuration of ring attention."""

from collections.abc import Callable
from enum import Enum, StrEnum
from typing import Any, TypeVar, cast

import torch
import torch.distributed as dist

import jigsaw._core.comm as comm_lib
from jigsaw._core.offload import NVLinkActivationOffloader


# --------------------------------------------------------------------------- #
# enums
# --------------------------------------------------------------------------- #
class SendForward(Enum):
    """Tensor communicated in the forward pass."""

    X = 0
    KV = 1


class SaveBackward(Enum):
    """Tensor saved for the backward pass."""

    X = 0
    KV = 1


class SaveScope(Enum):
    """Scope of the tensor saved for the backward pass."""

    LOCAL = 0
    GLOBAL = 1


class Offload(Enum):
    """Activation offloading policy."""

    DISABLED = 0
    CPU = 1


class SendBackward(Enum):
    """Strategy for accumulating dK and dV in the backward pass."""

    SEND_DKV = 0
    GATHER_DKV = 1


class SendOp(Enum):
    """Topology of the ring exchange.

    - ``RING``: each step every rank sends its chunk to its neighbor. Default.
    - ``BROADCAST``: each step one rank broadcasts its chunk to the ring.
    - ``HYBRID``: ``RING`` within intra groups of ``intra_size`` consecutive
      ranks, joined by an outer ring.
    - ``HYBRID_BROADCAST``: ``BROADCAST`` within the intra groups, joined by an
      outer ring.

    The transport of the sends is set separately by
    :class:`~jigsaw.modules.CommOp`.
    """

    RING = 0
    BROADCAST = 1
    HYBRID = 2
    HYBRID_BROADCAST = 3


class AttnBackend(StrEnum):
    """Attention core implementation, used both with and without ``ring``.

    - ``TE``: cuDNN fused kernels on Hopper and newer, the ``flash_attn``
      package on Ampere. CUDA only.
    - ``SDPA``: PyTorch scaled-dot-product attention on CUDA, the math kernel
      on CPU. Default.
    - ``MATH``: the unfused ``softmax(QK^T)V`` reference. Any device, shape and
      dtype.
    """

    TE = "te"
    SDPA = "sdpa"
    MATH = "math"


_E = TypeVar("_E", bound=Enum)


def _coerce_enum(enum_cls: type[_E], value: Any, arg: str) -> _E:
    """Return ``value`` as a member of ``enum_cls``.

    Accepts the member itself or its name as a string in any case
    (``send_op="hybrid"``), and for string enums also the value
    (``attn_backend="te"``). Anything else raises :class:`ValueError` rather
    than silently matching none of the policy branches.

    Parameters
    ----------
    enum_cls : type[Enum]
        The enum to convert to.
    value : Any
        The argument as passed by the caller.
    arg : str
        Argument name, for the error message.

    Returns
    -------
    Enum
        The matching member of ``enum_cls``.

    Raises
    ------
    ValueError
        If ``value`` names no member of ``enum_cls``.
    """
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        member = enum_cls.__members__.get(value.upper())
        if member is not None:
            return member
        if issubclass(enum_cls, str):
            try:
                return enum_cls(value)
            except ValueError:
                pass
    valid = ", ".join(repr(name.lower()) for name in enum_cls.__members__)
    raise ValueError(
        f"{arg}={value!r} is not a {enum_cls.__name__}; expected a "
        f"{enum_cls.__name__} member or one of {valid}."
    )


# --------------------------------------------------------------------------- #
# policy -> callable resolution
#
# Each strategy-specific callable on `RingAttentionConfig` comes from one
# `_select_*` function, so the forward and backward never branch on an enum.
# --------------------------------------------------------------------------- #
CommFn = Callable[..., Any]
AttnCoreFn = Callable[..., Any]
ForwardSendFn = Callable[..., tuple[torch.Tensor, dist.Work | None]]
ForwardRecvFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]
SaveForBackwardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor | None]


def _select_offloader(
    offload: Offload, device: torch.device
) -> NVLinkActivationOffloader | None:
    """Return the activation offloader for a policy, or ``None``."""
    match offload:
        case Offload.DISABLED:
            return None
        case Offload.CPU:
            return NVLinkActivationOffloader(device)
        case other:
            raise ValueError(f"unhandled offload option {other}")


def _te_core_or_raise() -> None:
    """Guard: the TE cores need ``transformer_engine`` or ``flash_attn``, plus CUDA.

    Either arm of :mod:`._functional.cores.te` satisfies this; which one actually
    runs is :func:`._functional.cores.te.select_core`'s call.
    """
    from ._functional.cores.te import TE_CORE_AVAILABLE

    if not TE_CORE_AVAILABLE:
        raise RuntimeError(
            "attn_backend=te needs transformer_engine or flash_attn "
            "(run inside the container)."
        )


def _select_attn_core_forward_fn(
    device: torch.device, attn_backend: AttnBackend = AttnBackend.SDPA
) -> AttnCoreFn:
    """Pick the ring inner-attention forward core for ``device`` / ``attn_backend``.

    ``SDPA`` picks by device (SDPA on CUDA, math on CPU), ``TE`` by GPU
    architecture (see :func:`._functional.cores.te.select_core`).
    """
    from ._functional.cores import math as math_core
    from ._functional.cores import sdpa as sdpa_core

    match AttnBackend(attn_backend):
        case AttnBackend.TE:
            _te_core_or_raise()
            from ._functional.cores import te as te_core

            return cast(AttnCoreFn, te_core.select_core(device).forward)
        case AttnBackend.MATH:
            return math_core.forward
        case AttnBackend.SDPA:
            match device.type:
                case "cuda":
                    return sdpa_core.forward
                case "cpu":
                    return math_core.forward
                case other:
                    raise ValueError(f"no attention forward core for device '{other}'")


def _select_attn_core_backward_fn(
    device: torch.device, attn_backend: AttnBackend = AttnBackend.SDPA
) -> AttnCoreFn:
    """Select the backward core of the ring's inner attention."""
    from ._functional.cores import math as math_core
    from ._functional.cores import sdpa as sdpa_core

    match AttnBackend(attn_backend):
        case AttnBackend.TE:
            _te_core_or_raise()
            from ._functional.cores import te as te_core

            return cast(AttnCoreFn, te_core.select_core(device).backward)
        case AttnBackend.MATH:
            return math_core.backward
        case AttnBackend.SDPA:
            match device.type:
                case "cuda":
                    return sdpa_core.backward
                case "cpu":
                    return math_core.backward
                case other:
                    raise ValueError(f"no attention backward core for device '{other}'")


def _select_send_op_fn(send_op: SendOp, ring_comm: comm_lib.CommOp) -> CommFn:
    """Select the communication function for the ring's per-step send."""
    if send_op == SendOp.BROADCAST:
        return comm_lib.comm_broadcast
    return comm_lib.select_comm_1d(ring_comm)


def _select_forward_send_fn(
    send_op: SendOp, send_in_forward: SendForward, send_op_fn: CommFn
) -> ForwardSendFn:
    """Select the send function of the forward ring.

    The returned callable has signature ``send(x, kv, **kw) -> (recv_buf, handle)``.
    """
    send_kv = send_in_forward == SendForward.KV

    def ring_send(
        x: torch.Tensor,
        kv: torch.Tensor,
        direction: int,
        group: dist.ProcessGroup,
        tag: int,
        group_size: int | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, dist.Work | None]:
        buf = kv if send_kv else x
        recv_buf, handle = send_op_fn(
            buf,
            torch.empty_like(buf),
            direction,
            group=group,
            async_op=True,
            tag=tag,
            group_size=group_size,
        )
        return recv_buf, handle

    def broadcast_send(
        x: torch.Tensor,
        kv: torch.Tensor,
        local_rank: int,
        group: dist.ProcessGroup,
        src_rank: int,
        **_: Any,
    ) -> tuple[torch.Tensor, dist.Work | None]:
        buf = kv if send_kv else x
        recv = buf if src_rank == local_rank else torch.empty_like(buf)
        return recv, dist.broadcast(
            recv, group_src=src_rank, group=group, async_op=True
        )

    return broadcast_send if send_op == SendOp.BROADCAST else ring_send


def _select_forward_recv_fn(send_in_forward: SendForward) -> ForwardRecvFn:
    """Select the receive function of the forward ring.

    The returned callable has signature
    ``recv(recv_buf, xj, kv_j, kj, vj) -> (xj, kv_j, kj, vj)``.
    """
    send_kv = send_in_forward == SendForward.KV

    def recv(
        recv_buf: torch.Tensor,
        xj: torch.Tensor,
        kv_j: torch.Tensor,
        kj: torch.Tensor,
        vj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if send_kv:
            return xj, recv_buf, recv_buf[0], recv_buf[1]
        return recv_buf, kv_j, kj, vj

    return recv


def _select_save_for_backward_fn(
    save_for_backward: SaveBackward,
    save_scope: SaveScope,
    offloader: NVLinkActivationOffloader | None,
) -> SaveForBackwardFn:
    """Select the function that captures the tensor kept for backward.

    The returned callable has signature ``save(x, kv) -> tensor``. It offloads
    the tensor if an offloader is set, clones it for ``GLOBAL`` scope, and
    otherwise returns it unchanged.
    """
    want_x = save_for_backward == SaveBackward.X
    if not (want_x or save_for_backward == SaveBackward.KV):
        return lambda x, kv: None

    def save(x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor | None:
        chunk = x if want_x else kv
        if isinstance(offloader, NVLinkActivationOffloader):
            return offloader.offload(chunk)
        if save_scope == SaveScope.GLOBAL:
            return chunk.clone()
        return chunk

    return save


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
class RingAttentionConfig:
    """Configuration of ring attention.

    Holds the communication, saving and offloading policies together with the
    functions that implement them.

    Parameters
    ----------
    device : torch.device
        Device on which tensors and communication are placed.
    send_in_forward : SendForward or str
        Tensor representation communicated during the forward pass.
    send_op : SendOp or str
        Topology of the ring exchange. ``RING`` (default) works for any ring
        size. The ``HYBRID*`` ops split the ring into intra groups of
        ``intra_size`` consecutive ranks joined by an outer ring, and need the
        ring size to be a multiple of ``intra_size``.
    save_for_backward : SaveBackward or str
        Tensor representation saved for the backward pass.
    save_scope : SaveScope or str
        Scope in which intermediate tensors are saved.
    send_backward : SendBackward or str
        Strategy used to communicate backward gradients.
    offload : Offload or str
        Activation offloading policy.
    intra_size : int
        Number of consecutive ring ranks in each intra group.
        Only used by the ``HYBRID*`` send ops.
    attn_backend : AttnBackend or str
        Inner per-chunk attention core. ``SDPA`` (default) keeps the
        device-based pick (flash on CUDA, math on CPU); ``TE`` uses the
        TransformerEngine fused core; ``MATH`` the unfused reference.
    overlap_comm : bool
        Put the ring's K/V circulation, and the backward dK/dV relay where
        the accumulation shape allows it, on a dedicated CUDA stream with
        double buffering, so each transfer overlaps the attention core on
        the default stream. CUDA only; elsewhere it is inert and the
        synchronous path runs unchanged. K/V circulation is overlapped for
        every ``send_op`` except ``BROADCAST`` (a fresh collective each
        step, nothing to prefetch ahead of); the dK/dV relay is overlapped
        too when ``save_scope=LOCAL`` and ``send_backward=SEND_DKV``,
        except under ``HYBRID_BROADCAST`` whose dK/dV accumulation doesn't
        fit the direct-swap relay. See :attr:`overlap_active`.
    comm : CommOp or str
        Transport of the ring's per-step sends (within the intra groups for
        the ``HYBRID*`` ops). ``"auto"`` picks all-to-all on CUDA and
        ``batch_isend_irecv`` on CPU. ``BROADCAST`` always broadcasts.
    outer_comm : CommOp or str, optional
        Transport of the outer ring of the ``HYBRID*`` ops and of the backward
        dK/dV relay. Defaults to ``comm``.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        send_in_forward: SendForward | str = SendForward.X,
        send_op: SendOp | str = SendOp.RING,
        save_for_backward: SaveBackward | str = SaveBackward.KV,
        save_scope: SaveScope | str = SaveScope.GLOBAL,
        send_backward: SendBackward | str = SendBackward.SEND_DKV,
        offload: Offload | str = Offload.DISABLED,
        intra_size: int = 4,
        attn_backend: AttnBackend | str = AttnBackend.SDPA,
        overlap_comm: bool = False,
        comm: comm_lib.CommOp | str = comm_lib.CommOp.AUTO,
        outer_comm: comm_lib.CommOp | str | None = None,
    ):
        """Initialize the configuration.

        Every enum argument also takes the member's name as a string, in any
        case (``send_op="hybrid"``); ``attn_backend`` also takes its value
        (``"te"``). Unknown names raise :class:`ValueError`.
        """
        if intra_size <= 0:
            raise ValueError("intra_size must be positive")
        send_in_forward = _coerce_enum(SendForward, send_in_forward, "send_in_forward")
        save_for_backward = _coerce_enum(
            SaveBackward, save_for_backward, "save_for_backward"
        )
        send_op = _coerce_enum(SendOp, send_op, "send_op")
        save_scope = _coerce_enum(SaveScope, save_scope, "save_scope")
        send_backward = _coerce_enum(SendBackward, send_backward, "send_backward")
        offload = _coerce_enum(Offload, offload, "offload")
        self.send_in_forward = send_in_forward
        self.save_for_backward = save_for_backward
        self.send_op = send_op
        self.save_scope = save_scope
        self.send_backward = send_backward
        self.offload = offload
        self.intra_size = intra_size
        self.device = device
        self.attn_backend = _coerce_enum(AttnBackend, attn_backend, "attn_backend")
        self.comm = comm_lib.resolve_comm_op(comm, device)
        self.outer_comm = (
            self.comm
            if outer_comm is None
            else comm_lib.resolve_comm_op(outer_comm, device)
        )

        # Dedicated CUDA stream for the ring's K/V circulation and backward dK/dV
        # relay. Created once here (the config outlives a single forward/backward)
        # and shared by both passes. CUDA-only: on CPU / meta devices there is no
        # stream and ``overlap_active`` stays False, so every provider falls back
        # to its synchronous path.
        self.overlap_comm = bool(overlap_comm)
        self._comm_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device)
            if self.overlap_comm and torch.device(device).type == "cuda"
            else None
        )

        # Layout the ring circulates q/k/v/out in, and hands to the inner core.
        # The TE cores (cuDNN / FA2) compute in bshd; carrying the ring in
        # bshd for them means zero transposes per chunk; SDPA / math are bhsd.
        self.core_layout = "bshd" if self.attn_backend == AttnBackend.TE else "bhsd"

        # Resolve every policy to its concrete callable, once.
        self.offloader = _select_offloader(offload, device)
        self.p2p_fn: CommFn = comm_lib.select_comm_1d(self.outer_comm)
        self.send_op_fn = _select_send_op_fn(send_op, self.comm)
        self.attn_core_forward_fn = _select_attn_core_forward_fn(
            device, self.attn_backend
        )
        self.attn_core_backward_fn = _select_attn_core_backward_fn(
            device, self.attn_backend
        )
        self.forward_send_fn = _select_forward_send_fn(
            send_op, send_in_forward, self.send_op_fn
        )
        self.forward_recv_fn = _select_forward_recv_fn(send_in_forward)
        self.save_for_backward_fn = _select_save_for_backward_fn(
            save_for_backward, save_scope, self.offloader
        )

    @property
    def comm_stream(self) -> torch.cuda.Stream | None:
        """Dedicated CUDA stream for ring comm, or ``None`` when overlap is off.

        Returns
        -------
        torch.cuda.Stream or None
            The comm stream, if overlap is active.
        """
        return self._comm_stream

    @property
    def overlap_active(self) -> bool:
        """Whether the comm-stream overlap path is actually engaged.

        ``True`` only when ``overlap_comm`` was requested *and* a CUDA stream
        could be built for it; provider builders gate on this so a non-CUDA
        device silently keeps the synchronous path.

        Returns
        -------
        bool
            Whether ring comm runs on :attr:`comm_stream`.
        """
        return self._comm_stream is not None

    def wait_offloads(self) -> None:
        """Wait for pending activation offloads."""
        if self.offloader is not None:
            self.offloader.wait_offloads()

    def __repr__(self) -> str:
        """Return a string representation of the configuration.

        Returns
        -------
        str
            The resolved policies and their values.
        """
        return (
            f"RingAttentionConfig("
            f"send_in_forward={self.send_in_forward.name}, "
            f"send_op={self.send_op.name}, "
            f"comm={self.comm.value}, "
            f"outer_comm={self.outer_comm.value}, "
            f"save_for_backward={self.save_for_backward.name}, "
            f"save_scope={self.save_scope.name}, "
            f"send_backward={self.send_backward.name}, "
            f"offload={self.offload.name}, "
            f"attn_backend={self.attn_backend.name}, "
            f"overlap_comm={self.overlap_comm})"
        )
