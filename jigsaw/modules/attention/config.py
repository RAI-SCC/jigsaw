"""Configuration of ring attention."""

import warnings
from enum import Enum
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

import jigsaw.comm as comm
from jigsaw.offload import NVLinkActivationOffloader


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
    """Communication operation of the ring."""

    ALLTOALL = 0
    P2P = 1
    BROADCAST = 2
    HYBRID_P2P = 3
    HYBRID_BROADCAST = 4
    HYBRID_ALLTOALL = 5


class AttnBackend(str, Enum):
    """Attention core implementation.

    Applies to both the plain :class:`~..attention.DistributedAttention` core and
    the ring's per-chunk inner attention.

    - ``TE``: whichever kernel TransformerEngine would run on this GPU -- the
      cuDNN fused kernels on Hopper and newer, the ``flash_attn`` package on
      Ampere. CUDA only; see :mod:`.functional.cores.te`.
    - ``SDPA``: PyTorch scaled-dot-product attention -- the flash kernel on CUDA
      (with a math fallback), the math kernel on CPU. Default.
    - ``MATH``: the unfused ``softmax(QK^T)V`` reference. Any device/shape/dtype.

    ``ring`` is a separate switch (see ``DistributedAttention``); it composes with
    every backend here.
    """

    TE = "te"
    SDPA = "sdpa"
    MATH = "math"


# --------------------------------------------------------------------------- #
# policy -> callable resolution
#
# Every `RingAttentionConfig` attribute that holds a strategy-specific callable
# is produced by one `_select_*` function here, so the forward/backward code
# only ever calls `config.<thing>()` and never branches on a raw enum. The
# attention cores themselves live in `functional/cores/{sdpa,math,te}.py`.
# --------------------------------------------------------------------------- #
CommFn = Callable[..., Any]
AttnCoreFn = Callable[..., Any]
ForwardSendFn = Callable[..., tuple[torch.Tensor, Optional[dist.Work]]]
ForwardRecvFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]
SaveForBackwardFn = Callable[[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]


def _select_offloader(
    offload: Offload, device: torch.device
) -> Optional[NVLinkActivationOffloader]:
    """Build the activation offloader for the configured policy.

    ``Offload.CPU`` stages GPU activations to pinned host memory over CUDA side
    streams; with no CUDA device there is nothing to stage and no stream to
    build, so it degrades to a no-op (``None``) with a warning rather than
    failing in :class:`~jigsaw.offload.NVLinkActivationOffloader`'s constructor.
    """
    match offload:
        case Offload.DISABLED:
            return None
        case Offload.CPU:
            if device.type != "cuda":
                warnings.warn(
                    "Offload.CPU has no effect without a CUDA device; "
                    "activation offloading disabled.",
                    stacklevel=2,
                )
                return None
            return NVLinkActivationOffloader(device)
        case other:
            raise ValueError(f"unhandled offload option {other}")


def _te_core_or_raise() -> None:
    """Guard: the TE cores need ``transformer_engine`` or ``flash_attn``, plus CUDA.

    Either arm of :mod:`.functional.cores.te` satisfies this; which one actually
    runs is :func:`.functional.cores.te.select_core`'s call.
    """
    from .functional.cores.te import TE_CORE_AVAILABLE

    if not TE_CORE_AVAILABLE:
        raise RuntimeError(
            "attn_backend=te needs transformer_engine (run inside the container)."
        )


def _select_attn_core_forward_fn(
    device: torch.device, attn_backend: AttnBackend = AttnBackend.SDPA
) -> AttnCoreFn:
    """Pick the ring inner-attention forward core for ``device`` / ``attn_backend``.

    ``SDPA`` keeps the historical device-based pick (flash on CUDA, math on CPU).
    ``TE`` picks by GPU architecture -- cuDNN on Hopper and newer, ``flash_attn``
    on Ampere; see :func:`.functional.cores.te.select_core`.
    Cores live in :mod:`.functional.cores`; each exposes ``forward`` / ``backward``.
    """
    from .functional.cores import math as math_core
    from .functional.cores import sdpa as sdpa_core

    match AttnBackend(attn_backend):
        case AttnBackend.TE:
            _te_core_or_raise()
            from .functional.cores import te as te_core

            return te_core.select_core(device).forward
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
    from .functional.cores import math as math_core
    from .functional.cores import sdpa as sdpa_core

    match AttnBackend(attn_backend):
        case AttnBackend.TE:
            _te_core_or_raise()
            from .functional.cores import te as te_core

            return te_core.select_core(device).backward
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


def _select_p2p_fn(send_op: SendOp) -> CommFn:
    """Select the communication function for inter-node hops and the dK/dV relay."""
    if dist.get_backend() == "gloo":
        return comm.comm_1d_p2p
    alltoall = {SendOp.ALLTOALL, SendOp.BROADCAST, SendOp.HYBRID_ALLTOALL}
    return comm.comm_1d_alltoall if send_op in alltoall else comm.comm_1d_p2p


def _select_send_op_fn(send_op: SendOp) -> CommFn:
    """Select the communication function for the ring's per-step send."""
    if dist.get_backend() == "gloo":
        return comm.comm_1d_p2p
    if send_op == SendOp.BROADCAST:
        return comm.comm_broadcast
    alltoall = {SendOp.ALLTOALL, SendOp.HYBRID_P2P, SendOp.HYBRID_ALLTOALL}
    return comm.comm_1d_alltoall if send_op in alltoall else comm.comm_1d_p2p


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
        group_size: Optional[int] = None,
        **_: Any,
    ) -> tuple[torch.Tensor, Optional[dist.Work]]:
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
    ) -> tuple[torch.Tensor, Optional[dist.Work]]:
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
    offloader: Optional[NVLinkActivationOffloader],
) -> SaveForBackwardFn:
    """Select the function that captures the tensor kept for backward.

    The returned callable has signature ``save(x, kv) -> tensor``. It offloads
    the tensor if an offloader is set, clones it for ``GLOBAL`` scope, and
    otherwise returns it unchanged.
    """
    want_x = save_for_backward == SaveBackward.X
    if not (want_x or save_for_backward == SaveBackward.KV):
        return lambda x, kv: None

    def save(x: torch.Tensor, kv: torch.Tensor) -> Optional[torch.Tensor]:
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
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        send_in_forward: SendForward = SendForward.X,
        send_op: SendOp = SendOp.HYBRID_P2P,
        save_for_backward: SaveBackward = SaveBackward.KV,
        save_scope: SaveScope = SaveScope.GLOBAL,
        send_backward: SendBackward = SendBackward.SEND_DKV,
        offload: Offload = Offload.DISABLED,
        gpus_per_node: int = 4,
        attn_backend: AttnBackend = AttnBackend.SDPA,
        causal: bool = False,
        overlap_comm: bool = False,
    ):
        """Initialize the configuration.

        Parameters
        ----------
        device : torch.device
            Device on which tensors and communication are placed.
        send_in_forward : SendForward
            Tensor representation communicated during the forward pass.
        send_op : SendOp
            Communication operation used by the ring.
        save_for_backward : SaveBackward
            Tensor representation saved for the backward pass.
        save_scope : SaveScope
            Scope in which intermediate tensors are saved.
        send_backward : SendBackward
            Strategy used to communicate backward gradients.
        offload : Offload
            Activation offloading policy.
        gpus_per_node : int
            Number of ranks assigned to each node for hybrid communication.
        attn_backend : AttnBackend
            Inner per-chunk attention core. ``SDPA`` (default) keeps the
            device-based pick (flash on CUDA, math on CPU); ``TE`` uses the
            TransformerEngine fused core; ``MATH`` the unfused reference.
        causal : bool
            Whether the ring performs causal (autoregressive) attention. Each
            rank's query chunk then only attends to K/V chunks at or before
            its own global position: chunks strictly in the future are
            skipped outright, the one chunk at the same global position (the
            "diagonal" block) is masked causally, and every chunk strictly in
            the past is attended in full -- see :func:`.forward.ring_forward`
            and :func:`.backward.ring_backward`.
        overlap_comm : bool
            Put the ring's K/V circulation -- and the backward dK/dV relay,
            where the accumulation shape allows it -- on a dedicated CUDA
            stream with double buffering, so each transfer overlaps the
            attention core on the default stream (the ``cp_stream`` trick
            Transformer Engine uses). CUDA only; elsewhere it is inert and the
            synchronous path runs unchanged. K/V circulation is overlapped for
            every ``send_op`` except ``BROADCAST`` (a fresh collective each
            step, nothing to prefetch ahead of); the dK/dV relay is overlapped
            too when ``save_scope=LOCAL`` and ``send_backward=SEND_DKV``,
            except under ``HYBRID_BROADCAST`` whose dK/dV accumulation doesn't
            fit the direct-swap relay. See :attr:`overlap_active`.
        """
        if gpus_per_node <= 0:
            raise ValueError("gpus_per_node must be positive")
        self.send_in_forward = send_in_forward
        self.save_for_backward = save_for_backward
        self.send_op = send_op
        self.save_scope = save_scope
        self.send_backward = send_backward
        self.offload = offload
        self.gpus_per_node = gpus_per_node
        self.device = device
        self.attn_backend = AttnBackend(attn_backend)
        self.causal = causal

        # Dedicated CUDA stream for the ring's K/V circulation and backward dK/dV
        # relay. Created once here (the config outlives a single forward/backward)
        # and shared by both passes. CUDA-only: on CPU / meta devices there is no
        # stream and ``overlap_active`` stays False, so every provider falls back
        # to its synchronous path.
        self.overlap_comm = bool(overlap_comm)
        self._comm_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device)
            if self.overlap_comm and torch.device(device).type == "cuda"
            else None
        )

        # Layout the ring circulates q/k/v/out in, and hands to the inner core.
        # The TE cores (cuDNN / FA2) are natively bshd -- carrying the ring in
        # bshd for them means zero transposes per chunk; SDPA / math are bhsd.
        self.core_layout = "bshd" if self.attn_backend == AttnBackend.TE else "bhsd"

        # Resolve every policy to its concrete callable, once.
        self.offloader = _select_offloader(offload, device)
        self.p2p_fn = _select_p2p_fn(send_op)
        self.send_op_fn = _select_send_op_fn(send_op)
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
    def comm_stream(self) -> Optional[torch.cuda.Stream]:
        """Dedicated CUDA stream for ring comm, or ``None`` when overlap is off."""
        return self._comm_stream

    @property
    def overlap_active(self) -> bool:
        """Whether the comm-stream overlap path is actually engaged.

        ``True`` only when ``overlap_comm`` was requested *and* a CUDA stream
        could be built for it; provider builders gate on this so a non-CUDA
        device silently keeps the synchronous path.
        """
        return self._comm_stream is not None

    def wait_offloads(self) -> None:
        """Wait for pending activation offloads."""
        if self.offloader is not None:
            self.offloader.wait_offloads()

    def __repr__(self) -> str:
        """Return a string representation of the configuration."""
        return (
            f"RingAttentionConfig("
            f"send_in_forward={self.send_in_forward.name}, "
            f"send_op={self.send_op.name}, "
            f"save_for_backward={self.save_for_backward.name}, "
            f"save_scope={self.save_scope.name}, "
            f"send_backward={self.send_backward.name}, "
            f"offload={self.offload.name}, "
            f"attn_backend={self.attn_backend.name}, "
            f"causal={self.causal}, "
            f"overlap_comm={self.overlap_comm})"
        )
