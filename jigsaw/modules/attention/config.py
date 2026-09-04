"""Config for attention."""

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
    """Enum for sending X or KV."""

    X = 0
    KV = 1


class SaveBackward(Enum):
    """Enum for saving X or KV."""

    X = 0
    KV = 1


class SaveScope(Enum):
    """Enum for saving X or KV locally or globally."""

    LOCAL = 0
    GLOBAL = 1


class Offload(Enum):
    """Enum for offloading to CPU."""

    DISABLED = 0
    CPU = 1


class SendBackward(Enum):
    """Enum for accumulating DKV in the backward pass."""

    SEND_DKV = 0
    GATHER_DKV = 1


class SendOp(Enum):
    """Enum for communication operation."""

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

    - ``TE``: TransformerEngine fused (cuDNN) kernels. CUDA + ``transformer_engine``
      only; the ring uses :mod:`.functional.cores.te`.
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
    """Build the activation offloader for the configured policy."""
    match offload:
        case Offload.DISABLED:
            return None
        case Offload.CPU:
            return NVLinkActivationOffloader(device)
        case other:
            raise ValueError(f"unhandled offload option {other}")


def _te_core_or_raise() -> None:
    """Guard: the TE fused-attention core needs ``transformer_engine`` + CUDA."""
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
    Cores live in :mod:`.functional.cores`; each exposes ``forward`` / ``backward``.
    """
    from .functional.cores import math as math_core
    from .functional.cores import sdpa as sdpa_core

    match AttnBackend(attn_backend):
        case AttnBackend.TE:
            _te_core_or_raise()
            from .functional.cores import te as te_core

            return te_core.forward
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
    """Pick the ring inner-attention backward core for ``device`` / ``attn_backend``."""
    from .functional.cores import math as math_core
    from .functional.cores import sdpa as sdpa_core

    match AttnBackend(attn_backend):
        case AttnBackend.TE:
            _te_core_or_raise()
            from .functional.cores import te as te_core

            return te_core.backward
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
    """Pick the primitive for the inter-node hop and the backward dK/dV relay.

    Point-to-point, except the all-to-all-flavoured ops. ``gloo`` has no
    all-to-all, so it always falls back to p2p.
    """
    if dist.get_backend() == "gloo":
        return comm.comm_1d_p2p
    alltoall = {SendOp.ALLTOALL, SendOp.BROADCAST, SendOp.HYBRID_ALLTOALL}
    return comm.comm_1d_alltoall if send_op in alltoall else comm.comm_1d_p2p


def _select_send_op_fn(send_op: SendOp) -> CommFn:
    """Pick the primitive for the ring's per-step send.

    ``BROADCAST`` uses a real broadcast; the all-to-all-flavoured ops (including
    both hybrid rings, whose intra-node step is a masked all-to-all) use
    ``comm_1d_alltoall``; everything else is p2p. ``gloo`` always falls back to
    p2p.
    """
    if dist.get_backend() == "gloo":
        return comm.comm_1d_p2p
    if send_op == SendOp.BROADCAST:
        return comm.comm_broadcast
    alltoall = {SendOp.ALLTOALL, SendOp.HYBRID_P2P, SendOp.HYBRID_ALLTOALL}
    return comm.comm_1d_alltoall if send_op in alltoall else comm.comm_1d_p2p


def _select_forward_send_fn(
    send_op: SendOp, send_in_forward: SendForward, send_op_fn: CommFn
) -> ForwardSendFn:
    """Return ``send(x, kv, **kw) -> (recv_buf, handle)`` for the forward ring.

    Two shapes: a broadcast send (``BROADCAST``) and a per-neighbour ring send
    (everything else -- the hybrid ops drive their own communication and only
    take this so config construction is uniform). Both send ``x`` or ``kv`` per
    ``send_in_forward``.
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
        return recv, dist.broadcast(recv, src=src_rank, group=group, async_op=True)

    return broadcast_send if send_op == SendOp.BROADCAST else ring_send


def _select_forward_recv_fn(send_in_forward: SendForward) -> ForwardRecvFn:
    """Return ``recv(recv_buf, xj, kv_j, kj, vj) -> (xj, kv_j, kj, vj)``.

    ``X``: the buffer is the new ``xj`` and K/V are recomputed by the caller.
    ``KV``: the buffer is the new ``kv_j``; ``kj / vj`` are its two halves.
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
    """Return ``save(x, kv) -> tensor`` capturing the chunk kept for backward.

    Picks ``x`` or ``kv`` per ``save_for_backward``, then: offload to CPU when an
    offloader is set; else ``.clone()`` for GLOBAL scope (the source buffer is
    overwritten by the next ring recv); else pass the tensor through untouched.
    ``save_for_backward`` values other than X / KV keep nothing.
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
    """Config class for ring attention.

    Bundles every policy decision (what to send, what to save, how to
    accumulate dK/dV, whether to offload) together with the concrete
    callables that implement each policy, so the forward/backward loops
    only ever call `self.config.<thing>` and never branch on raw enums
    themselves.
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
    ):
        """Initialize the object.

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

    def wait_offloads(self) -> None:
        """Block until any in-flight activation offloads have completed."""
        if self.offloader is not None:
            self.offloader.wait_offloads()

    def __repr__(self) -> str:
        """Return string representation of config class."""
        return (
            f"RingAttentionConfig("
            f"send_in_forward={self.send_in_forward.name}, "
            f"send_op={self.send_op.name}, "
            f"save_for_backward={self.save_for_backward.name}, "
            f"save_scope={self.save_scope.name}, "
            f"send_backward={self.send_backward.name}, "
            f"offload={self.offload.name}, "
            f"attn_backend={self.attn_backend.name})"
        )
