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


# --------------------------------------------------------------------------- #
# attention cores
# --------------------------------------------------------------------------- #
def _attn_core_flash(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, dropout_p: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flash attention forward pass. Returns (out, logsumexp)."""
    q, k, v = [x.to(torch.float16) for x in (q, k, v)]
    out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale,
        return_debug_mask=False,
    )

    return out, logsumexp


def _attn_core_math(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = 0.0,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vanilla attention forward pass. Returns (out, logsumexp)."""
    scale = scale or (q.size(-1) ** -0.5)
    scores = (q @ k.mT) * scale
    logsumexp = scores.logsumexp(dim=-1)  # (B, H, Sq)
    out = torch.softmax(scores, dim=-1) @ v
    return out, logsumexp


def _attn_core_flash_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    scale: float,
    max_q: float,
    max_k: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward call to flash attention. Returns (dq, dk, dv)."""
    dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        grad_out,
        q,
        k,
        v,
        out,
        logsumexp,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        cum_seq_q=None,
        cum_seq_k=None,
        max_q=max_q,
        max_k=max_k,
        philox_seed=torch.zeros(1, dtype=torch.long, device=q.device),
        philox_offset=torch.zeros(1, dtype=torch.long, device=q.device),
    )
    return dq, dk, dv


def _attn_core_math_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    scale: float,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vanilla attention backward pass using saved (global) LSE. Returns (dq, dk, dv)."""
    attn = (q @ k.mT) * scale  # (B, H, Sq, Sk)
    p = torch.exp(attn - logsumexp.unsqueeze(-1))  # global norm via saved lse

    dv = p.mT @ grad_out

    dp = grad_out @ v.mT
    d = (grad_out * out).sum(dim=-1, keepdim=True)  # GLOBAL row-sum, from saved out
    d_attn = p * (dp - d) * scale

    dq = d_attn @ k
    dk = d_attn.mT @ q
    return dq, dk, dv


# --------------------------------------------------------------------------- #
# policy -> callable resolution
#
# Every `RingAttentionConfig` attribute that holds a strategy-specific callable
# is produced by one `_select_*` function here, so the forward/backward code
# only ever calls `config.<thing>()` and never branches on a raw enum.
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


def _select_attn_core_forward_fn(device: torch.device) -> AttnCoreFn:
    """Pick the attention forward kernel for ``device``."""
    match device.type:
        case "cuda":
            return _attn_core_flash
        case "cpu":
            return _attn_core_math
        case other:
            raise ValueError(f"no attention forward core for device '{other}'")


def _select_attn_core_backward_fn(device: torch.device) -> AttnCoreFn:
    """Pick the attention backward kernel for ``device``."""
    match device.type:
        case "cuda":
            return _attn_core_flash_backward
        case "cpu":
            return _attn_core_math_backward
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
        empty: torch.Tensor,
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
            empty=empty,
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

        # Resolve every policy to its concrete callable, once.
        self.offloader = _select_offloader(offload, device)
        self.p2p_fn = _select_p2p_fn(send_op)
        self.send_op_fn = _select_send_op_fn(send_op)
        self.attn_core_forward_fn = _select_attn_core_forward_fn(device)
        self.attn_core_backward_fn = _select_attn_core_backward_fn(device)
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
            f"offload={self.offload.name})"
        )
