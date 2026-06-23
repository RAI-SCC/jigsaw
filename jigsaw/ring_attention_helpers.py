"""Ring attention forward-backward implementations."""

from collections.abc import Callable
from enum import Enum
from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from . import comm
from .offload import NVLinkActivationOffloader


def _online_softmax_merge(
    o1: torch.Tensor, l1: torch.Tensor, o2: torch.Tensor, l2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    l_new = torch.logaddexp(l1, l2)
    e1 = torch.exp(l1 - l_new)
    e2 = torch.exp(l2 - l_new)
    o_new = e1.unsqueeze(-1) * o1 + e2.unsqueeze(-1) * o2
    return o_new, l_new


def _compute_kv(
    xj: torch.Tensor, w_qkv: torch.Tensor, b: int, n_local: int, h: int, d: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (K, V) from X. Returns (KV, Kj, Vj). KV is the underlying buffer."""
    kv = (
        (xj @ w_qkv[d * h :, :].mT).reshape(b, n_local, 2, h, d).permute(2, 0, 3, 1, 4)
    ).contiguous()
    kj = kv[0]
    vj = kv[1]
    return kv, kj, vj


# Module-level enum definitions
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
    """Enum for backward communication."""

    SEND_DKV = 0
    GATHER_DKV = 1


class SendForwardOp(Enum):
    """Enum for forward communication operation."""

    SENDRECV = 0
    BROADCAST = 1


class RingAttentionConfig:
    """Config class for ring attention."""

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        send_in_forward: SendForward = SendForward.KV,
        send_forward_op: SendForwardOp = SendForwardOp.SENDRECV,
        save_for_backward: SaveBackward = SaveBackward.KV,
        save_scope: SaveScope = SaveScope.LOCAL,
        send_backward: SendBackward = SendBackward.SEND_DKV,
        offload: Offload = Offload.DISABLED,
    ):
        self.send_in_forward = send_in_forward
        self.save_for_backward = save_for_backward
        self.send_forward_op = send_forward_op
        self.save_scope = save_scope
        self.send_backward = send_backward
        self.offload = offload
        self.offloader = self.init_offloader(offload, device)
        self.device = device
        self.ring_comm_fn = self._select_ring_comm_fn(device)
        self.attn_core_forward_fn = self._select_attn_core_forward_fn(device)
        self.attn_core_backward_fn = self._select_attn_core_backward_fn(device)
        self.forward_send_fn, self.forward_recv_fn = _select_forward_comm_fns(
            send_forward_op, send_in_forward, self.ring_comm_fn
        )
        self.save_for_backward_fn = _select_save_for_backward_fn(
            save_for_backward, save_scope, self.offloader
        )

    def init_offloader(
        self, offload: Offload, device: torch.device
    ) -> Optional[NVLinkActivationOffloader]:
        """Initialize offloader based on config."""
        if offload == Offload.DISABLED:
            return None
        elif offload == Offload.CPU:
            return NVLinkActivationOffloader(device)

    def _select_ring_comm_fn(self, device: torch.device) -> Callable:
        if device.type == "cuda":
            return comm.comm_1d_alltoall
        elif device.type == "cpu":
            return comm.comm_1d_p2p
        else:
            raise ValueError(f"No valid ring comm functions for device {device.type}")

    def _select_attn_core_forward_fn(self, device: torch.device) -> Callable:
        if device.type == "cuda":
            return _attn_core_flash
        elif device.type == "cpu":
            return _attn_core_math
        else:
            raise ValueError(
                f"No valid attention core forward for device {device.type}"
            )

    def _select_attn_core_backward_fn(self, device: torch.device) -> Callable:
        if device.type == "cuda":
            return _attn_core_flash_backward
        elif device.type == "cpu":
            return _attn_core_math_backward
        else:
            raise ValueError(
                f"No valid attention core backward for device {device.type}"
            )

    def __repr__(self) -> str:
        """Print statement of RingAttentionConfig class."""
        return (
            f"RingAttentionConfig("
            f"send_in_forward={self.send_in_forward.name}, "
            f"send_forward_op={self.send_forward_op.name}, "
            f"save_for_backward={self.save_for_backward.name}, "
            f"save_scope={self.save_scope.name}, "
            f"send_backward={self.send_backward.name}, "
            f"offload={self.offload.name})"
        )


def _attn_core_flash(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, dropout_p: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Call flash attention forward pass.

    q: torch.Tensor
        Queries.
    k: torch.Tensor
        Keys.
    v: torch.Tensor
        Values.
    scale: float
        Scale parameter used to normalize attention score.
    dropout_p: float
        Dropout probability.

    Returns
    -------
    out: torch.Tensor
        Result of attention.
    logsumexp: torch.Tensor
        Intermediate result for online softmax algorithm.
    """
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
    """
    Call vanilla attention forward pass.

    q: torch.Tensor
        Queries.
    k: torch.Tensor
        Keys.
    v: torch.Tensor
        Values.
    scale: float
        Scale parameter used to normalize attention score.
    dropout_p: float
        Dropout probability.

    Returns
    -------
    out: torch.Tensor
        Result of attention.
    logsumexp: torch.Tensor
        Intermediate result for online softmax algorithm.
    """
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
    """
    Backward call to flash attention.

    dout: torch.Tensor
        Gradient wrt output.
    q: torch.Tensor
        Queries.
    k: torch.Tensor
        Keys.
    v: torch.Tensor
        Values.
    out: torch.Tensor
        Output from forward pass.
    logsumexp: torch.Tensor
        Logsumexp from forward pass.
    scale: float
        Scale parameter for normalizing Attention.
    max_q: float
        Max length of q.
    max_k: float
        Max length of k.
    """
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
    """
    Backward call to flash attention.

    dout: torch.Tensor
        Gradient wrt output.
    q: torch.Tensor
        Queries.
    k: torch.Tensor
        Keys.
    v: torch.Tensor
        Values.
    out: torch.Tensor
        Output from forward pass.
    logsumexp: torch.Tensor
        Logsumexp from forward pass.
    scale: float
        Scale parameter for normalizing Attention.
    args: Any
    kwargs: Any
    """
    attn = (q @ k.mT) * scale  # (B, H, Sq, Sk)
    p = torch.exp(attn - logsumexp.unsqueeze(-1))  # global norm via saved lse

    dv = p.mT @ grad_out

    dp = grad_out @ v.mT
    d = (grad_out * out).sum(dim=-1, keepdim=True)  # GLOBAL row-sum, from saved out
    d_attn = p * (dp - d) * scale

    dq = d_attn @ k
    dk = d_attn.mT @ q
    return dq, dk, dv


class RingCtx(Protocol):
    """Protocol for types stored within ctx."""

    empty: torch.Tensor
    scale: float
    local_rank: int
    group: dist.ProcessGroup
    world_size: int
    config: RingAttentionConfig
    saved_tensors: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple
    ]


class RingAttentionFunction(torch.autograd.Function):
    """Ring Attention autograd function."""

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        scale: float,
        dropout_p: float,
        group: dist.ProcessGroup,
        empty: torch.Tensor,
        local_rank: int,
        config: RingAttentionConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Forward pass of ring attention."""
        b, h, n_local, d = q.shape

        world_size = dist.get_world_size(group)
        out = torch.zeros_like(q)
        lse = torch.full((b, n_local, h), float("-inf"), device=q.device, dtype=q.dtype)
        empty = torch.empty(0, device=x.device, dtype=x.dtype)

        ring_forward = build_ring_forward(config)
        out, lse, chunks_to_save = ring_forward(
            q,
            k,
            v,
            x,
            w_qkv,
            scale,
            dropout_p,
            group,
            empty,
            local_rank,
            world_size,
            config,
        )

        return out, lse, chunks_to_save

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: Any,
        output: Any,
    ) -> None:
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, w, b, local_rank, device, config, spatial_group, bayesian)
        output: xwt
        """
        q, k, v, x, w_qkv, scale, group, empty, local_rank, world_size, config = inputs

        out, lse, chunks_to_save = output
        ctx.mark_non_differentiable(lse)
        ctx.mark_non_differentiable(*chunks_to_save)

        ctx.save_for_backward(
            q, out, x, lse.mT, w_qkv, *[t for pair in chunks_to_save for t in pair]
        )
        c = cast(RingCtx, ctx)

        c.empty = empty
        c.scale = scale
        c.local_rank = local_rank
        c.group = group
        c.world_size = world_size
        c.config = config

    @staticmethod
    def backward(
        ctx: RingCtx,
        dout: torch.Tensor,
        grad_l: torch.Tensor,
        grad_kv_chunks: list,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        """Backward pass of ring attention."""
        scale = ctx.scale
        group = ctx.group
        world_size = ctx.world_size
        empty = ctx.empty
        local_rank = ctx.local_rank
        config = ctx.config

        q, out, x, lse, w_qkv = ctx.saved_tensors[:5]
        chunks_to_save = ctx.saved_tensors[5:]

        dq, dk, dv = ring_backward(
            dout,
            q,
            out,
            x,
            lse,
            w_qkv,
            scale,
            group,
            local_rank,
            world_size,
            cast(tuple[torch.Tensor, ...], chunks_to_save),
            ring_config=config,
            empty=empty,
        )

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def broadcast_forward(
    # qkv
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    scale: float,
    dropout_p: float,
    group: dist.ProcessGroup,
    empty: torch.Tensor,
    local_rank: int,
    world_size: int,
    ring_config: RingAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    """
    Forward pass for ring attention using broadcast.

    Each rank will first broadcast either X or KV, instead of performing "P2P" communication.

    q: torch.Tensor
    Query tensor.
    k: torch.Tensor
    Key tensor.
    v: torch.Tensor
    Values tensor
    w_qkv: torch.Tensor
    Weights wrt QKV.
    scale: float
    Scale parameter used in attention calculation.
    dropout_p: float
    Dropout probability.
    group: dist.ProcessGroup
    Process Group containing the ranks that are performing Ring attention.
    empty: torch.Tensor
    Empty tensor for computational reasons.
    local_rank: int
    Local rank of the process group.
    world_size: int
    Size of the ring-attention group.
    """
    b, heads, n_local, d = q.shape

    out = torch.zeros_like(q)
    lse = torch.full((b, heads, n_local), float("-inf"), device=q.device, dtype=q.dtype)

    x_j = x

    # Local KV
    kv = torch.concat([k.unsqueeze(0), v.unsqueeze(0)]).clone().contiguous()
    kv_j, kj, vj = kv, k, v

    chunks_to_save: list[torch.Tensor | None]  # type hint
    if not (ring_config.save_scope == SaveScope.GLOBAL):
        chunks_to_save = [ring_config.save_for_backward_fn(x, kv)]
    else:
        chunks_to_save = [
            None
        ] * world_size  # placeholder list to insert into for each step

    # Kick off first send
    # recv_buf either holds x or kv
    recv_buf, handle = ring_config.forward_send_fn(
        x, kv, local_rank, group, step=0, empty=empty
    )
    handle.wait()
    x_j, kv_j, kj, vj = ring_config.forward_recv_fn(recv_buf, x_j, kv_j, kj, vj)

    for step in range(world_size):
        if step + 1 < world_size:
            recv_buf, handle = ring_config.forward_send_fn(
                x,
                kv,
                local_rank,
                group,
                step + 1,
            )

        if ring_config.send_in_forward == SendForward.X:
            if step != local_rank:
                # Recompute KV from current Xj (received in previous iter)
                kv_j, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)
            else:
                # For the step where we broadcast our own chunk, we can reuse the local KV without recomputing
                kv_j, kj, vj = kv, k, v

        # ---- save for backward ----
        # save_local_or_global == 0: only save own (step 0) chunk; backward re-rings
        # save_local_or_global == 1: save every step's chunk

        if ring_config.save_scope == SaveScope.GLOBAL:
            # append what we need to save for backward.
            # this can be x, kv, or the pointer to cpu offloaded copies of these
            chunks_to_save[(local_rank - step) % world_size] = (
                ring_config.save_for_backward_fn(x_j, kv_j)
            )

        # ---- compute flash attention on local Q vs current Kj, Vj ----
        # Kj, Vj are views into KV; flash attention requires contiguous, so materialize.
        kj_c = kj.contiguous()
        vj_c = vj.contiguous()

        # call torch sdpa
        out_j, l_j = ring_config.attn_core_forward_fn(q, kj_c, vj_c, scale, dropout_p)
        out, lse = _online_softmax_merge(out, lse, out_j, l_j)
        del out_j, l_j, kj_c, vj_c

        # ---- ring rotate ----
        if step < world_size - 1:
            handle.wait()

        # read either x or kv values from the communicated buffers
        # can put into a helper function when/if there are more complicated options involved
        x_j, kv_j, kj, vj = ring_config.forward_recv_fn(recv_buf, x_j, kv_j, kj, vj)

    # Make sure all offloads complete and GPU refs to offloaded tensors are dropped
    if ring_config.offloader is not None:
        ring_config.offloader.wait_offloads()

    out = out
    chunks_to_save_tup = tuple(chunks_to_save)
    return out, lse, chunks_to_save_tup


def ring_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    scale: float,
    dropout_p: float,
    group: dist.ProcessGroup,
    empty: torch.Tensor,
    local_rank: int,
    world_size: int,
    ring_config: RingAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor]]:
    """Forward pass for ring attention."""
    b, heads, n_local, d = q.shape
    out = torch.zeros_like(q)
    lse = torch.full((b, heads, n_local), float("-inf"), device=q.device, dtype=q.dtype)

    x_j = x

    chunks_to_save: tuple | list
    chunks_to_save = []

    # Local KV
    kv = torch.concat([k.unsqueeze(0), v.unsqueeze(0)])
    kv_j, kj, vj = kv, k, v

    # Kick off first send
    # recv_buf either holds x or kv
    recv_buf, handle = ring_config.forward_send_fn(
        x_j, kv_j, local_rank, group, step=0, empty=empty
    )

    for step in range(world_size):
        if (step > 0) and (ring_config.send_in_forward == SendForward.X):
            # Recompute KV from current Xj (received in previous iter)
            kv_j, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)

        # ---- save for backward ----
        # save_local_or_global == 0: only save own (step 0) chunk; backward re-rings
        # save_local_or_global == 1: save every step's chunk
        should_save = (ring_config.save_scope == SaveScope.GLOBAL) or (step == 0)

        if should_save:
            # append what we need to save for backward.
            # this can be x, kv, or the pointer to cpu offloaded copies of these
            chunks_to_save.append(ring_config.save_for_backward_fn(x_j, kv_j))

        # ---- compute flash attention on local Q vs current Kj, Vj ----
        # Kj, Vj are views into KV; flash attention requires contiguous, so materialize.
        kj_c = kj.contiguous()
        vj_c = vj.contiguous()

        # call torch sdpa
        out_j, l_j = ring_config.attn_core_forward_fn(q, kj_c, vj_c, scale, dropout_p)
        out, lse = _online_softmax_merge(out, lse, out_j, l_j)
        del out_j, l_j, kj_c, vj_c

        # ---- ring rotate ----
        if step < world_size - 1:
            handle.wait()

            # read either x or kv values from the communicated buffers
            # can put into a helper function when/if there are more complicated options involved
            x_j, kv_j, kj, vj = ring_config.forward_recv_fn(recv_buf, x_j, kv_j, kj, vj)

            # send x or k/v to the next rank
            recv_buf, handle = ring_config.forward_send_fn(
                x_j,
                kv_j,
                local_rank,
                group,
                step=step,
                empty=empty,
            )

    # Make sure all offloads complete and GPU refs to offloaded tensors are dropped
    if ring_config.offloader is not None:
        ring_config.offloader.wait_offloads()

    # out = out.to(torch.bfloat16)

    chunks_to_save = tuple(chunks_to_save)
    return out, lse, chunks_to_save


def build_ring_forward(
    ring_config: RingAttentionConfig,
) -> Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        float,
        dist.ProcessGroup,
        torch.Tensor,
        int,
        int,
        RingAttentionConfig,
    ],
    tuple[torch.Tensor, torch.Tensor, tuple],
]:
    """Build the forward function for ring attention based on the config."""
    if ring_config.send_forward_op == SendForwardOp.BROADCAST:
        return broadcast_forward
    else:
        return ring_forward


def ring_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    out: torch.Tensor,
    x: torch.Tensor,
    lse: torch.Tensor,
    w_qkv: torch.Tensor,
    scale: float,
    group: dist.ProcessGroup,
    local_rank: int,
    world_size: int,
    chunks_to_save: tuple[torch.Tensor, ...],
    ring_config: RingAttentionConfig,
    empty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implement backward pass for ring attention."""
    # helper definition for implementation
    match ring_config.save_for_backward:
        case SaveBackward.X:
            x_chunks = chunks_to_save
        case SaveBackward.KV:
            kv_chunks = chunks_to_save

    b, h, seq_local, d = q.shape
    dq = torch.zeros_like(q, dtype=torch.float32)

    # Get buffers for dK, dV
    dk_buf, dv_buf = get_dkv_buffer(ring_config, q, world_size)

    # Pre-issue first prefetch (for save_local_or_global == 1) so H2D overlaps
    # with the compute of step 0.
    next_prefetch = None
    next_prefetch_event = None

    def _start_prefetch(
        idx: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.cuda.Event]]:
        """Kick off async prefetch for chunk idx; returns (gpu_tensor, event) or None."""
        if ring_config.save_scope == SaveScope.LOCAL or ring_config.offloader is None:
            return None, None
        if idx >= len(chunks_to_save):
            return None, None
        gpu, ev = ring_config.offloader.prefetch(chunks_to_save[idx])
        return gpu, ev

    # Step 0 chunk: prefetch if applicable
    if ring_config.save_scope == SaveScope.GLOBAL and ring_config.offloader is not None:
        cur_chunk_gpu, cur_chunk_event = _start_prefetch(0)
        next_prefetch, next_prefetch_event = _start_prefetch(1)
    else:
        cur_chunk_gpu, cur_chunk_event = None, None

    # State for ring rotation
    recv_buf = empty

    for step in range(world_size):
        # ---- obtain Kj, Vj for this step ----
        chunk: torch.Tensor  # type-hint
        kv: torch.Tensor
        xj: torch.Tensor
        match ring_config.save_scope:
            case SaveScope.GLOBAL:
                # we have all chunks saved; use the prefetched one
                cur_chunk_event = cast(torch.cuda.Event, cur_chunk_event)
                cur_chunk_gpu = cast(torch.Tensor, cur_chunk_gpu)

                if isinstance(ring_config.offloader, NVLinkActivationOffloader):
                    ring_config.offloader.wait_prefetch(cur_chunk_event)
                    chunk = cur_chunk_gpu
                    # kick off next prefetch ASAP to overlap with compute
                    cur_chunk_gpu, cur_chunk_event = next_prefetch, next_prefetch_event
                    next_prefetch, next_prefetch_event = _start_prefetch(step + 2)
                else:
                    chunk = chunks_to_save[step]

                if ring_config.save_for_backward == SaveBackward.KV:
                    kv = chunk
                    kj, vj = kv[0].contiguous(), kv[1].contiguous()
                elif ring_config.save_for_backward == SaveBackward.X:
                    xj = chunk
                    kv, kj, vj = _compute_kv(xj, w_qkv, b, seq_local, h, d)
                    kj, vj = kj.contiguous(), vj.contiguous()

            case SaveScope.LOCAL:
                # save_local_or_global == 0: only own chunk saved; ring around
                if step == 0:
                    match ring_config.save_for_backward:
                        case SaveBackward.KV:
                            if ring_config.offloader is not None:
                                gpu, ev = ring_config.offloader.prefetch(kv_chunks[0])
                                ring_config.offloader.wait_prefetch(ev)
                                kv = gpu
                            else:
                                kv = kv_chunks[0]
                            kj, vj = kv[0].contiguous(), kv[1].contiguous()
                        case SaveBackward.X:
                            if ring_config.offloader is not None:
                                gpu, ev = ring_config.offloader.prefetch(x_chunks[0])
                                ring_config.offloader.wait_prefetch(ev)
                                xj = gpu
                            else:
                                xj = x_chunks[0]
                            kv, kj, vj = _compute_kv(xj, w_qkv, b, seq_local, h, d)
                            kj, vj = kj.contiguous(), vj.contiguous()

                    # Kick off first ring send
                    match ring_config.send_in_forward:
                        case SendForward.X:
                            recv_buf = torch.empty_like(x)
                            recv_buf, handle = ring_config.ring_comm_fn(
                                x,
                                recv_buf,
                                offset=1,
                                group=group,
                                async_op=True,
                                empty=empty,
                                step=step,
                            )
                            _, kj, vj = _compute_kv(x, w_qkv, b, seq_local, h, d)

                        case SendForward.KV:
                            recv_buf = torch.empty_like(kv)
                            recv_buf, handle = ring_config.ring_comm_fn(
                                kv,
                                recv_buf,
                                offset=1,
                                group=group,
                                async_op=True,
                                empty=empty,
                                step=step,
                            )
                            kj, vj = kv[0].contiguous(), kv[1].contiguous()

                else:
                    # step > 0: receive next chunk
                    handle.wait()
                    match ring_config.send_in_forward:
                        case SendForward.X:
                            xj = recv_buf
                            kv, kj, vj = _compute_kv(xj, w_qkv, b, seq_local, h, d)
                            kj, vj = kj.contiguous(), vj.contiguous()
                        case SendForward.KV:
                            kv = recv_buf
                            kj, vj = kv[0].contiguous(), kv[1].contiguous()

                    # Send either x or k,v to the next rank
                    if step < world_size - 1:
                        next_recv = torch.empty_like(recv_buf)
                        recv_buf, handle = ring_config.ring_comm_fn(
                            recv_buf,
                            next_recv,
                            offset=1,
                            group=group,
                            async_op=True,
                            empty=empty,
                            step=step,
                        )

        # ---- flash attention backward ----
        dq_j, dk_j, dv_j = ring_config.attn_core_backward_fn(
            dout, q, kj, vj, out, lse, scale, seq_local, seq_local
        )
        dq += dq_j
        del dq_j, kj, vj, kv

        # ---- accumulate dK, dV ----
        # Handles how the local calculation of dk_j, dv_j should be accumulated with the "true" value of dk, dv
        # communication is also handled here
        dk_buf, dv_buf = accumulate_dkv(
            ring_config,
            dk_buf,
            dv_buf,
            dk_j,
            dv_j,
            step,
            world_size,
            seq_local,
            local_rank,
            group,
            empty,
        )

    # return the final values for dk, dv
    dk_local, dv_local = postprocess_dkv(
        ring_config, dk_buf, dv_buf, local_rank, seq_local, group
    )

    return dq, dk_local, dv_local


def _select_forward_comm_fns(
    send_forward_op: SendForwardOp, send_in_forward: SendForward, ring_comm_fn: Callable
) -> tuple[Callable, Callable]:
    """Build the send and receive functions for forward based on the config."""
    send_func: Callable[..., tuple[torch.Tensor, dist.Work]]

    match send_forward_op:
        case SendForwardOp.SENDRECV:
            match send_in_forward:
                case SendForward.X:

                    def send_func(  # type: ignore[assignment]
                        x: torch.Tensor,
                        kv: torch.Tensor,
                        local_rank: int,
                        group: dist.ProcessGroup,
                        step: int,
                        empty: torch.Tensor,
                        *args: Any,
                        **kwargs: Any,
                    ) -> tuple[torch.Tensor, dist.Work]:
                        recv_buf = torch.empty_like(x)
                        recv_buf, handle = ring_comm_fn(
                            x,
                            recv_buf,
                            offset=1,
                            group=group,
                            async_op=True,
                            tag=step,
                            empty=empty,
                        )

                        return recv_buf, handle

                case SendForward.KV:

                    def send_func(  # type: ignore[assignment]
                        x: torch.Tensor,
                        kv: torch.Tensor,
                        local_rank: int,
                        group: dist.ProcessGroup,
                        step: int,
                        empty: torch.Tensor,
                        *args: Any,
                        **kwargs: Any,
                    ) -> tuple[torch.Tensor, dist.Work]:
                        recv_buf = torch.empty_like(kv)
                        recv_buf, handle = ring_comm_fn(
                            kv,
                            recv_buf,
                            offset=1,
                            group=group,
                            async_op=True,
                            empty=empty,
                            tag=step,
                        )

                        return recv_buf, handle

                case _:
                    raise NotImplementedError(
                        f"send_in_forward option {send_in_forward} not implemented."
                    )

        case SendForwardOp.BROADCAST:
            match send_in_forward:
                case SendForward.X:

                    def send_func(  # type: ignore[assignment]
                        x: torch.Tensor,
                        kv: torch.Tensor,
                        local_rank: int,
                        group: dist.ProcessGroup,
                        step: int,
                        *args: Any,
                        **kwargs: Any,
                    ) -> tuple[torch.Tensor, dist.Work]:
                        if step == local_rank:
                            recv_buf = x
                        else:
                            recv_buf = torch.empty_like(x)
                        handle = dist.broadcast(
                            recv_buf, src=step, group=group, async_op=True
                        )
                        return recv_buf, handle

                case SendForward.KV:

                    def send_func(  # type: ignore[assignment]
                        x: torch.Tensor,
                        kv: torch.Tensor,
                        local_rank: int,
                        group: dist.ProcessGroup,
                        step: int,
                        *args: Any,
                        **kwargs: Any,
                    ) -> tuple[torch.Tensor, dist.Work]:
                        if step == local_rank:
                            recv_buf = kv
                        else:
                            recv_buf = torch.empty_like(kv)
                        handle = dist.broadcast(
                            recv_buf, src=step, group=group, async_op=True
                        )
                        return recv_buf, handle

                case _:
                    raise NotImplementedError(
                        f"send_in_forward option {send_in_forward} not implemented."
                    )
        case _:
            raise NotImplementedError(
                f"send_forward_op option {send_forward_op} not implemented."
            )

    match send_in_forward:
        case SendForward.X:

            def recv_func(
                recv_buf: torch.Tensor,
                xj: torch.Tensor,
                kv_j: torch.Tensor,
                kj: torch.Tensor,
                vj: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                return recv_buf, kv_j, kj, vj

        case SendForward.KV:

            def recv_func(
                recv_buf: torch.Tensor,
                xj: torch.Tensor,
                kv_j: torch.Tensor,
                kj: torch.Tensor,
                vj: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                kv_j = recv_buf
                kj = kv_j[0]
                vj = kv_j[1]
                return xj, kv_j, kj, vj

        case _:
            raise NotImplementedError(
                f"send_in_forward option {send_in_forward} not implemented."
            )

    return send_func, recv_func


def _select_save_for_backward_fn(
    save_for_backward: SaveBackward,
    save_scope: SaveScope,
    offloader: Optional[NVLinkActivationOffloader],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the function for saving tensors for backward based on the config."""
    match save_for_backward:
        case SaveBackward.X:
            # offload X
            offloader = cast(NVLinkActivationOffloader, offloader)
            if isinstance(offloader, NVLinkActivationOffloader):
                return lambda x, kv: offloader.offload(x)
            else:
                # keep on GPU — must clone if Xj will be overwritten by recv next iter
                if save_scope == SaveScope.GLOBAL:
                    return lambda x, kv: x.clone()
                else:
                    return lambda x, kv: x
        case SaveBackward.KV:
            # offload KV (note: KV is already a contiguous tensor of shape [2,B,H,N,d])
            if isinstance(offloader, NVLinkActivationOffloader):

                def offload_kv(x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
                    kv_cpu = offloader.offload(kv)  # type: ignore[union-attr]
                    kv_cpu = cast(torch.Tensor, kv_cpu)
                    del kv
                    return kv_cpu

                return offload_kv
            else:
                if save_scope == SaveScope.GLOBAL:
                    return lambda x, kv: kv.clone()
                else:
                    return lambda x, kv: kv

        case _:
            return lambda xj, kv_j: None  # default no-op


def get_dkv_buffer(
    ring_config: RingAttentionConfig, q: torch.Tensor, world_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the appropriate dK, dV buffers based on the ring config."""
    # Get the dkv buffer that will be needed
    # If all-gathering the buffer, this will be where the dK, dV are accumulated
    # else, these are the buffers used to send dK and dV from rank to rank

    match ring_config.send_backward:
        case SendBackward.GATHER_DKV:  # send dK = 1
            dk_buf = torch.zeros(
                [*q.shape[0:2], world_size * q.shape[2], *q.shape[3:]],
                device=q.device,
                dtype=q.dtype,
            )
            dv_buf = torch.zeros(
                [*q.shape[0:2], world_size * q.shape[2], *q.shape[3:]],
                device=q.device,
                dtype=q.dtype,
            )
        case SendBackward.SEND_DKV:
            dk_buf = torch.zeros(q.shape, device=q.device, dtype=q.dtype)
            dv_buf = torch.zeros(q.shape, device=q.device, dtype=q.dtype)

    return dk_buf, dv_buf


def accumulate_dkv(
    ring_config: RingAttentionConfig,
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    dk_j: torch.Tensor,
    dv_j: torch.Tensor,
    step: int,
    world_size: int,
    seq_local: int,
    local_rank: int,
    group: dist.ProcessGroup,
    empty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate the locally computed dk_j, dv_j with the "true" dk, dv across the ring rotation."""
    match ring_config.send_backward:
        case SendBackward.GATHER_DKV:
            global_chunk_idx = (local_rank - step) % world_size
            dk_buf[
                :, :, global_chunk_idx * seq_local : (global_chunk_idx + 1) * seq_local
            ] += dk_j
            dv_buf[
                :, :, global_chunk_idx * seq_local : (global_chunk_idx + 1) * seq_local
            ] += dv_j
            del dk_j, dv_j
        case SendBackward.SEND_DKV:
            if step == 0:
                dk_buf += dk_j
                dv_buf += dv_j
                del dk_j, dv_j
            else:
                dkv = torch.cat([dk_j, dv_j], dim=0).contiguous()
                del dk_j, dv_j
                recv_buf = torch.empty_like(dkv)
                recv_buf, _ = ring_config.ring_comm_fn(
                    dkv,
                    recv_buf,
                    offset=-step,
                    group=group,
                    async_op=False,
                    empty=empty,
                    step=step,
                )

                del dkv
                dkj_r, dvj_r = torch.chunk(recv_buf, 2, dim=0)
                dk_buf += dkj_r
                dv_buf += dvj_r
                del dkj_r, dvj_r, recv_buf

    return dk_buf, dv_buf


def postprocess_dkv(
    ring_config: RingAttentionConfig,
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    local_rank: int,
    seq_local: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Post-process the dK, dV buffers after the ring rotation is done to get the final dK, dV for the local chunk."""
    match ring_config.send_backward:
        case SendBackward.GATHER_DKV:
            dist.all_reduce(dk_buf, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(dv_buf, op=dist.ReduceOp.SUM, group=group)
            dk_local = dk_buf[
                :, :, local_rank * seq_local : (local_rank + 1) * seq_local
            ]
            dv_local = dv_buf[
                :, :, local_rank * seq_local : (local_rank + 1) * seq_local
            ]
        case SendBackward.SEND_DKV:
            dk_local = dk_buf
            dv_local = dv_buf

    return dk_local, dv_local
