"""Ring attention forward-backward implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from jigsaw.offload import NVLinkActivationOffloader

from .config import (
    RingAttentionConfig,
    SaveBackward,
    SaveScope,
    SendBackward,
    SendForward,
    SendOp,
)


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
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
    return kv, kv[0], kv[1]


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


# --------------------------------------------------------------------------- #
# autograd.Function glue
# --------------------------------------------------------------------------- #
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
        world_size = dist.get_world_size(group)
        empty = torch.empty(0, device=x.device, dtype=x.dtype)

        ring_forward_fn = build_ring_forward(config)
        out, lse, chunks_to_save = ring_forward_fn(
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
        """Set up context. inputs/output match `forward`'s signature/return."""
        q, k, v, x, w_qkv, scale, dropout_p, group, empty, local_rank, config = inputs
        out, lse, chunks_to_save = output
        ctx.mark_non_differentiable(lse)
        ctx.mark_non_differentiable(*chunks_to_save)
        ctx.save_for_backward(
            q, out, x, lse.mT, w_qkv, *chunks_to_save # *[t for pair in chunks_to_save for t in pair]
        )
        c = cast(RingCtx, ctx)
        c.empty = empty
        c.scale = scale
        c.local_rank = local_rank
        c.group = group
        c.world_size = dist.get_world_size(group) # This is recomputed in forward and in setup context
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
        q, out, x, lse, w_qkv = ctx.saved_tensors[:5]
        chunks_to_save = ctx.saved_tensors[5:]

        dq, dk, dv = ring_backward(
            dout,
            q,
            out,
            x,
            lse,
            w_qkv,
            ctx.scale,
            ctx.group,
            ctx.local_rank,
            ctx.world_size,
            cast(tuple[torch.Tensor, ...], chunks_to_save),
            ring_config=ctx.config,
            empty=ctx.empty,
        )

        return dq, dk, dv, None, None, None, None, None, None, None, None


# --------------------------------------------------------------------------- #
# forward implementations
# --------------------------------------------------------------------------- #
def _attend_step(
    ring_config: RingAttentionConfig,
    q: torch.Tensor,
    kj: torch.Tensor,
    vj: torch.Tensor,
    scale: float,
    dropout_p: float,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the attention core on (q, kj, vj) and merge into the running (out, lse).

    kj/vj are views into the KV buffer; flash attention requires contiguous
    inputs, so they're materialized here.
    """
    out_j, l_j = ring_config.attn_core_forward_fn(
        q, kj.contiguous(), vj.contiguous(), scale, dropout_p
    )

    return _online_softmax_merge(out, lse, out_j, l_j)


def broadcast_forward(
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
    """Forward pass for ring attention using broadcast instead of P2P sends."""
    b, heads, n_local, d = q.shape

    out = torch.zeros_like(q)
    lse = torch.full((b, heads, n_local), float("-inf"), device=q.device, dtype=q.dtype)

    x_j = x

    # Local KV
    kv = torch.concat([k.unsqueeze(0), v.unsqueeze(0)]).clone().contiguous()
    kv_j, kj, vj = kv, k, v

    chunks_to_save: list[torch.Tensor | None]
    if ring_config.save_scope != SaveScope.GLOBAL:
        chunks_to_save = [ring_config.save_for_backward_fn(x, kv)]
    else:
        chunks_to_save = [None] * world_size  # filled in per-step below

    # Kick off first send
    recv_buf, handle = ring_config.forward_send_fn(
        x=x, kv=kv, local_rank=local_rank, group=group, src_rank=0, empty=empty
    )
    handle.wait()
    x_j, kv_j, kj, vj = ring_config.forward_recv_fn(recv_buf, x_j, kv_j, kj, vj)

    for step in range(world_size):
        if step + 1 < world_size:
            recv_buf, handle = ring_config.forward_send_fn(
                x=x,
                kv=kv,
                local_rank=local_rank,
                group=group,
                src_rank=step + 1,
                empty=empty,
            )

        if ring_config.send_in_forward == SendForward.X:
            if step != local_rank:
                # Recompute KV from current Xj (received in previous iter)
                kv_j, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)
            else:
                # Step where we broadcast our own chunk: reuse local KV.
                kv_j, kj, vj = kv, k, v

        if ring_config.save_scope == SaveScope.GLOBAL:
            chunks_to_save[(local_rank - step) % world_size] = (
                ring_config.save_for_backward_fn(x_j, kv_j)
            )

        out, lse = _attend_step(ring_config, q, kj, vj, scale, dropout_p, out, lse)
        if step < world_size - 1:
            handle.wait()

        x_j, kv_j, kj, vj = ring_config.forward_recv_fn(recv_buf, x_j, kv_j, kj, vj)

    ring_config.wait_offloads()
    out = out.to(q.dtype)
    return out, lse, tuple(chunks_to_save)


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
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Forward pass for ring attention using point-to-point sends."""
    b, heads, n_local, d = q.shape
    out = torch.zeros_like(q)
    lse = torch.full((b, heads, n_local), float("-inf"), device=q.device, dtype=q.dtype)

    x_j = x

    # Local KV
    kv = torch.concat([k.unsqueeze(0), v.unsqueeze(0)])
    kv_j, kj, vj = kv, k, v

    chunks_to_save: list[torch.Tensor] = []

    # Kick off first send
    recv_buf, handle = ring_config.forward_send_fn(
        x_j, kv_j, direction=1, group=group, tag=0, empty=empty
    )

    for step in range(world_size):
        if step > 0 and ring_config.send_in_forward == SendForward.X:
            kv_j, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)

        # save_scope == LOCAL: only own (step 0) chunk; backward re-rings.
        # save_scope == GLOBAL: save every step's chunk.
        if ring_config.save_scope == SaveScope.GLOBAL or step == 0:
            chunks_to_save.append(ring_config.save_for_backward_fn(x_j, kv_j))

        out, lse = _attend_step(ring_config, q, kj, vj, scale, dropout_p, out, lse)

        if step < world_size - 1:
            handle.wait()
            # receive either x or kv_j
            x_j, kv_j, kj, vj = ring_config.forward_recv_fn(recv_buf, x_j, kv_j, kj, vj)
            recv_buf, handle = ring_config.forward_send_fn(
                x_j, kv_j, direction=1, group=group, tag=step, empty=empty
            )

    ring_config.wait_offloads()
    out = out.to(q.dtype)
    return out, lse, tuple(chunks_to_save)

def get_send_recv_ranks_hybrid(local_rank, gpus_per_node=4):
    base_node_rank = local_rank // gpus_per_node
    send_to = (local_rank + 1) % gpus_per_node + base_node_rank * gpus_per_node
    recv_from = (local_rank - 1) % gpus_per_node + base_node_rank * gpus_per_node

    return [send_to, recv_from]

def hybrid_forward(
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
    """Forward pass for ring attention using broadcast instead of P2P sends."""
    b, heads, n_local, d = q.shape
    gpus_per_node = 4  # TODO: indicate the number of GPUs per node
    n_nodes = world_size // gpus_per_node

    out = torch.zeros_like(q)
    lse = torch.full((b, heads, n_local), float("-inf"), device=q.device, dtype=q.dtype)
    x_j = x

    # Local KV
    kv = torch.concat([k.unsqueeze(0), v.unsqueeze(0)]).clone().contiguous()
    kv_j, kj, vj = kv, k, v

    chunks_to_save: list[torch.Tensor | None]
    if ring_config.save_scope != SaveScope.GLOBAL:
        chunks_to_save = [ring_config.save_for_backward_fn(x, kv)]
    else:
        chunks_to_save = [None] * world_size  # filled in per-step below

    for node_step in range(n_nodes):
        # Kick off first send across nodes
        if node_step < n_nodes - 1:
            internode_recv_buf = (
                torch.empty_like(x_j)
                if ring_config.send_in_forward == SendForward.X
                else torch.empty_like(kv_j)
            )
            internode_recv_buf, internode_handle = ring_config.p2p_fn(
                data=(
                    x.clone()
                    if ring_config.send_in_forward == SendForward.X
                    else kv.clone()
                ),
                buffer=internode_recv_buf,
                direction=gpus_per_node,
                group=group,
                async_op=True,
                tag=node_step,
                empty=empty,
            )
        send_recv_ranks = get_send_recv_ranks_hybrid(local_rank, gpus_per_node=gpus_per_node)
        for step in range(gpus_per_node):
            if step + 1 < gpus_per_node:
                recv_buf = (
                    torch.empty_like(x_j)
                    if ring_config.send_in_forward == SendForward.X
                    else torch.empty_like(kv_j)
                )
                # Start communicating information in a ring
                recv_buf, handle = ring_config.send_op_fn(
                    x_j if ring_config.send_in_forward == SendForward.X else kv_j,
                    recv_buf,
                    send_recv_ranks=send_recv_ranks,
                    group=group,
                    tag=step,
                )

            if ring_config.send_in_forward == SendForward.X:
                kv_j, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)

            if ring_config.save_scope == SaveScope.GLOBAL:
                global_step = node_step * gpus_per_node + step
                chunks_to_save[(local_rank - global_step) % world_size] = (
                    ring_config.save_for_backward_fn(x_j, kv_j)
                )
            out, lse = _attend_step(ring_config, q, kj, vj, scale, dropout_p, out, lse)
            if step < gpus_per_node - 1:
                handle.wait()

            x_j, kv_j, kj, vj = ring_config.forward_recv_fn(recv_buf, x_j, kv_j, kj, vj)

        # --- now consume the internode data that arrived during the inner loop ---
        if node_step < n_nodes - 1:
            internode_handle.wait()
            x_j, kv_j, kj, vj = ring_config.forward_recv_fn(
                internode_recv_buf, x_j, kv_j, kj, vj
            )
            if ring_config.send_in_forward == SendForward.X:
                kv_j, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)

    ring_config.wait_offloads()
    out = out.to(q.dtype)
    return out, lse, tuple(chunks_to_save)


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
    """Pick the forward implementation (P2P vs broadcast) based on config."""
    if ring_config.send_op == SendOp.BROADCAST:
        return broadcast_forward
    elif ring_config.send_op == SendOp.HYBRID:
        return hybrid_forward
    elif ring_config.send_op == SendOp.P2P or ring_config.send_op == SendOp.ALLTOALL:
        return ring_forward
    else:
        raise NotImplementedError(f"Unknown send_op: {ring_config.send_op}")


# --------------------------------------------------------------------------- #
# backward: dK/dV chunk providers
#
# How to obtain (kj, vj, kv) for a given ring step, including any prefetching/communication bookkeeping.
# --------------------------------------------------------------------------- #
class _BackwardChunkProvider(ABC):
    """Yields (kj, vj, kv) for each backward ring step."""

    @abstractmethod
    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return contiguous (kj, vj) and the underlying kv buffer for `step`."""


class _GlobalChunkProvider(_BackwardChunkProvider):
    """
    Look up the relevant chunk that was saved during the forward pass.

    If the data is offloaded to the CPU, the data movement is handeled here.
    """

    def __init__(
        self,
        ring_config: RingAttentionConfig,
        chunks_to_save: tuple[torch.Tensor, ...],
        w_qkv: torch.Tensor,
        b: int,
        seq_local: int,
        h: int,
        d: int,
    ) -> None:
        """
        Initialize _GlobalChunkProvider.

        Parameters
        ----------
        ring_config: RingAttentionConfig
            Configuration class for ring attention.
        chunks_to_save: tuple[torch.Tensor, ...]
            Chunks saved between the forward and backward pass.
        w_qkv: torch.Tensor
            Weights of the QKV mapping.
        b: int
            Batch dimension.
        seq_local: int
            Local sequence length.
        h: int
            Number of attention heads.
        d: int
            Embedding dimension per head.
        """
        self.config = ring_config
        self.chunks = chunks_to_save
        self.w_qkv = w_qkv
        self.b, self.seq_local, self.h, self.d = b, seq_local, h, d

        self._cur_gpu: Optional[torch.Tensor] = None
        self._cur_event: Optional[torch.cuda.Event] = None
        self._next_gpu: Optional[torch.Tensor] = None
        self._next_event: Optional[torch.cuda.Event] = None

        if ring_config.offloader is not None:
            self._cur_gpu, self._cur_event = self._start_prefetch(0)
            self._next_gpu, self._next_event = self._start_prefetch(1)

    def _start_prefetch(
        self, idx: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.cuda.Event]]:
        """
        Begin prefetching process for a given chunk.

        Parameters
        ----------
        idx: int
            Step of the ring communication.

        Returns
        -------
        Prefetched variable and event.
        """
        if self.config.offloader is None or idx >= len(self.chunks):
            return None, None
        return self.config.offloader.prefetch(self.chunks[idx])

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the current chunk and kick off the process for fetching the next chunk.

        Parameters
        ----------
        step: int
            Current step of ring attention.

        Returns
        -------
        Values of k, v, and kv (duplicate information, retained to reduce number of
        messages sent in future steps.)
        """
        offloader = self.config.offloader
        if isinstance(offloader, NVLinkActivationOffloader):
            offloader.wait_prefetch(cast(torch.cuda.Event, self._cur_event))
            chunk = cast(torch.Tensor, self._cur_gpu)
            # advance the double buffer, kicking off the next prefetch ASAP
            self._cur_gpu, self._cur_event = self._next_gpu, self._next_event
            self._next_gpu, self._next_event = self._start_prefetch(step + 2)
        else:
            chunk = self.chunks[step]

        if self.config.save_for_backward == SaveBackward.KV:
            kv = chunk
            return kv[0].contiguous(), kv[1].contiguous(), kv
        elif self.config.save_for_backward == SaveBackward.X:
            kv, kj, vj = _compute_kv(
                chunk, self.w_qkv, self.b, self.seq_local, self.h, self.d
            )
            return kj.contiguous(), vj.contiguous(), kv
        else:
            raise NotImplementedError


class _LocalChunkProvider(_BackwardChunkProvider):
    """
    Provide the relevant chunk for a given step.

    Only the local (step-0) chunk (either X or KV) was saved;
    Every other rank's chunk is re-circulated around the ring with one P2P send/recv per step, mirroring
    the forward pass's communication pattern but in reverse role.
    """

    def __init__(
        self,
        ring_config: RingAttentionConfig,
        chunks_to_save: tuple[torch.Tensor, ...],
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        b: int,
        seq_local: int,
        h: int,
        d: int,
        group: dist.ProcessGroup,
        empty: torch.Tensor,
    ) -> None:
        """
        Initialize LocalChunkProvider.

        Parameters
        ----------
        ring_config: RingAttentionConfig
            configuration for ring attention.
        chunks_to_save: tuple[torch.Tensor, ...]
            Chunks that were saved from the forward pass.
        x: torch.Tensor
            X data.
        w_qkv: torch.Tensor
            Weight matrix for attention.
        b: int
            Batch size.
        seq_local: int
            Local sequence length.
        h: int
            Number of local heads.
        d: int
            Embedding dimension per head.
        group: dist.ProcessGroup
            Process group upon which ring attention is performed.
        empty: torch.Tensor
            Empty tensor for computational reasons.
        """
        self.config = ring_config
        self.chunk0 = chunks_to_save[0]
        self.x = x
        self.w_qkv = w_qkv
        self.b, self.seq_local, self.h, self.d = b, seq_local, h, d
        self.group = group
        self.empty = empty

        self._recv_buf = empty
        self._handle: Optional[dist.Work] = None

    def _materialize_chunk0(self) -> torch.Tensor:
        """
        Load local saved chunk onto device.

        If the data is not offloaded and is already on the device,
        simply return the data item. Otherwise, if it is offloaded,
        fetch the sample.

        Returns
        -------
        chunk: torch.Tensor
            Data chunk that was offloaded.
        """
        chunk = self.chunk0
        if isinstance(self.config.offloader, NVLinkActivationOffloader):
            gpu, ev = self.config.offloader.prefetch(chunk)
            self.config.offloader.wait_prefetch(ev)
            chunk = gpu
        return chunk

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if step == 0:
            chunk = self._materialize_chunk0()
            if self.config.save_for_backward == SaveBackward.KV:
                kv = chunk
            else:
                kv, _, _ = _compute_kv(
                    chunk, self.w_qkv, self.b, self.seq_local, self.h, self.d
                )

            # Kick off the first ring send of whichever buffer is configured.
            send_buf = self.x if self.config.send_in_forward == SendForward.X else kv
            self._recv_buf = torch.empty_like(send_buf)
            self._recv_buf, self._handle = self.config.send_op_fn(
                send_buf,
                self._recv_buf,
                1,
                group=self.group,
                async_op=True,
                empty=self.empty,
                step=step,
            )

            if self.config.send_in_forward == SendForward.X:
                # Local Q always attends its own local K/V on step 0.
                kv, kj, vj = _compute_kv(
                    self.x, self.w_qkv, self.b, self.seq_local, self.h, self.d
                )
            else:
                kj, vj = kv[0].contiguous(), kv[1].contiguous()
            return kj, vj, kv

        # step > 0: receive the chunk that's been circulating, then forward
        # it on (unless this is the last step).
        assert self._handle is not None
        self._handle.wait()
        recv_buf = self._recv_buf
        if self.config.send_in_forward == SendForward.X:
            kv, kj, vj = _compute_kv(
                recv_buf, self.w_qkv, self.b, self.seq_local, self.h, self.d
            )
            kj, vj = kj.contiguous(), vj.contiguous()
        else:
            kv = recv_buf
            kj, vj = kv[0].contiguous(), kv[1].contiguous()

        # `world_size` isn't tracked locally; ring_backward only calls us
        # `world_size` times total, so re-sending on the final step is
        # harmless but wasteful -- guarded via `has_more` below instead.
        return kj, vj, kv

    def maybe_forward(self, step: int, world_size: int) -> None:
        """
        Send chunk to next partner rank if necessary.

        Chunk does not get sent if it is the first (treated differently) or final (done) step.

        Parameters
        ----------
        step: int
            stage of the ring attention.
        world_size: int
            Size of the domain-parallel group.
        """
        if step == 0 or step >= world_size - 1:
            return
        next_recv = torch.empty_like(self._recv_buf)
        self._recv_buf, self._handle = self.config.send_op_fn(
            self._recv_buf,
            next_recv,
            1,
            group=self.group,
            async_op=True,
            empty=self.empty,
            step=step,
        )


class _LocalChunkProviderBroadcast(_LocalChunkProvider):
    def __init__(
        self,
        ring_config: RingAttentionConfig,
        chunks_to_save: tuple[torch.Tensor, ...],
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        b: int,
        seq_local: int,
        h: int,
        d: int,
        group: dist.ProcessGroup,
        empty: torch.Tensor,
        local_rank: torch.Tensor,
    ) -> None:
        super().__init__(
            ring_config, chunks_to_save, x, w_qkv, b, seq_local, h, d, group, empty
        )
        self._recv_buf = torch.empty_like(
            x if ring_config.send_in_forward == SendForward.X else chunks_to_save[0]
        )
        self._handle: Optional[dist.Work] = None
        self.local_rank = local_rank

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        chunk = self._materialize_chunk0()
        if self.config.save_for_backward == SaveBackward.KV:
            kv = chunk
        else:
            kv, _, _ = _compute_kv(
                chunk, self.w_qkv, self.b, self.seq_local, self.h, self.d
            )

        # Kick off the first ring send of whichever buffer is configured.
        send_buf = self.x if self.config.send_in_forward == SendForward.X else kv
        self._recv_buf = (
            send_buf if self.local_rank == step else torch.empty_like(send_buf)
        )
        self._handle = dist.broadcast(
            self._recv_buf, src=step, group=self.group, async_op=True
        )

        # step > 0: receive the chunk that's been circulating, then forward
        # it on (unless this is the last step).
        assert self._handle is not None
        self._handle.wait()
        recv_buf = self._recv_buf

        if self.config.send_in_forward == SendForward.X:
            kv, kj, vj = _compute_kv(
                recv_buf, self.w_qkv, self.b, self.seq_local, self.h, self.d
            )
            kj, vj = kj.contiguous(), vj.contiguous()
        else:
            kv = recv_buf
            kj, vj = kv[0].contiguous(), kv[1].contiguous()

        # `world_size` isn't tracked locally; ring_backward only calls us
        # `world_size` times total, so re-sending on the final step is
        # harmless but wasteful -- guarded via `has_more` below instead.
        return kj, vj, kv

    def maybe_forward(self, step, world_size):
        return


class _LocalChunkProviderHybrid(_LocalChunkProvider):
    def __init__(
        self,
        ring_config: RingAttentionConfig,
        chunks_to_save: tuple[torch.Tensor, ...],
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        b: int,
        seq_local: int,
        h: int,
        d: int,
        group: dist.ProcessGroup,
        empty: torch.Tensor,
        local_rank: torch.Tensor,
    ) -> None:
        super().__init__(
            ring_config, chunks_to_save, x, w_qkv, b, seq_local, h, d, group, empty
        )
        self._recv_buf = torch.empty_like(
            x if ring_config.send_in_forward == SendForward.X else chunks_to_save[0]
        )
        self._handle: Optional[dist.Work] = None
        self.local_rank = local_rank
        self.gpus_per_node = 4
        self._internode_handle = None
        self._internode_recv_buf = None

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if step == 0:
            chunk = self._materialize_chunk0()
            if self.config.save_for_backward == SaveBackward.KV:
                kv = chunk
            else:
                kv, _, _ = _compute_kv(
                    chunk, self.w_qkv, self.b, self.seq_local, self.h, self.d
                )
            
            # Kick off the first ring send of whichever buffer is configured.
            internode_recv_buf = (
                torch.empty_like(self.x)
                if self.config.send_in_forward == SendForward.X
                else torch.empty_like(kv)
            )
            # print(f"R{self.local_rank} Step {step} Internode send send buf is {self.x[0][0][0]}")

            self._internode_recv_buf, self._internode_handle = self.config.p2p_fn(
                data=(
                    self.x.clone()
                    if self.config.send_in_forward == SendForward.X
                    else kv.clone()
                ),
                buffer=internode_recv_buf,
                direction=self.gpus_per_node,
                group=self.group,
                async_op=True,
                tag=step % 4,
                empty=self.empty,
            )

            send_buf = self.x if self.config.send_in_forward == SendForward.X else kv
            self._recv_buf = torch.empty_like(send_buf)

            # if we are at a multiple of 4,
            # wait for what is received from the internode buffer
            if step % self.gpus_per_node == 0 and step > 0:
                self._internode_handle.wait()
                send_buf = self._internode_recv_buf
            send_recv_ranks = get_send_recv_ranks_hybrid(self.local_rank, gpus_per_node=self.gpus_per_node)
            # Launch send_op to the neighbouring ranks
            self._recv_buf, self._handle = self.config.send_op_fn(
                send_buf,
                self._recv_buf,
                send_recv_ranks=send_recv_ranks,
                group=self.group,
                async_op=True,
                empty=self.empty,
                step=step,
            )
            print(f'R{dist.get_rank()} STEP {step} recv_buf {self.x[0][0][0]}')
            
            if self.config.send_in_forward == SendForward.X:
                # Local Q always attends its own local K/V on step 0.
                kv, kj, vj = _compute_kv(
                    self.x, self.w_qkv, self.b, self.seq_local, self.h, self.d
                )
            else:
                kj, vj = kv[0].contiguous(), kv[1].contiguous()
            return kj, vj, kv

        # step > 0: receive the chunk that's been circulating, then forward
        # it on (unless this is the last step).
        
        if step % self.gpus_per_node == 0: # if we are receiving from internode
            assert self._internode_handle is not None            
            self._internode_handle.wait()
            recv_buf = self._internode_recv_buf
        else:
            assert self._handle is not None
            self._handle.wait()
            recv_buf = self._recv_buf        

        if self.config.send_in_forward == SendForward.X:
            print(f'R{dist.get_rank()} STEP {step} recv_buf {recv_buf[0][0][0]}')
            kv, kj, vj = _compute_kv(
                recv_buf, self.w_qkv, self.b, self.seq_local, self.h, self.d
            )
            kj, vj = kj.contiguous(), vj.contiguous()
        else:
            kv = recv_buf
            kj, vj = kv[0].contiguous(), kv[1].contiguous()

        # `world_size` isn't tracked locally; ring_backward only calls us
        # `world_size` times total, so re-sending on the final step is
        # harmless but wasteful -- guarded via `has_more` below instead.
        return kj, vj, kv

    def maybe_forward(self, step, world_size):
        if step == 0 or step >= world_size - 1:
            return


        if step % self.gpus_per_node == 0 and step // self.gpus_per_node < (
            self.group.size() // self.gpus_per_node - 1
        ):  # if we are at the start of a "internode" step
            
            # internode if necessary
            internode_recv_buf = (
                torch.empty_like(self.x)
                if self.config.send_in_forward == SendForward.X
                else torch.empty_like(kv)
            )
            print(f'Internode send {step} sending {self._internode_recv_buf[0][0][0]}')
            self._internode_recv_buf, self._internode_handle = self.config.p2p_fn(
                data=self._internode_recv_buf,
                buffer=internode_recv_buf,
                direction=self.gpus_per_node,
                group=self.group,
                async_op=True,
                tag=step % 4,
                empty=self.empty,
                group_size=self.group.size()
            )
        # if we are here, intranode recv buffer is overwritten
        # by internode message
        if step % self.gpus_per_node == 0:
            self._recv_buf = self._internode_recv_buf

        next_recv = torch.empty_like(self._recv_buf)
        # intranode if necessary
        send_recv_ranks = get_send_recv_ranks_hybrid(self.local_rank, gpus_per_node=self.gpus_per_node)
        self._recv_buf, self._handle = self.config.send_op_fn(
            self._recv_buf,
            next_recv,
            send_recv_ranks=send_recv_ranks,
            group=self.group,
            async_op=True,
            empty=self.empty,
            step=step,        )


def _build_backward_chunk_provider(
    ring_config: RingAttentionConfig,
    chunks_to_save: tuple[torch.Tensor, ...],
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    b: int,
    seq_local: int,
    h: int,
    d: int,
    group: dist.ProcessGroup,
    empty: torch.Tensor,
    local_rank: int,
) -> _BackwardChunkProvider:
    """
    Build corresponding class that will return the Chunk provider.

    Parameters
    ----------
    ring_config: RingAttentionConfig
        Configuration for ring attention.
    chunks_to_save: tuple[torch.Tensor, ...]
        Chunks (either X or KV) saved between the forward and backward pass.
    x: torch.Tensor
        Input data x.
    w_qkv: torch.Tensor
        Weight matrix for attention.
    b: int
        Batch size.
    seq_local: int
        Local sequence length.
    h: int
        Number of attention heads.
    d: int
        Per-head embedding dimension.
    group: dist.ProcessGroup
        Process group for ring attention.
    empty: torch.Tensor
        Empty tensor for computation reasons.

    Returns
    -------
    _BackwardChunkProvider
    """
    if ring_config.save_scope == SaveScope.GLOBAL:
        return _GlobalChunkProvider(
            ring_config, chunks_to_save, w_qkv, b, seq_local, h, d
        )
    elif ring_config.save_scope == SaveScope.LOCAL:
        if ring_config.send_op == SendOp.P2P or ring_config.send_op == SendOp.ALLTOALL:
            return _LocalChunkProvider(
                ring_config, chunks_to_save, x, w_qkv, b, seq_local, h, d, group, empty
            )
        elif ring_config.send_op == SendOp.BROADCAST:
            return _LocalChunkProviderBroadcast(
                ring_config,
                chunks_to_save,
                x,
                w_qkv,
                b,
                seq_local,
                h,
                d,
                group,
                empty,
                local_rank,
            )
        elif ring_config.send_op == SendOp.HYBRID:
            return _LocalChunkProviderHybrid(
                ring_config,
                chunks_to_save,
                x,
                w_qkv,
                b,
                seq_local,
                h,
                d,
                group,
                empty,
                local_rank,
            )
    else:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# backward: dK/dV accumulation across the ring
# --------------------------------------------------------------------------- #
def get_dkv_buffer(
    ring_config: RingAttentionConfig, q: torch.Tensor, world_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the dK, dV buffers appropriate for the accumulation strategy.

    GATHER_DKV: one global-sized buffer that every step writes its chunk
        into at the right offset, all-reduced at the end.
    SEND_DKV: a local-sized buffer that dK/dV are added into, with partial
        sums P2P-relayed around the ring each step.
    """
    match ring_config.send_backward:
        case SendBackward.GATHER_DKV:
            shape = [*q.shape[0:2], world_size * q.shape[2], *q.shape[3:]]
        case SendBackward.SEND_DKV:
            shape = list(q.shape)
    return (
        torch.zeros(shape, device=q.device, dtype=q.dtype),
        torch.zeros(shape, device=q.device, dtype=q.dtype),
    )


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
    """Fold this step's locally computed dk_j, dv_j into the running dK, dV."""
    match ring_config.send_backward:
        case SendBackward.GATHER_DKV:
            if ring_config.send_op == SendOp.HYBRID:
                return _accumulate_dkv_hybrid(
                    dk_buf, dv_buf, dk_j, dv_j, step, seq_local, local_rank, world_size
                )
            else:
                return _accumulate_dkv_gather(
                    dk_buf, dv_buf, dk_j, dv_j, step, seq_local, local_rank, world_size
                )
        case SendBackward.SEND_DKV:
            return _accumulate_dkv_send(
                ring_config, dk_buf, dv_buf, dk_j, dv_j, step, group, empty
            )


def _accumulate_dkv_gather(
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    dk_j: torch.Tensor,
    dv_j: torch.Tensor,
    step: int,
    seq_local: int,
    local_rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Accumulate local dk, dv in the correct space in the buffer.

    Used when each rank stores the full global size of dk, dv buffers.

    Parameters
    ----------
    dk_buf: torch.Tensor
        Buffer for accumulating dk.
    dv_buf: torch.Tensor
        Buffer for accumulating dv.
    dk_j: torch.Tensor
        Local result for dk for step j.
    dv_j: torch.Tensor
        Local result for dk for step j.
    step: int
        Step number in ring attention.
    seq_local: int
        Local sequence length.
    local_rank: int
        Local rank within ring attention.
    world_size: int
        World size of ring attention.

    Returns
    -------
    dk_buf, dv_buf: torch.Tensor, torch.Tensor
        Accumulated buffers for dk and dv.
    """
    global_chunk_idx = (local_rank - step) % world_size
    lo, hi = global_chunk_idx * seq_local, (global_chunk_idx + 1) * seq_local
    dk_buf[:, :, lo:hi] += dk_j
    dv_buf[:, :, lo:hi] += dv_j
    return dk_buf, dv_buf

def _accumulate_dkv_hybrid(
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    dk_j: torch.Tensor,
    dv_j: torch.Tensor,
    step: int,
    seq_local: int,
    local_rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Accumulate local dk, dv in the correct space in the buffer.

    Used when each rank stores the full global size of dk, dv buffers.

    Parameters
    ----------
    dk_buf: torch.Tensor
        Buffer for accumulating dk.
    dv_buf: torch.Tensor
        Buffer for accumulating dv.
    dk_j: torch.Tensor
        Local result for dk for step j.
    dv_j: torch.Tensor
        Local result for dk for step j.
    step: int
        Step number in ring attention.
    seq_local: int
        Local sequence length.
    local_rank: int
        Local rank within ring attention.
    world_size: int
        World size of ring attention.

    Returns
    -------
    dk_buf, dv_buf: torch.Tensor, torch.Tensor
        Accumulated buffers for dk and dv.
    """
    node, local = divmod(local_rank, 4)
    if step % 4 == 0 : # temporarily block out this path
        # node-dimension ring, local index fixed
        send_node = (node - (step)) % world_size // 4
        send_to   = send_node * 4 + local
        if local_rank >=4 and step == 4:
            send_to = (local_rank - 4) % 4
        elif step == 4:
            send_to = local_rank + 4
        if step == 0: 
            send_to = local_rank
    else:
        # local-dimension ring, node fixed
        base_node_rank = local_rank // 4
        send_to = (local_rank - step ) % 4 + base_node_rank * 4
            
        if step > 4 and local_rank >= 4:
            send_to -= 4
        elif step > 4 and local_rank < 4:
            send_to += 4

    print(f'step {step} global chunk idx {send_to}')
    global_chunk_idx = send_to
    lo, hi = global_chunk_idx * seq_local, (global_chunk_idx + 1) * seq_local
    dk_buf[:, :, lo:hi] += dk_j
    dv_buf[:, :, lo:hi] += dv_j
    return dk_buf, dv_buf

def get_send_recv_ranks(rank, step, nodes, gpus_per_node):
    node, local = divmod(rank, gpus_per_node)
    if step % gpus_per_node == 0 : # temporarily block out this path
        # node-dimension ring, local index fixed
        send_node = (node - (step+1)) % nodes
        recv_node = (node + (step+1)) % nodes
        send_to   = send_node * gpus_per_node + local
        recv_from = recv_node * gpus_per_node + local
    else:
        # local-dimension ring, node fixed
        
        base_node_rank = rank // gpus_per_node
        send_to = (rank - step ) % gpus_per_node + base_node_rank * gpus_per_node
        recv_from = (rank + step) % gpus_per_node + base_node_rank * gpus_per_node
        
        if step > 4 and rank >= 4:
            send_to -= 4
            recv_from -= 4
        elif step > 4 and rank < 4:
            send_to += 4
            recv_from += 4 

    return send_to, recv_from

def _accumulate_dkv_send(
    ring_config: RingAttentionConfig,
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    dk_j: torch.Tensor,
    dv_j: torch.Tensor,
    step: int,
    group: dist.ProcessGroup,
    empty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Send dk and dv to the corresponding rank.

    After a rank computes dk and dv on the chunk of the data that it sees,
    send the values back to the home rank for accumulation.

    Parameters
    ----------
    ring_config: RingAttentionConfig
        Configuration for ring attention.
    dk_buf: torch.Tensor
        Buffer to accumulate dk.
    dv_buf: torch.Tensor
        Buffer to accumulate dv.
    dk_j: torch.Tensor
        Local result of dk.
    dv_j: torch.Tensor
        Local result of dv.
    step: int
        Step.
    group: dist.ProcessGroup
        Process group for ring attention.
    empty: torch.Tensor
        Empty tensor for computational reasons.

    Returns
    -------
    dk_buf, dv_buf: torch.Tensor, torch.Tensor
        Accumulated values of dk and dv.
    """
    if step == 0:
        dk_buf += dk_j
        dv_buf += dv_j
        return dk_buf, dv_buf

    dkv = torch.cat([dk_j, dv_j], dim=0).contiguous()
    recv_buf = torch.empty_like(dkv)

    # hybrid
    if ring_config.send_op == SendOp.HYBRID:
        # step 1 --> rank 3 2-> rank 2, 3-> rank 1
        # step 4 --> rank 4, 5 --> rank 7, 6 -> rank
        
        # Rank 4: step 1 --> rank 7
        send_recv_ranks = get_send_recv_ranks(dist.get_rank(group), step=step, nodes=group.size()//4, gpus_per_node=4)
        if step % 4 != 0:
            # groups are not set correctly
            # since we artificially have a group of size 4
            # rank 6 needs to send to 5 in the beginning, and not 3
            # print(f'DKV: R{dist.get_rank()}  step {step} sending to {send_recv_ranks[0]} recv from {send_recv_ranks[1]}')
            recv_buf, _ = ring_config.p2p_fn(
                dkv, recv_buf, send_recv_ranks=send_recv_ranks, group=group, async_op=False, empty=empty, step=step
            ) 
        else:
            # print(f'DKV: R{dist.get_rank()}  step {step} sending to {send_recv_ranks[0]} recv from {send_recv_ranks[1]}')
            recv_buf, _ = ring_config.p2p_fn(
                dkv, recv_buf,  send_recv_ranks=send_recv_ranks, group=group, async_op=False, empty=empty, step=step, group_size=8
            )
    else:
        recv_buf, _ = ring_config.p2p_fn(
            dkv, recv_buf, group.size()-step, group=group, async_op=False, empty=empty, step=step
        )
    dkj_r, dvj_r = torch.chunk(recv_buf, 2, dim=0)
    dk_buf += dkj_r
    dv_buf += dvj_r
    return dk_buf, dv_buf


def postprocess_dkv(
    ring_config: RingAttentionConfig,
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    local_rank: int,
    seq_local: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Postprocess the dk and dv results at the end.

    If global dk, dv values are gathered, we need to perform an all-reduction
    and final slide to match the per-rank values.

    Parameters
    ----------
    ring_config: RingAttentionConfig
        Configuration class for ring attention.
    dk_buf: torch.Tensor
        Buffer where the intermediate values of dk are stored.
    dv_buf: torch.Tensor
        Buffer where the intermediate values of dv are stored.
    local_rank: int
        Local rank within ring group.
    seq_local: int
        Length of local sequence.
    group: dist.ProcessGroup
        Process group upon which ring attention is computed.

    Returns
    -------
    dk_buf, dv_buf: tuple[torch.Tensor, torch.Tensor]
        Final results of dk and dv that correspond to each rank.
    """
    match ring_config.send_backward:
        case SendBackward.GATHER_DKV:
            dist.all_reduce(dk_buf, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(dv_buf, op=dist.ReduceOp.SUM, group=group)
            lo, hi = local_rank * seq_local, (local_rank + 1) * seq_local
            return dk_buf[:, :, lo:hi], dv_buf[:, :, lo:hi]
        case SendBackward.SEND_DKV:
            return dk_buf, dv_buf


# --------------------------------------------------------------------------- #
# backward: main loop
# --------------------------------------------------------------------------- #
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
    """Backward pass for ring attention."""
    b, h, seq_local, d = q.shape
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk_buf, dv_buf = get_dkv_buffer(ring_config, q, world_size)

    chunk_provider = _build_backward_chunk_provider(
        ring_config,
        chunks_to_save,
        x,
        w_qkv,
        b,
        seq_local,
        h,
        d,
        group,
        empty,
        local_rank,
    )

    for step in range(world_size):
        kj, vj, kv = chunk_provider.get(step)

        dq_j, dk_j, dv_j = ring_config.attn_core_backward_fn(
            dout, q, kj, vj, out, lse, scale, seq_local, seq_local
        )
        dq += dq_j
        del dq_j, kj, vj, kv

        if (
            ring_config.send_op == SendOp.P2P
            or (ring_config.save_scope == SaveScope.GLOBAL)
            or ring_config.send_op == SendOp.HYBRID
        ):
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
        else:
            dist.all_reduce(dk_j, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(dv_j, op=dist.ReduceOp.SUM, group=group)
            if local_rank == step:
                dk_local = dk_j
                dv_local = dv_j

        if isinstance(chunk_provider, _LocalChunkProvider) or isinstance(
            chunk_provider, _LocalChunkProviderHybrid
        ):
            chunk_provider.maybe_forward(step, world_size)

    if ring_config.send_op != SendOp.BROADCAST or (
        ring_config.save_scope == SaveScope.GLOBAL
    ):
        dk_local, dv_local = postprocess_dkv(
            ring_config, dk_buf, dv_buf, local_rank, seq_local, group
        )
    return dq, dk_local, dv_local
