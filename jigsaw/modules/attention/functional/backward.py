"""Backward implementations for ring attention."""

from typing import Optional

import torch
import torch.distributed as dist

from jigsaw.modules.attention.config import (
    RingAttentionConfig,
    SaveScope,
    SendBackward,
    SendOp,
)

from .backward_chunk_provider import _build_backward_chunk_provider
from .functional import HybridTopology, combine_dropout_seed, get_hybrid_topology

# Hybrid send-ops whose backward chunk provider rotates chunks intra-node as well
# as inter-node, so a step's dK/dV belongs to local index ``local_idx + step``
# rather than ``step`` (HYBRID_BROADCAST leaves the chunk put intra-node).
_INTRA_ROTATING_HYBRIDS = frozenset({SendOp.HYBRID_P2P, SendOp.HYBRID_ALLTOALL})


# --------------------------------------------------------------------------- #
# backward: which global chunk does step `step` hold?
#
# Shared by the dropout-mask seed lookup (below, before the core call) and
# `accumulate_dkv`'s dK/dV bucketing (which needs the same answer *after* the
# core call) -- both need to know, for LOCAL save_scope, which global chunk
# the backward ring re-circulated into this step, and the mapping is the same
# regardless of what either caller does with it.
# --------------------------------------------------------------------------- #
def _local_scope_global_chunk_idx(
    ring_config: RingAttentionConfig,
    local_rank: int,
    step: int,
    ring_group_size: int,
    group: dist.ProcessGroup,
) -> int:
    """Return the global chunk index held at backward ``step`` for ``SaveScope.LOCAL``."""
    match ring_config.send_op:
        case SendOp.HYBRID_P2P | SendOp.HYBRID_BROADCAST | SendOp.HYBRID_ALLTOALL:
            # The chunk sits on node `node_idx - step // gpn` (the inter-node
            # ring runs backwards relative to the forward). HYBRID_P2P and
            # HYBRID_ALLTOALL also rotate it intra-node, so its local index
            # is `local_idx + step`; HYBRID_BROADCAST leaves it put and
            # node-local rank `step % gpn` was the broadcast source, so
            # `step` is the local index.
            topo = get_hybrid_topology(group, ring_config.gpus_per_node)
            node_step = step // topo.gpus_per_node
            local = (
                topo.local_idx + step
                if ring_config.send_op in _INTRA_ROTATING_HYBRIDS
                else step
            )
            return topo.rank_at(topo.node_idx - node_step, local)
        case SendOp.BROADCAST:
            # `_LocalChunkProviderBroadcast.get(step)` broadcasts from rank
            # `step` (its *own* local chunk, i.e. global chunk `step`) to
            # everyone -- the receiving rank's identity doesn't matter, mirroring
            # the forward `_BroadcastForwardChunkProvider.global_chunk_idx(step) == step`.
            return step
        case _:
            return (local_rank + step) % ring_group_size


def global_chunk_idx_backward(
    ring_config: RingAttentionConfig,
    local_rank: int,
    step: int,
    ring_group_size: int,
    group: dist.ProcessGroup,
) -> int:
    """Return the global chunk index processed at backward ``step``."""
    if ring_config.save_scope == SaveScope.GLOBAL:
        return step
    return _local_scope_global_chunk_idx(
        ring_config, local_rank, step, ring_group_size, group
    )


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
    ring_group_size: int,
    chunks_to_save: tuple[torch.Tensor, ...],
    ring_config: RingAttentionConfig,
    dropout_p: float = 0.0,
    base_seed: Optional[int] = None,
    rng_states: Optional[tuple[Optional[torch.Tensor], ...]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the backward pass of ring attention.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Gradients ``(dq, dk, dv)``.
    """
    b, h, seq_local, d = q.shape
    # Reconcile dtypes before the cores run. Under autocast the forward saves
    # bf16/fp16 tensors (``out``, ``lse`` and the K/V chunks the provider
    # replays) and ``q``/``dout`` may arrive downcast too; the cores matmul
    # ``q`` against those chunks, so a mixed-dtype pair raises. Run the cores in
    # the projection weight's (master) precision -- the same
    # backward-in-master-precision rule ``XWT`` follows when it casts
    # ``grad_output``/``input`` to ``weight.dtype`` -- so a genuinely bf16/fp16
    # (or fp64) model keeps its dtype instead of being forced to fp32.
    core_dtype = w_qkv.dtype
    dout, q = dout.to(core_dtype), q.to(core_dtype)
    out = out.to(core_dtype)
    # ``lse`` is *not* folded into ``core_dtype``: it is a softmax normaliser,
    # never matmul'd against a chunk, and the fused SDPA/TE backward kernels
    # read it as fp32 -- a 16-bit ``lse`` makes them recompute the softmax
    # wrong. Each core casts it to what its kernel wants (the ``math`` core
    # keeps whatever precision the forward produced).
    dq = torch.zeros_like(q, dtype=core_dtype)
    dk_buf, dv_buf = get_dkv_buffer(ring_config, q, ring_group_size)

    # Cases where every rank processes the *same* global chunk `step` at ring
    # step `step`: the partial dK/dV for chunk `step` just needs a SUM
    # all-reduce, with the owning rank (`local_rank == step`) keeping it. The
    # kept buffer is already the final local per-rank gradient, so it bypasses
    # `postprocess_dkv` entirely (its GATHER_DKV branch would slice a global
    # buffer this path never builds).
    #   * BROADCAST + LOCAL: rank `step` fans its chunk out each step.
    #   * any GLOBAL + SEND_DKV: the forward stored `chunks_to_save` by global
    #     chunk id, so `_GlobalChunkProvider` hands every rank global chunk
    #     `step`; SEND_DKV keeps a local-sized buffer (GATHER_DKV keeps its
    #     global buffer + final all-reduce instead).
    all_reduce_dkv = (
        ring_config.send_op == SendOp.BROADCAST
        and ring_config.save_scope == SaveScope.LOCAL
    ) or (
        ring_config.save_scope == SaveScope.GLOBAL
        and ring_config.send_backward == SendBackward.SEND_DKV
    )

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
        local_rank,
    )

    for step in range(ring_group_size):
        kj, vj, kv = chunk_provider.get(step)
        global_chunk_idx = global_chunk_idx_backward(
            ring_config, local_rank, step, ring_group_size, group
        )
        seed = (
            combine_dropout_seed(base_seed, local_rank, global_chunk_idx)
            if base_seed is not None
            else None
        )
        rng_state = rng_states[global_chunk_idx] if rng_states is not None else None
        dq_j, dk_j, dv_j = ring_config.attn_core_backward_fn(
            dout,
            q,
            kj.to(core_dtype),
            vj.to(core_dtype),
            out,
            lse,
            scale,
            seq_local,
            seq_local,
            dropout_p=dropout_p,
            seed=seed,
            rng_state=rng_state,
        )
        dq += dq_j
        del dq_j, kj, vj, kv

        if all_reduce_dkv:
            dist.all_reduce(dk_j, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(dv_j, op=dist.ReduceOp.SUM, group=group)
            if local_rank == step:
                dk_buf = dk_j
                dv_buf = dv_j
        else:
            dk_buf, dv_buf = accumulate_dkv(
                ring_config,
                dk_buf,
                dv_buf,
                dk_j,
                dv_j,
                step,
                ring_group_size,
                seq_local,
                local_rank,
                group,
            )

        # No-op for providers that don't re-circulate chunks (e.g. global).
        chunk_provider.maybe_forward(step, ring_group_size)

    if all_reduce_dkv:
        # `dk_buf`/`dv_buf` already hold this rank's final local gradient.
        return dq, dk_buf, dv_buf

    dk_local, dv_local = postprocess_dkv(
        ring_config, dk_buf, dv_buf, local_rank, seq_local, group
    )

    return dq, dk_local, dv_local


# --------------------------------------------------------------------------- #
# backward: dK/dV accumulation across the ring
# --------------------------------------------------------------------------- #
def _use_global_dkv_buffer(ring_config: RingAttentionConfig) -> bool:
    """Return whether dK and dV accumulate in a global-sized buffer."""
    return ring_config.send_backward == SendBackward.GATHER_DKV


def get_dkv_buffer(
    ring_config: RingAttentionConfig, q: torch.Tensor, ring_group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the dK and dV buffers of the accumulation strategy."""
    if _use_global_dkv_buffer(ring_config):
        shape = [*q.shape[0:2], ring_group_size * q.shape[2], *q.shape[3:]]
    else:
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
    ring_group_size: int,
    seq_local: int,
    local_rank: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate the dK and dV of one step into the buffers."""
    if not _use_global_dkv_buffer(ring_config):
        if (
            ring_config.send_op == SendOp.HYBRID_BROADCAST
            and ring_config.save_scope == SaveScope.LOCAL
        ):
            return _accumulate_dkv_hybrid_broadcast(
                ring_config, dk_buf, dv_buf, dk_j, dv_j, step, group
            )
        return _accumulate_dkv_send(
            ring_config, dk_buf, dv_buf, dk_j, dv_j, step, group
        )

    # See `global_chunk_idx_backward` for GLOBAL vs LOCAL scope; ring_group_size
    # is unused for GLOBAL scope's `step`-is-the-index shortcut but harmless.
    global_chunk_idx = global_chunk_idx_backward(
        ring_config, local_rank, step, ring_group_size, group
    )
    return _accumulate_dkv_gather(
        dk_buf, dv_buf, dk_j, dv_j, seq_local, global_chunk_idx
    )


def _accumulate_dkv_gather(
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    dk_j: torch.Tensor,
    dv_j: torch.Tensor,
    seq_local: int,
    global_chunk_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the dk and dv of one step to their slot in the global buffers.

    Parameters
    ----------
    dk_buf : torch.Tensor
        Buffer accumulating dk.
    dv_buf : torch.Tensor
        Buffer accumulating dv.
    dk_j, dv_j : torch.Tensor
        Gradients of the current step.
    seq_local : int
        Local sequence length.
    global_chunk_idx : int
        Index of the sequence chunk represented by the local gradients.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Updated dk and dv buffers.
    """
    lo, hi = global_chunk_idx * seq_local, (global_chunk_idx + 1) * seq_local
    dk_buf[:, :, lo:hi] += dk_j
    dv_buf[:, :, lo:hi] += dv_j
    return dk_buf, dv_buf


def _hybrid_dkv_relay_peers(topo: HybridTopology, step: int) -> tuple[int, int]:
    """Return the ranks to send a dK/dV partial to and receive one from."""
    node_step = step // topo.gpus_per_node
    send_to = topo.rank_at(topo.node_idx - node_step, topo.local_idx + step)
    recv_from = topo.rank_at(topo.node_idx + node_step, topo.local_idx - step)
    return send_to, recv_from


def _accumulate_dkv_send(
    ring_config: RingAttentionConfig,
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    dk_j: torch.Tensor,
    dv_j: torch.Tensor,
    step: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relay dk and dv to the rank owning the chunk.

    Parameters
    ----------
    ring_config : RingAttentionConfig
        Ring-attention configuration.
    dk_buf : torch.Tensor
        Buffer accumulating dk.
    dv_buf : torch.Tensor
        Buffer accumulating dv.
    dk_j, dv_j : torch.Tensor
        Gradients of the current step.
    step : int
        Current ring step.
    group : dist.ProcessGroup
        Process group of the ring.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Updated dk and dv buffers.
    """
    if step == 0:
        dk_buf += dk_j
        dv_buf += dv_j
        return dk_buf, dv_buf

    dkv = torch.cat([dk_j, dv_j], dim=0).contiguous()
    recv_buf = torch.empty_like(dkv)

    # hybrid
    if ring_config.send_op in _INTRA_ROTATING_HYBRIDS:
        topo = get_hybrid_topology(group, ring_config.gpus_per_node)
        send_recv_ranks = _hybrid_dkv_relay_peers(topo, step)
        recv_buf, _ = ring_config.p2p_fn(
            dkv,
            recv_buf,
            send_recv_ranks=send_recv_ranks,
            group=group,
            async_op=False,
            step=step,
        )
    else:
        recv_buf, _ = ring_config.p2p_fn(
            dkv,
            recv_buf,
            step if ring_config.send_op != SendOp.BROADCAST else -step,
            group=group,
            async_op=False,
            step=step,
        )
    dkj_r, dvj_r = torch.chunk(recv_buf, 2, dim=0)
    dk_buf += dkj_r
    dv_buf += dvj_r
    return dk_buf, dv_buf


def _accumulate_dkv_hybrid_broadcast(
    ring_config: RingAttentionConfig,
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    dk_j: torch.Tensor,
    dv_j: torch.Tensor,
    step: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate dK and dV of a hybrid broadcast step into local-sized buffers."""
    topo = get_hybrid_topology(group, ring_config.gpus_per_node)
    node_step, phase = divmod(step, topo.gpus_per_node)
    is_owner = topo.local_idx == phase

    dkv = torch.cat([dk_j, dv_j], dim=0).contiguous()
    dist.all_reduce(dkv, op=dist.ReduceOp.SUM, group=topo.intranode_group)

    if node_step != 0:
        # Inter-node ring hop as a whole-group permutation: every rank hands its
        # node-sum to its counterpart at node ``node_idx - node_step`` and
        # receives from ``node_idx + node_step``. Only node-local rank ``phase``
        # ends up with the partial for a chunk it owns; the rest take part to
        # keep the collective whole and discard what they receive.
        recv = torch.empty_like(dkv)
        recv, _ = ring_config.p2p_fn(
            dkv,
            recv,
            send_recv_ranks=(
                topo.inter_peer(-node_step),
                topo.inter_peer(node_step),
            ),
            group=group,
            async_op=False,
            step=step,
        )
        dkv = recv

    if is_owner:
        dkj_r, dvj_r = torch.chunk(dkv, 2, dim=0)
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
    """Reduce the global dk and dv buffers and select the local slice.

    Parameters
    ----------
    ring_config : RingAttentionConfig
        Ring-attention configuration.
    dk_buf : torch.Tensor
        Buffer accumulating dk.
    dv_buf : torch.Tensor
        Buffer accumulating dv.
    local_rank : int
        Rank within the ring group.
    seq_local : int
        Local sequence length.
    group : dist.ProcessGroup
        Process group of the ring.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Local dk and dv.
    """
    if _use_global_dkv_buffer(ring_config):
        dist.all_reduce(dk_buf, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(dv_buf, op=dist.ReduceOp.SUM, group=group)
        lo, hi = local_rank * seq_local, (local_rank + 1) * seq_local
        return dk_buf[:, :, lo:hi], dv_buf[:, :, lo:hi]
    return dk_buf, dv_buf
