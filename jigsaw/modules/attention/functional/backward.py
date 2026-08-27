"""Backward implementations for ring attention."""

import torch
import torch.distributed as dist

from jigsaw.modules.attention.config import (
    RingAttentionConfig,
    SaveScope,
    SendBackward,
    SendOp,
)

from .chunk_provider import (
    _build_backward_chunk_provider,
    _LocalChunkProvider,
    _LocalChunkProviderHybridBroadcast,
)
from .functional import HybridTopology, get_hybrid_topology

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
                world_size,
                seq_local,
                local_rank,
                group,
                empty,
            )

        if isinstance(chunk_provider, _LocalChunkProvider) or isinstance(
            chunk_provider, _LocalChunkProviderHybridBroadcast
        ):
            chunk_provider.maybe_forward(step, world_size)

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
    """Whether dK/dV accumulate into a global-sized buffer + a final all-reduce.

    True for GATHER_DKV always, and for HYBRID_BROADCAST + LOCAL regardless of
    ``send_backward``: its intra-node broadcast makes every node rank produce a
    partial for the *same* chunk, so a positional scatter + final all-reduce is
    the natural fold and there is no meaningful P2P relay. (Mirrors how
    BROADCAST + LOCAL ignores ``send_backward``.)
    """
    if ring_config.send_backward == SendBackward.GATHER_DKV:
        return True
    return (
        ring_config.send_op == SendOp.HYBRID_BROADCAST
        and ring_config.save_scope == SaveScope.LOCAL
    )


def get_dkv_buffer(
    ring_config: RingAttentionConfig, q: torch.Tensor, world_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the dK, dV buffers appropriate for the accumulation strategy.

    Global buffer (see :func:`_use_global_dkv_buffer`): one global-sized buffer
        that every step writes its chunk into at the right offset, all-reduced
        at the end.
    Otherwise (SEND_DKV relay): a local-sized buffer that dK/dV are added into,
        with partial sums P2P-relayed around the ring each step.
    """
    if _use_global_dkv_buffer(ring_config):
        shape = [*q.shape[0:2], world_size * q.shape[2], *q.shape[3:]]
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
    world_size: int,
    seq_local: int,
    local_rank: int,
    group: dist.ProcessGroup,
    empty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold this step's locally computed dk_j, dv_j into the running dK, dV."""
    if not _use_global_dkv_buffer(ring_config):
        return _accumulate_dkv_send(
            ring_config, dk_buf, dv_buf, dk_j, dv_j, step, group, empty
        )

    if ring_config.save_scope == SaveScope.GLOBAL:
        # The forward stored `chunks_to_save` by global chunk id and
        # `_GlobalChunkProvider` looks them up directly, so ring step
        # `step` == global chunk index for every send_op.
        global_chunk_idx = step
    elif ring_config.send_op in (SendOp.HYBRID_P2P, SendOp.HYBRID_ALLTOALL):
        # LOCAL hybrid: chunks re-circulate through the node/local ring.
        topo = get_hybrid_topology(group, ring_config.gpus_per_node)
        return _accumulate_dkv_hybrid(dk_buf, dv_buf, dk_j, dv_j, step, seq_local, topo)
    elif ring_config.send_op == SendOp.HYBRID_BROADCAST:
        # LOCAL hybrid broadcast: every node rank saw the same chunk this step.
        topo = get_hybrid_topology(group, ring_config.gpus_per_node)
        return _accumulate_dkv_hybrid_broadcast(
            dk_buf, dv_buf, dk_j, dv_j, step, seq_local, topo
        )
    elif ring_config.send_op == SendOp.BROADCAST:
        global_chunk_idx = (local_rank - step) % world_size
    else:
        global_chunk_idx = (local_rank + step) % world_size
    return _accumulate_dkv_gather(
        dk_buf,
        dv_buf,
        dk_j,
        dv_j,
        step,
        seq_local,
        local_rank,
        world_size,
        global_chunk_idx,
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
    global_chunk_idx: int,
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
    global_chunk_idx: int
        Index of the sequence chunk represented by the local gradients.

    Returns
    -------
    dk_buf, dv_buf: torch.Tensor, torch.Tensor
        Accumulated buffers for dk and dv.
    """
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
    topo: HybridTopology,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add this step's dK/dV into the global buffer at the right chunk slot.

    Used for LOCAL-scope hybrid + GATHER_DKV, where the backward chunk provider
    re-circulates chunks: after ``step`` steps this rank holds the chunk owned
    by ``node_idx - step // gpn`` (the inter-node ring runs backwards relative
    to the forward), ``local_idx + step``.

    Parameters
    ----------
    dk_buf, dv_buf : torch.Tensor
        Global-sized buffers accumulating dK / dV.
    dk_j, dv_j : torch.Tensor
        This step's local dK / dV partials.
    step : int
        Backward ring step.
    seq_local : int
        Local sequence length.
    topo : HybridTopology
        Node/rank layout of the ring.

    Returns
    -------
    dk_buf, dv_buf : torch.Tensor
        The updated buffers.
    """
    node_step = step // topo.gpus_per_node
    global_chunk_idx = topo.rank_at(topo.node_idx - node_step, topo.local_idx + step)
    lo, hi = global_chunk_idx * seq_local, (global_chunk_idx + 1) * seq_local
    dk_buf[:, :, lo:hi] += dk_j
    dv_buf[:, :, lo:hi] += dv_j
    return dk_buf, dv_buf


def _accumulate_dkv_hybrid_broadcast(
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    dk_j: torch.Tensor,
    dv_j: torch.Tensor,
    step: int,
    seq_local: int,
    topo: HybridTopology,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add this step's dK/dV into the global buffer at the right chunk slot.

    Used for LOCAL-scope HYBRID_BROADCAST. The backward chunk provider
    broadcasts the chunk owned by node-local rank ``step % gpn`` of the node
    ``node_idx - step // gpn`` (the inter-node ring runs backwards relative to
    the forward), so -- unlike :func:`_accumulate_dkv_hybrid` -- the local index
    is the broadcast source, not ``local_idx + step``.

    Parameters
    ----------
    dk_buf, dv_buf : torch.Tensor
        Global-sized buffers accumulating dK / dV.
    dk_j, dv_j : torch.Tensor
        This step's local dK / dV partials.
    step : int
        Backward ring step.
    seq_local : int
        Local sequence length.
    topo : HybridTopology
        Node/rank layout of the ring.

    Returns
    -------
    dk_buf, dv_buf : torch.Tensor
        The updated buffers.
    """
    node_step = step // topo.gpus_per_node
    global_chunk_idx = topo.rank_at(topo.node_idx - node_step, step)
    lo, hi = global_chunk_idx * seq_local, (global_chunk_idx + 1) * seq_local
    dk_buf[:, :, lo:hi] += dk_j
    dv_buf[:, :, lo:hi] += dv_j
    return dk_buf, dv_buf


def _hybrid_dkv_relay_peers(topo: HybridTopology, step: int) -> tuple[int, int]:
    """(send_to, recv_from) for relaying a dK/dV partial to its home rank.

    At backward ``step`` this rank holds the gradient for the chunk owned by
    ``rank_at(node_idx - step // gpn, local_idx + step)`` -- hand it straight to
    that owner, and receive (from the mirror-image rank) the partial destined
    for our own chunk.
    """
    node_step = step // topo.gpus_per_node
    send_to = topo.rank_at(topo.node_idx - node_step, topo.local_idx + step)
    recv_from = topo.rank_at(topo.node_idx + node_step, topo.local_idx - step)
    return send_to, recv_from


def _accumulate_dkv_send_broadcast(
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
    dist.all_reduce(dk_j, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(dv_j, op=dist.ReduceOp.SUM, group=group)
    return dk_j, dv_j


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
    if ring_config.send_op in (SendOp.HYBRID_P2P, SendOp.HYBRID_ALLTOALL):
        topo = get_hybrid_topology(group, ring_config.gpus_per_node)
        send_recv_ranks = _hybrid_dkv_relay_peers(topo, step)
        recv_buf, _ = ring_config.p2p_fn(
            dkv,
            recv_buf,
            send_recv_ranks=send_recv_ranks,
            group=group,
            async_op=False,
            empty=empty,
            step=step,
        )
    else:
        recv_buf, _ = ring_config.p2p_fn(
            dkv,
            recv_buf,
            step if ring_config.send_op != SendOp.BROADCAST else -step,
            group=group,
            async_op=False,
            empty=empty,
            step=step,
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
    if _use_global_dkv_buffer(ring_config):
        dist.all_reduce(dk_buf, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(dv_buf, op=dist.ReduceOp.SUM, group=group)
        lo, hi = local_rank * seq_local, (local_rank + 1) * seq_local
        return dk_buf[:, :, lo:hi], dv_buf[:, :, lo:hi]
    return dk_buf, dv_buf
