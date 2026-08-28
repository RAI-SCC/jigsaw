"""Backward implementations for ring attention."""

import torch
import torch.distributed as dist

from jigsaw.modules.attention.config import (
    RingAttentionConfig,
    SaveScope,
    SendBackward,
    SendOp,
)

from .backward_chunk_provider import _build_backward_chunk_provider
from .functional import HybridTopology, get_hybrid_topology

# Hybrid send-ops whose backward chunk provider rotates chunks intra-node as well
# as inter-node, so a step's dK/dV belongs to local index ``local_idx + step``
# rather than ``step`` (HYBRID_BROADCAST leaves the chunk put intra-node).
_INTRA_ROTATING_HYBRIDS = frozenset({SendOp.HYBRID_P2P, SendOp.HYBRID_ALLTOALL})

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for ring attention."""
    b, h, seq_local, d = q.shape
    dq = torch.zeros_like(q, dtype=torch.float32)
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
    """Whether dK/dV accumulate into a global-sized buffer + a final all-reduce.

    True for GATHER_DKV always. HYBRID_BROADCAST + LOCAL folds into a
    local-sized buffer instead (see :func:`_accumulate_dkv_hybrid_broadcast`):
    an intra-node all-reduce collapses the per-node partials, then the node-sum
    is relayed to its owning node over the inter-node ring.
    """
    return ring_config.send_backward == SendBackward.GATHER_DKV


def get_dkv_buffer(
    ring_config: RingAttentionConfig, q: torch.Tensor, ring_group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the dK, dV buffers appropriate for the accumulation strategy.

    Global buffer (see :func:`_use_global_dkv_buffer`): one global-sized buffer
        that every step writes its chunk into at the right offset, all-reduced
        at the end.
    Otherwise (SEND_DKV relay): a local-sized buffer that dK/dV are added into,
        with partial sums P2P-relayed around the ring each step.
    """
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
    """Fold this step's locally computed dk_j, dv_j into the running dK, dV."""
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

    if ring_config.save_scope == SaveScope.GLOBAL:
        # The forward stored `chunks_to_save` by global chunk id and
        # `_GlobalChunkProvider` looks them up directly, so ring step
        # `step` == global chunk index for every send_op.
        global_chunk_idx = step
    else:
        # LOCAL scope: the backward chunk provider re-circulates chunks, so the
        # global chunk this step's dK/dV belongs to depends on the send_op.
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
                global_chunk_idx = topo.rank_at(topo.node_idx - node_step, local)
            case SendOp.BROADCAST:
                global_chunk_idx = (local_rank - step) % ring_group_size
            case _:
                global_chunk_idx = (local_rank + step) % ring_group_size
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
    """
    Scatter-add this step's dk, dv into their slot of the global buffer.

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
        Local result for dv for step j.
    seq_local: int
        Local sequence length.
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


def _accumulate_dkv_send(
    ring_config: RingAttentionConfig,
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    dk_j: torch.Tensor,
    dv_j: torch.Tensor,
    step: int,
    group: dist.ProcessGroup,
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
    """Fold a HYBRID_BROADCAST + LOCAL step's dK/dV into local-sized buffers.

    The intra-node broadcast makes every rank on a node compute a partial for
    the *same* chunk ``(node_idx - node_step, phase)``, so the ``gpus_per_node``
    partials are first summed with an intra-node all-reduce. When that chunk
    lives on another node (``node_step != 0``) the node-sum is relayed to it
    over the inter-node ring, and the mirror node hands back the partial for our
    own chunk. Only the chunk owner -- node-local rank ``phase`` -- keeps a
    partial.
    """
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
