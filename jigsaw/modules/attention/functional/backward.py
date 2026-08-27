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
        
        # Cases where every rank processes the *same* global chunk `step` at
        # ring step `step`: the partial dK/dV for chunk `step` just needs a SUM
        # all-reduce, with the owning rank (`local_rank == step`) keeping it.
        #   * BROADCAST + LOCAL: rank `step` fans its chunk out each step.
        #   * any GLOBAL + SEND_DKV: the forward stored `chunks_to_save` by
        #     global chunk id, so `_GlobalChunkProvider` hands every rank
        #     global chunk `step`; SEND_DKV keeps a local-sized buffer, so an
        #     all-reduce is the natural fold (GATHER_DKV keeps its global
        #     buffer + final all-reduce instead).
        all_reduce_dkv = (
            ring_config.send_op == SendOp.BROADCAST
            and ring_config.save_scope == SaveScope.LOCAL
        ) or (
            ring_config.save_scope == SaveScope.GLOBAL
            and ring_config.send_backward == SendBackward.SEND_DKV
        )
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
        
    dk_local, dv_local = postprocess_dkv(
        ring_config, dk_buf, dv_buf, local_rank, seq_local, group
    )

    return dq, dk_local, dv_local


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
            if ring_config.save_scope == SaveScope.GLOBAL:
                # The forward stored `chunks_to_save` by global chunk id and
                # `_GlobalChunkProvider` looks them up directly, so ring step
                # `step` == global chunk index for every send_op.
                global_chunk_idx = step
            elif ring_config.send_op == SendOp.HYBRID_P2P:
                # LOCAL hybrid: chunks re-circulate through the node/local ring.
                return _accumulate_dkv_hybrid(
                    dk_buf, dv_buf, dk_j, dv_j, step, seq_local, local_rank, world_size
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
        case SendBackward.SEND_DKV:
            # if ring_config.send_op == SendOp.BROADCAST:
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
    # TODO: gpus per node as variable instead of hard-coded
    # 0 1 2 3 4 5 6 7
    
    node, local = divmod(local_rank, 4)
    if step % 4 == 0:  # temporarily block out this path
        # node-dimension ring, local index fixed
        send_node = (node + (step)) % world_size // 4
        send_to = send_node * 4 + local
        if local_rank >= 4 and step == 4:
            send_to = (local_rank - 4) % 4
        elif step == 4:
            send_to = local_rank + 4
        if step == 0:
            send_to = local_rank
    else:
        # local-dimension ring, node fixed
        base_node_rank = local_rank // 4
        send_to = (local_rank + step) % 4 + base_node_rank * 4

        if step > 4 and local_rank >= 4:
            send_to -= 4
        elif step > 4 and local_rank < 4:
            send_to += 4

    global_chunk_idx = send_to
    lo, hi = global_chunk_idx * seq_local, (global_chunk_idx + 1) * seq_local
    dk_buf[:, :, lo:hi] += dk_j
    dv_buf[:, :, lo:hi] += dv_j
    return dk_buf, dv_buf


def get_send_recv_ranks(
    rank: int, step: int, nodes: int, gpus_per_node: int
) -> tuple[int, int]:
    """Return the send and receive ranks for a hybrid backward step."""
    node, local = divmod(rank, gpus_per_node)
    if step % gpus_per_node == 0:  # temporarily block out this path
        # node-dimension ring, local index fixed
        send_node = (node + (step + 1)) % nodes
        recv_node = (node - (step + 1)) % nodes
        send_to = send_node * gpus_per_node + local
        recv_from = recv_node * gpus_per_node + local
    else:
        # local-dimension ring, node fixed

        base_node_rank = rank // gpus_per_node
        send_to = (rank + step) % gpus_per_node + base_node_rank * gpus_per_node
        recv_from = (rank - step) % gpus_per_node + base_node_rank * gpus_per_node

        if step > 4 and rank >= 4:
            send_to -= 4
            recv_from -= 4
        elif step > 4 and rank < 4:
            send_to += 4
            recv_from += 4

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
    if ring_config.send_op == SendOp.HYBRID_P2P:
        send_recv_ranks = get_send_recv_ranks(
            dist.get_rank(group), step=step, nodes=group.size() // 4, gpus_per_node=4
        )
        print(f'step {step} send_recv_ranks {send_recv_ranks}')
        if step % 4 != 0:
            # groups are not set correctly
            # since we artificially have a group of size 4
            # rank 6 needs to send to 5 in the beginning, and not 3
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
            # print(f'DKV: R{dist.get_rank()}  step {step} sending to {send_recv_ranks[0]} recv from {send_recv_ranks[1]}')
            recv_buf, _ = ring_config.p2p_fn(
                dkv,
                recv_buf,
                send_recv_ranks=send_recv_ranks,
                group=group,
                async_op=False,
                empty=empty,
                step=step,
                group_size=8,
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
    match ring_config.send_backward:
        case SendBackward.GATHER_DKV:
            dist.all_reduce(dk_buf, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(dv_buf, op=dist.ReduceOp.SUM, group=group)
            lo, hi = local_rank * seq_local, (local_rank + 1) * seq_local
            return dk_buf[:, :, lo:hi], dv_buf[:, :, lo:hi]
        case SendBackward.SEND_DKV:
            return dk_buf, dv_buf
