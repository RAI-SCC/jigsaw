"""Ring attention forward implementations."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

from jigsaw.modules.attention.config import (
    RingAttentionConfig,
    SaveScope,
    SendForward,
    SendOp,
)

from .functional import _compute_kv, get_hybrid_topology


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def _online_softmax_merge(
    o1: torch.Tensor, l1: torch.Tensor, o2: torch.Tensor, l2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two partial online-softmax attention results.

    Parameters
    ----------
    o1 : torch.Tensor
        Output tensor for the first partial result.
    l1 : torch.Tensor
        Log-sum-exp values for the first partial result.
    o2 : torch.Tensor
        Output tensor for the second partial result.
    l2 : torch.Tensor
        Log-sum-exp values for the second partial result.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Result returned by the operation.
    """
    l_new = torch.logaddexp(l1, l2)
    e1 = torch.exp(l1 - l_new)
    e2 = torch.exp(l2 - l_new)
    o_new = e1.unsqueeze(-1) * o1 + e2.unsqueeze(-1) * o2
    return o_new, l_new


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


# --------------------------------------------------------------------------- #
# forward implementations for ring attention
# --------------------------------------------------------------------------- #


def build_ring_forward(
    ring_config: RingAttentionConfig,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor, tuple]]:
    """Pick the forward implementation (P2P vs broadcast) based on config."""
    match ring_config.send_op:
        case SendOp.BROADCAST:
            return broadcast_forward
        case SendOp.P2P | SendOp.ALLTOALL:
            return ring_forward
        case SendOp.HYBRID_P2P:
            return hybrid_p2p_forward
        case SendOp.HYBRID_BROADCAST:
            return hybrid_broadcast_forward
        case _:
            raise NotImplementedError(f"Unknown send_op: {ring_config.send_op}")


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
    ring_group_size: int,
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
        chunks_to_save = [None] * ring_group_size  # filled in per-step below

    # Kick off first send
    recv_buf, handle = ring_config.forward_send_fn(
        x=x, kv=kv, local_rank=local_rank, group=group, src_rank=0, empty=empty
    )
    handle.wait()
    x_j, kv_j, kj, vj = ring_config.forward_recv_fn(recv_buf, x_j, kv_j, kj, vj)

    for step in range(ring_group_size):
        if step + 1 < ring_group_size:
            recv_buf, handle = ring_config.forward_send_fn(
                x=x,
                kv=kv,
                local_rank=local_rank,
                group=group,
                src_rank=(step + 1) % group.size(),
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
            # `x_j` at loop iteration `step` is global chunk `step` (identical on
            # every rank: it was received from the `src_rank=step` broadcast).
            chunks_to_save[step] = ring_config.save_for_backward_fn(x_j, kv_j)

        out, lse = _attend_step(ring_config, q, kj, vj, scale, dropout_p, out, lse)
        if step < ring_group_size - 1:
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
    ring_group_size: int,
    ring_config: RingAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    """Forward pass for ring attention using point-to-point sends."""
    b, heads, n_local, d = q.shape
    out = torch.zeros_like(q)
    lse = torch.full((b, heads, n_local), float("-inf"), device=q.device, dtype=q.dtype)

    x_j = x

    # Local KV
    kv = torch.concat([k.unsqueeze(0), v.unsqueeze(0)])
    kv_j, kj, vj = kv, k, v

    # LOCAL: only own (step-0) chunk is kept; backward re-rings the rest.
    # GLOBAL: keep every chunk, indexed by *global chunk id* so the backward
    # `_GlobalChunkProvider` (a plain `chunks[step]` lookup) sees ring step
    # == global chunk index.
    chunks_to_save: list[torch.Tensor | None]
    if ring_config.save_scope == SaveScope.GLOBAL:
        chunks_to_save = [None] * ring_group_size
    else:
        chunks_to_save = []

    # Kick off first send
    recv_buf, handle = ring_config.forward_send_fn(
        x_j, kv_j, direction=-1, group=group, tag=0, empty=empty
    )

    for step in range(ring_group_size):
        if step > 0 and ring_config.send_in_forward == SendForward.X:
            kv_j, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)

        if ring_config.save_scope == SaveScope.GLOBAL:
            # `direction=-1` ring: after `step` receives, `x_j` originates from
            # rank `(local_rank + step) % ring_group_size`.
            chunks_to_save[(local_rank + step) % ring_group_size] = (
                ring_config.save_for_backward_fn(x_j, kv_j)
            )
        elif step == 0:
            chunks_to_save.append(ring_config.save_for_backward_fn(x_j, kv_j))

        out, lse = _attend_step(ring_config, q, kj, vj, scale, dropout_p, out, lse)
        if step < ring_group_size - 1:
            handle.wait()
            # receive either x or kv_j
            x_j, kv_j, kj, vj = ring_config.forward_recv_fn(recv_buf, x_j, kv_j, kj, vj)
            recv_buf, handle = ring_config.forward_send_fn(
                x_j, kv_j, direction=-1, group=group, tag=step, empty=empty
            )

    ring_config.wait_offloads()
    out = out.to(q.dtype)
    return out, lse, tuple(chunks_to_save)


def hybrid_p2p_forward(
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
    ring_group_size: int,
    ring_config: RingAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    """Forward pass for ring attention with a hybrid intra-/inter-node ring.

    Each "node round" rotates a chunk through the ``gpus_per_node`` ranks of a
    node with an intra-node P2P ring, then hops to the next node with a single
    inter-node P2P send. Peers come from :class:`HybridTopology`, so there is no
    ``// gpus_per_node`` arithmetic here.
    """
    b, heads, n_local, d = q.shape
    topo = get_hybrid_topology(group, ring_config.gpus_per_node)
    gpn, n_nodes = topo.gpus_per_node, topo.n_nodes

    # Intra-node ring: send to the previous local rank, receive from the next.
    # Inter-node hop: send to the previous node, receive from the next.
    intra_ranks = [topo.intra_peer(-1), topo.intra_peer(1)]
    inter_ranks = [topo.inter_peer(-1), topo.inter_peer(1)]

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
        chunks_to_save = [None] * ring_group_size  # filled in per-step below

    for node_step in range(n_nodes):
        # Kick off this round's inter-node send of the original local chunk.
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
                send_recv_ranks=inter_ranks,
                group=group,
                async_op=True,
                tag=node_step,
                empty=empty,
            )
        for step in range(gpn):
            if step + 1 < gpn:
                recv_buf = (
                    torch.empty_like(x_j)
                    if ring_config.send_in_forward == SendForward.X
                    else torch.empty_like(kv_j)
                )
                # Start communicating information in a ring
                recv_buf, handle = ring_config.send_op_fn(
                    x_j if ring_config.send_in_forward == SendForward.X else kv_j,
                    recv_buf,
                    send_recv_ranks=intra_ranks,
                    group=group,
                    tag=step,
                )

            if ring_config.send_in_forward == SendForward.X:
                kv_j, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)

            if ring_config.save_scope == SaveScope.GLOBAL:
                # `x_j`/`kv_j` currently hold the chunk owned by the ring rank
                #   (node_idx + node_step) % n_nodes, (local_idx + step) % gpn
                # -- the node index advances by one per inter-node hop and the
                # local index by one per intra-node step. Store it at that
                # global-chunk slot so the backward `_GlobalChunkProvider` (a
                # plain `chunks[step]` lookup) sees ring step == global chunk.
                cur_node = (topo.node_idx + node_step) % n_nodes
                global_chunk_idx = cur_node * gpn + (topo.local_idx + step) % gpn
                chunks_to_save[global_chunk_idx] = ring_config.save_for_backward_fn(
                    x_j, kv_j
                )
            out, lse = _attend_step(ring_config, q, kj, vj, scale, dropout_p, out, lse)
            if step < gpn - 1:
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


def hybrid_broadcast_forward(
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
    ring_group_size: int,
    ring_config: RingAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    """Forward pass: intra-node broadcast, inter-node P2P.

    Structured like :func:`hybrid_p2p_forward` -- an outer loop over node rounds
    with one inter-node P2P hop each -- but the intra-node ring is replaced by a
    per-round sweep of ``gpus_per_node`` broadcasts (à la :func:`broadcast_forward`):
    at inner step ``s`` the node-local rank ``s`` broadcasts its chunk to the
    node. Peers/sub-groups come from :class:`HybridTopology`, so there is no
    ``// gpus_per_node`` arithmetic here.
    """
    b, heads, n_local, d = q.shape
    topo = get_hybrid_topology(group, ring_config.gpus_per_node)
    gpn, n_nodes = topo.gpus_per_node, topo.n_nodes
    intranode_group = topo.intranode_group
    send_x = ring_config.send_in_forward == SendForward.X

    # Inter-node hop: send to the previous node, receive from the next (mirrors
    # hybrid_p2p_forward).
    inter_ranks = [topo.inter_peer(-1), topo.inter_peer(1)]

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
        chunks_to_save = [None] * ring_group_size  # filled in per-step below

    for node_step in range(n_nodes):
        cur_node = (topo.node_idx + node_step) % n_nodes

        # Kick off this round's inter-node send of the original local chunk.
        if node_step < n_nodes - 1:
            internode_recv_buf = torch.empty_like(x_j if send_x else kv_j)
            internode_recv_buf, internode_handle = ring_config.p2p_fn(
                data=(x.clone() if send_x else kv.clone()),
                buffer=internode_recv_buf,
                send_recv_ranks=inter_ranks,
                group=group,
                async_op=True,
                tag=node_step,
                empty=empty,
            )

        # This rank's chunk for the round: broadcast unchanged when it is our
        # turn (mirrors broadcast_forward's fixed local x/kv).
        own_x, own_kv = x_j, kv_j
        own_buf = own_x if send_x else own_kv

        for step in range(gpn):
            src = topo.rank_at(topo.node_idx, step)
            is_src = topo.local_idx == step
            recv_buf = own_buf if is_src else torch.empty_like(own_buf)
            dist.broadcast(recv_buf, src=src, group=intranode_group)

            if send_x:
                x_j = recv_buf
                if is_src:
                    kv_j, kj, vj = own_kv, own_kv[0], own_kv[1]
                else:
                    kv_j, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)
            else:
                kv_j, kj, vj = recv_buf, recv_buf[0], recv_buf[1]

            if ring_config.save_scope == SaveScope.GLOBAL:
                # Broadcast source is node-local index `step`, so the chunk seen
                # here is global chunk `cur_node * gpn + step` -- identical on
                # every node rank. Store it so the backward `_GlobalChunkProvider`
                # (a plain `chunks[step]` lookup) sees ring step == global chunk.
                chunks_to_save[cur_node * gpn + step] = (
                    ring_config.save_for_backward_fn(x_j, kv_j)
                )

            out, lse = _attend_step(ring_config, q, kj, vj, scale, dropout_p, out, lse)

        # --- consume the internode data that arrived during the round ---
        if node_step < n_nodes - 1:
            internode_handle.wait()
            x_j, kv_j, kj, vj = ring_config.forward_recv_fn(
                internode_recv_buf, x_j, kv_j, kj, vj
            )
            if send_x:
                kv_j, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)

    ring_config.wait_offloads()
    out = out.to(q.dtype)
    return out, lse, tuple(chunks_to_save)
