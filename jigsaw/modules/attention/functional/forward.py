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
from .functional import _compute_kv, get_send_recv_ranks_hybrid


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
        case SendOp.HYBRID_P2P:
            return hybrid_p2p_forward
        case SendOp.P2P | SendOp.ALLTOALL:
            return ring_forward
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
        x_j, kv_j, direction=-1, group=group, tag=0, empty=empty
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
        send_recv_ranks = get_send_recv_ranks_hybrid(
            local_rank, gpus_per_node=gpus_per_node
        )
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


def hybrid_broadcast_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    scale: float,
    dropout_p: float,
    internode_group: dist.ProcessGroup,
    intranode_group: dist.ProcessGroup,
    empty: torch.Tensor,
    local_rank: int,
    world_size: int,
    ring_config: RingAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    """Forward pass for ring attention: intra-node broadcast, inter-node P2P."""
    b, heads, n_local, d = q.shape
    gpus_per_node = 4  # TODO: indicate the number of GPUs per node
    n_nodes = world_size // gpus_per_node
    intranode_rank = local_rank % gpus_per_node

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
        # Kick off first send across nodes (unchanged)
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
                group=internode_group,
                async_op=True,
                tag=node_step,
                empty=empty,
            )

        # own_x/own_kv: this rank's current chunk for the round (fixed for the
        # duration of the intra-node broadcast, mirrors broadcast_forward's x/kv)
        own_x, own_kv = x_j, kv_j

        # Kick off first intra-node broadcast
        recv_buf, handle = ring_config.forward_send_fn(
            x=own_x,
            kv=own_kv,
            local_rank=intranode_rank,
            group=intranode_group,
            src_rank=0,
            empty=empty,
        )
        handle.wait()
        x_j, kv_j, kj, vj = ring_config.forward_recv_fn(recv_buf, x_j, kv_j, kj, vj)

        for step in range(gpus_per_node):
            if step + 1 < gpus_per_node:
                recv_buf, handle = ring_config.forward_send_fn(
                    x=own_x,
                    kv=own_kv,
                    local_rank=intranode_rank,
                    group=intranode_group,
                    src_rank=step + 1,
                    empty=empty,
                )

            if ring_config.send_in_forward == SendForward.X:
                if step != intranode_rank:
                    kv_j, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)
                else:
                    kv_j, kj, vj = own_kv, own_kv[0], own_kv[1]

            if ring_config.save_scope == SaveScope.GLOBAL:
                global_step = node_step * gpus_per_node + step
                chunks_to_save[(local_rank - global_step) % world_size] = (
                    ring_config.save_for_backward_fn(x_j, kv_j)
                )

            out, lse = _attend_step(ring_config, q, kj, vj, scale, dropout_p, out, lse)
            if step < gpus_per_node - 1:
                handle.wait()

            x_j, kv_j, kj, vj = ring_config.forward_recv_fn(recv_buf, x_j, kv_j, kj, vj)

        # --- consume the internode data that arrived during the intra-node round ---
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
