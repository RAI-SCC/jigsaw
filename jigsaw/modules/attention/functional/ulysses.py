"""Ulysses (DeepSpeed-style) sequence-parallel attention.

An alternative to the ring (:mod:`.forward`) for parallelising attention over
``spatial_group``. Instead of circulating K/V around a ring of
``spatial_group.size()`` P2P steps -- each followed by an online-softmax merge --
Ulysses does a single all-to-all that trades the local *sequence* shard for a
*head* shard, runs **one** local attention over the full sequence on
``local_heads / spatial_group.size()`` heads, then an all-to-all back.

Trade-off vs. the ring:

* O(1) collectives instead of ``spatial_group.size()`` P2P steps, and one big
  fused attention kernel instead of as many small ones plus their LSE merges --
  much closer to roofline. This is Transformer Engine's ``cp_comm_type="a2a"``.
* It needs the head count to divide the group: ``local_heads % P == 0`` where
  ``local_heads = heads // channel_group.size()``. The ring works for any head
  count. When the a2a group crosses a slow fabric boundary (multi-node) the
  all-to-all is exposed; the hierarchical ``a2a+p2p`` scheme -- Ulysses over an
  intra-node sub-group nested in a ring over the inter-node one -- is the
  follow-on and is not implemented here yet.

Composes with channel/tensor parallelism unchanged: the all-to-all sits between
the (already channel-sharded) ``qkv`` projection and ``head_concat``, operating
on the projected Q/K/V, so it is orthogonal to how ``qkv`` was sharded.

All tensors here are ``bshd`` -- ``[batch, seq, heads, head_dim]`` -- the layout
the TE cores speak natively; the SDPA path transposes around its kernel call in
:meth:`..attention.DistributedAttention._ulysses_attn_core`.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.distributed as dist


def _scatter_heads_gather_seq(
    x: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """``[b, s_local, h, d] -> [b, s_local * P, h // P, d]`` via one all-to-all.

    Sends head-block ``p`` of the local sequence to rank ``p`` and concatenates
    the sequence shards received from every rank, so the result is
    sequence-complete on ``h // P`` heads. ``P = group.size()``; the global
    sequence order is rank-0 tokens, rank-1 tokens, ... which is exactly how the
    sequence was sharded across ``group``.
    """
    p = dist.get_world_size(group)
    if p == 1:
        return x
    b, s, h, d = x.shape
    assert h % p == 0, f"ulysses all-to-all: heads {h} not divisible by group size {p}"
    # [b, s, P, h//P, d] -> [P, b, s, h//P, d]
    send = x.reshape(b, s, p, h // p, d).permute(2, 0, 1, 3, 4).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    # recv[src] is src's local sequence on our head-slice; stack seq by source.
    return recv.permute(1, 0, 2, 3, 4).reshape(b, p * s, h // p, d).contiguous()


def _scatter_seq_gather_heads(
    x: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """``[b, s_full, h_p, d] -> [b, s_full // P, h_p * P, d]`` -- inverse of
    :func:`_scatter_heads_gather_seq`.

    Sends sequence-block ``p`` (keeping our ``h_p`` heads) to rank ``p`` and
    concatenates the head shards received from every rank, so the result is
    head-complete on ``s_full // P`` tokens with the heads back in source-rank
    order (reconstructing the pre-Ulysses head layout).
    """
    p = dist.get_world_size(group)
    if p == 1:
        return x
    b, s, h_p, d = x.shape
    assert s % p == 0, f"ulysses all-to-all: seq {s} not divisible by group size {p}"
    # [b, P, s//P, h_p, d] -> [P, b, s//P, h_p, d]
    send = x.reshape(b, p, s // p, h_p, d).permute(1, 0, 2, 3, 4).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    # recv[src] is our sequence-block on src's heads; stack heads by source.
    return recv.permute(1, 2, 0, 3, 4).reshape(b, s // p, h_p * p, d).contiguous()


class _UlyssesAllToAll(torch.autograd.Function):
    """Autograd-aware Ulysses all-to-all.

    ``gather_seq=True`` goes sequence-parallel -> head-parallel
    (:func:`_scatter_heads_gather_seq`); ``False`` is the inverse. The backward
    of one direction is the forward of the other, so gradients flow straight
    through without saving anything but the direction and the group.
    """

    @staticmethod
    def forward(
        x: torch.Tensor, gather_seq: bool, group: dist.ProcessGroup
    ) -> torch.Tensor:
        return (
            _scatter_heads_gather_seq(x, group)
            if gather_seq
            else _scatter_seq_gather_heads(x, group)
        )

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:  # type: ignore[no-untyped-def]
        _x, gather_seq, group = inputs
        ctx.gather_seq = gather_seq
        ctx.group = group

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[no-untyped-def]
        grad_in = (
            _scatter_seq_gather_heads(grad_out, ctx.group)
            if ctx.gather_seq
            else _scatter_heads_gather_seq(grad_out, ctx.group)
        )
        return grad_in, None, None


AttnCore = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: dist.ProcessGroup,
    core: AttnCore,
) -> torch.Tensor:
    """Run ``core`` on the full sequence via a head-parallel Ulysses all-to-all.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Local, sequence-sharded projections, ``bshd``
        ``[b, seq // P, local_heads, head_dim]`` (``P = group.size()``).
    group : dist.ProcessGroup
        The all-to-all group to gather the sequence over -- the whole spatial
        group for flat Ulysses, or the intra-node sub-group for the
        hierarchical scheme.
    core : callable
        ``core(q, k, v) -> out`` running attention over the sequence the
        all-to-all made local, on ``bshd`` tensors, and returning ``bshd``
        ``[b, seq_after_a2a, local_heads // P, head_dim]``. For flat Ulysses
        this is a single dense kernel over the full sequence; for the
        hierarchical scheme it is a P2P ring over the inter-node group.

    Returns
    -------
    torch.Tensor
        ``bshd`` ``[b, seq // P, local_heads, head_dim]`` -- same layout and
        sharding as ``q``.
    """
    q = _UlyssesAllToAll.apply(q, True, group)
    k = _UlyssesAllToAll.apply(k, True, group)
    v = _UlyssesAllToAll.apply(v, True, group)
    out = core(q, k, v)
    return _UlyssesAllToAll.apply(out, False, group)


def hierarchical_ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    intra_group: dist.ProcessGroup,
    ring_core: AttnCore,
) -> torch.Tensor:
    """Ulysses over an intra-node sub-group, a P2P ring over the inter-node one.

    The mirror of Transformer Engine's ``cp_comm_type="a2a+p2p"``: the
    ``intra_group`` all-to-all (fast, NVLink) scatters heads and gathers one
    node's worth of sequence, so ``ring_core`` -- a ring over the inter-node
    group -- circulates only ``n_nodes`` K/V chunks instead of the full spatial
    ring, and its inner attention runs on ``local_heads // intra_group.size()``
    heads over ``seq // n_nodes`` tokens.

    ``q``/``k``/``v`` are ``bshd`` ``[b, seq // spatial, local_heads, d]``;
    ``ring_core`` gets ``[b, seq // n_nodes, local_heads // A, d]`` (``A =
    intra_group.size()``) and must return the same shape. Needs
    ``local_heads % A == 0``.
    """
    return ulysses_attention(q, k, v, intra_group, ring_core)
