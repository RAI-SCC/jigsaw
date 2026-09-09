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
  all-to-all is exposed; :func:`hierarchical_ulysses_attention` -- Ulysses over
  an intra-node sub-group nested in a ring over the inter-node one -- is the
  ``cp_comm_type="a2a+p2p"`` analogue that keeps the a2a on the fast fabric.

The three Q/K/V shards ride **one** all-to-all: the scatter/gather helpers take
arbitrary leading dims, so a ``[3, b, s, h, d]`` stack goes through a single
``all_to_all_single`` instead of three.

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


def _a2a_heads_to_seq(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """``[..., s_local, h, d] -> [..., s_local * P, h // P, d]`` via one all-to-all.

    Scatters the head axis (``-2``) across ``group`` and gathers the sequence
    axis (``-3``): sends head-block ``p`` of the local sequence to rank ``p`` and
    stacks the sequence shards received from every rank in source-rank order --
    exactly how the sequence was sharded across ``group``. Leading dims (batch,
    or a ``3`` for a stacked Q/K/V) are carried through untouched.
    """
    p = dist.get_world_size(group)
    if p == 1:
        return x
    *lead, s, h, d = x.shape
    nlead = len(lead)
    assert h % p == 0, f"ulysses all-to-all: heads {h} not divisible by group size {p}"
    # [..., s, P, h//P, d] -> [P, ..., s, h//P, d]
    send = x.reshape(*lead, s, p, h // p, d).movedim(nlead + 1, 0).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    # recv[src] is src's local sequence on our head-slice; put P back just
    # before s, so the gathered sequence is source-rank-major.
    return recv.movedim(0, nlead).reshape(*lead, p * s, h // p, d).contiguous()


def _a2a_seq_to_heads(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """``[..., s_full, h_p, d] -> [..., s_full // P, h_p * P, d]`` -- inverse of
    :func:`_a2a_heads_to_seq`.

    Scatters the sequence axis (``-3``) across ``group`` and gathers the head
    axis (``-2``): the heads come back in source-rank order, reconstructing the
    pre-Ulysses head layout.
    """
    p = dist.get_world_size(group)
    if p == 1:
        return x
    *lead, s, hp, d = x.shape
    nlead = len(lead)
    assert s % p == 0, f"ulysses all-to-all: seq {s} not divisible by group size {p}"
    # [..., P, s//P, h_p, d] -> [P, ..., s//P, h_p, d]
    send = x.reshape(*lead, p, s // p, hp, d).movedim(nlead, 0).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    # recv[src] is our sequence-block on src's heads; put P just before h_p, so
    # the gathered heads are source-rank-major (the pre-Ulysses layout).
    return recv.movedim(0, nlead + 1).reshape(*lead, s // p, hp * p, d).contiguous()


class _UlyssesAllToAll(torch.autograd.Function):
    """Autograd-aware Ulysses all-to-all over an arbitrary leading batch.

    ``gather_seq=True`` goes sequence-parallel -> head-parallel
    (:func:`_a2a_heads_to_seq`); ``False`` is the inverse. The backward of one
    direction is the forward of the other, so gradients flow straight through
    without saving anything but the direction and the group. A ``[3, b, s, h, d]``
    stack of Q/K/V is one call.
    """

    @staticmethod
    def forward(
        x: torch.Tensor, gather_seq: bool, group: dist.ProcessGroup
    ) -> torch.Tensor:
        return (
            _a2a_heads_to_seq(x, group)
            if gather_seq
            else _a2a_seq_to_heads(x, group)
        )

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:  # type: ignore[no-untyped-def]
        _x, gather_seq, group = inputs
        ctx.gather_seq = gather_seq
        ctx.group = group

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[no-untyped-def]
        grad_in = (
            _a2a_seq_to_heads(grad_out, ctx.group)
            if ctx.gather_seq
            else _a2a_heads_to_seq(grad_out, ctx.group)
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
    # One all-to-all for all three shards: stack -> [3, b, s, h, d] -> a2a.
    qkv = _UlyssesAllToAll.apply(torch.stack((q, k, v)), True, group)
    out = core(*qkv.unbind(0))
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

    ``ring_core`` is expected to be an overlapped ring (jigsaw's P2P provider
    kicks each inter-node K/V send before the matching attention block, so the
    inter-node exchange runs under the node-local attention compute).

    ``q``/``k``/``v`` are ``bshd`` ``[b, seq // spatial, local_heads, d]``;
    ``ring_core`` gets ``[b, seq // n_nodes, local_heads // A, d]`` (``A =
    intra_group.size()``) and must return the same shape. Needs
    ``local_heads % A == 0``.
    """
    return ulysses_attention(q, k, v, intra_group, ring_core)
