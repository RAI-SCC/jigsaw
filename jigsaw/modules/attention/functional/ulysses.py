"""Ulysses (DeepSpeed-style) sequence-parallel attention.

An alternative to the ring (:mod:`.forward`) for parallelising over
``spatial_group``: instead of circulating K/V through P2P steps with an
online-softmax merge after each, one all-to-all trades the local *sequence*
shard for a *head* shard, runs a single local attention over the full
sequence on ``local_heads / spatial_group.size()`` heads, then all-to-alls
back.

Trade-off vs. the ring: O(1) collectives and one fused kernel instead of many
small ones plus LSE merges (TE's ``cp_comm_type="a2a"``), but it needs
``local_heads % P == 0`` (the ring works for any head count), and exposes the
all-to-all directly when the group crosses a slow multi-node fabric --
:func:`hierarchical_ulysses_attention` (Ulysses intra-node, ring inter-node,
the ``cp_comm_type="a2a+p2p"`` analogue) keeps the a2a on the fast fabric.

Q/K/V ride one all-to-all: the scatter/gather helpers take arbitrary leading
dims, so a ``[3, b, s, h, d]`` stack goes through a single
``all_to_all_single`` instead of three.

Composes with channel/tensor parallelism unchanged -- the all-to-all sits
between the (already channel-sharded) ``qkv`` projection and ``head_concat``,
orthogonal to how ``qkv`` was sharded.

All tensors here are ``bshd`` (the TE cores' native layout); the SDPA path
transposes around its kernel call in
:meth:`..attention.DistributedAttention._ulysses_attn_core`.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.distributed as dist

from jigsaw.modules.buffers import get_preallocated_buffer


def _a2a_heads_to_seq(
    x: torch.Tensor, group: dist.ProcessGroup, local_rank: int, device: torch.device
) -> torch.Tensor:
    """``[..., s_local, h, d] -> [..., s_local * P, h // P, d]`` via one all-to-all.

    Scatters the head axis (``-2``) across ``group`` and gathers the sequence
    axis (``-3``): sends head-block ``p`` of the local sequence to rank ``p`` and
    stacks the sequence shards received from every rank in source-rank order --
    exactly how the sequence was sharded across ``group``. Leading dims (batch,
    or a ``3`` for a stacked Q/K/V) are carried through untouched.

    ``local_rank``/``device`` key the recv buffer in :func:`.buffers.get_preallocated_buffer`
    -- reused across calls from this same call site instead of a fresh
    ``torch.empty_like`` every forward/backward, which otherwise adds up to a
    full extra ``send``-sized allocation per call at large ``seq_len``.
    """
    p = dist.get_world_size(group)
    if p == 1:
        return x
    *lead, s, h, d = x.shape
    nlead = len(lead)
    assert h % p == 0, f"ulysses all-to-all: heads {h} not divisible by group size {p}"
    # [..., s, P, h//P, d] -> [P, ..., s, h//P, d]
    send = x.reshape(*lead, s, p, h // p, d).movedim(nlead + 1, 0).contiguous()
    recv = get_preallocated_buffer(send.shape, send.dtype, local_rank, device)
    dist.all_to_all_single(recv, send, group=group)
    # recv[src] is src's local sequence on our head-slice; put P back just
    # before s, so the gathered sequence is source-rank-major.
    return recv.movedim(0, nlead).reshape(*lead, p * s, h // p, d).contiguous()


def _a2a_seq_to_heads(
    x: torch.Tensor, group: dist.ProcessGroup, local_rank: int, device: torch.device
) -> torch.Tensor:
    """``[..., s_full, h_p, d] -> [..., s_full // P, h_p * P, d]`` -- inverse of
    :func:`_a2a_heads_to_seq`.

    Scatters the sequence axis (``-3``) across ``group`` and gathers the head
    axis (``-2``): the heads come back in source-rank order, reconstructing the
    pre-Ulysses head layout.

    ``local_rank``/``device``: see :func:`_a2a_heads_to_seq`.
    """
    p = dist.get_world_size(group)
    if p == 1:
        return x
    *lead, s, hp, d = x.shape
    nlead = len(lead)
    assert s % p == 0, f"ulysses all-to-all: seq {s} not divisible by group size {p}"
    # [..., P, s//P, h_p, d] -> [P, ..., s//P, h_p, d]
    send = x.reshape(*lead, p, s // p, hp, d).movedim(nlead, 0).contiguous()
    recv = get_preallocated_buffer(send.shape, send.dtype, local_rank, device)
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
        x: torch.Tensor,
        gather_seq: bool,
        group: dist.ProcessGroup,
        local_rank: int,
        device: torch.device,
    ) -> torch.Tensor:
        return (
            _a2a_heads_to_seq(x, group, local_rank, device)
            if gather_seq
            else _a2a_seq_to_heads(x, group, local_rank, device)
        )

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:  # type: ignore[no-untyped-def]
        _x, gather_seq, group, local_rank, device = inputs
        ctx.gather_seq = gather_seq
        ctx.group = group
        ctx.local_rank = local_rank
        ctx.device = device

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[no-untyped-def]
        grad_in = (
            _a2a_seq_to_heads(grad_out, ctx.group, ctx.local_rank, ctx.device)
            if ctx.gather_seq
            else _a2a_heads_to_seq(grad_out, ctx.group, ctx.local_rank, ctx.device)
        )
        return grad_in, None, None, None, None


AttnCore = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: dist.ProcessGroup,
    core: AttnCore,
    local_rank: int,
    device: torch.device,
    qkv_stack: torch.Tensor | None = None,
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
    local_rank, device : int, torch.device
        Forwarded to :func:`.buffers.get_preallocated_buffer` so the a2a recv
        buffer is cached per call site instead of freshly allocated every
        forward/backward -- see :func:`_a2a_heads_to_seq`.
    qkv_stack : torch.Tensor, optional
        The contiguous ``[3, b, seq // P, local_heads, head_dim]`` tensor
        ``q``/``k``/``v`` were unbound from (:meth:`DistributedAttention.forward`
        keeps it when ``qk_norm`` is off). When given it is fed to the
        all-to-all as-is, skipping a ``torch.stack((q, k, v))`` -- a full
        ``[3, ...]`` copy each forward, plus its transpose in the backward.

    Returns
    -------
    torch.Tensor
        ``bshd`` ``[b, seq // P, local_heads, head_dim]`` -- same layout and
        sharding as ``q``.
    """
    # One all-to-all for all three shards: [3, b, s, h, d] -> a2a. Reuse the
    # caller's stacked q/k/v when it has one; else stack the three views here.
    stacked = qkv_stack if qkv_stack is not None else torch.stack((q, k, v))
    qkv = _UlyssesAllToAll.apply(stacked, True, group, local_rank, device)
    out = core(*qkv.unbind(0))
    return _UlyssesAllToAll.apply(out, False, group, local_rank, device)


def hierarchical_ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    intra_group: dist.ProcessGroup,
    ring_core: AttnCore,
    local_rank: int,
    device: torch.device,
    qkv_stack: torch.Tensor | None = None,
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
    ``local_heads % A == 0``. ``local_rank``/``device``/``qkv_stack`` are
    forwarded to :func:`ulysses_attention` (the latter skips the
    ``torch.stack``).
    """
    return ulysses_attention(
        q, k, v, intra_group, ring_core, local_rank, device, qkv_stack=qkv_stack
    )
