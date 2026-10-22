"""Ulysses (DeepSpeed-style) sequence-parallel attention.

An all-to-all trades each rank's sequence shard for a head shard, so attention
runs head-parallel over the full sequence; a second all-to-all reverses it.
Needs the local head count divisible by the group size.
:func:`hierarchical_ulysses_attention` nests it in a P2P ring. All
tensors are ``bshd``.
"""

from __future__ import annotations

from typing import Callable, cast

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
    """``[..., s_full, h_p, d] -> [..., s_full // P, h_p * P, d]`` -- inverse of :func:`_a2a_heads_to_seq`.

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
            _a2a_heads_to_seq(x, group) if gather_seq else _a2a_seq_to_heads(x, group)
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
        group for flat Ulysses, or the intra group for the hierarchical
        scheme.
    core : callable
        ``core(q, k, v) -> out`` on ``bshd`` tensors with ``local_heads // P``
        heads: a dense kernel (flat) or a ring across the all-to-all groups
        (hierarchical).
    qkv_stack : torch.Tensor, optional
        The ``[3, b, seq // P, local_heads, head_dim]`` tensor ``q``/``k``/``v``
        were unbound from; skips re-stacking them.

    Returns
    -------
    torch.Tensor
        ``bshd`` ``[b, seq // P, local_heads, head_dim]`` -- same layout and
        sharding as ``q``.
    """
    # One all-to-all for all three shards: [3, b, s, h, d] -> a2a. Reuse the
    # caller's stacked q/k/v when it has one; else stack the three views here.
    stacked = qkv_stack if qkv_stack is not None else torch.stack((q, k, v))
    qkv = _UlyssesAllToAll.apply(stacked, True, group)
    out = core(*qkv.unbind(0))
    return cast(torch.Tensor, _UlyssesAllToAll.apply(out, False, group))


def hierarchical_ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    intra_group: dist.ProcessGroup,
    ring_core: AttnCore,
    qkv_stack: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Ulysses over ``intra_group``, nested in a P2P ring across those groups.

    ``intra_group`` can be node-local, keeping the all-to-all on the fast fabric.

    The all-to-all runs over ``intra_group`` (``A`` ranks); ``ring_core`` then
    rings K/V over the ``spatial // A`` remaining peers, receiving
    ``[b, seq // (spatial // A), local_heads // A, d]`` and returning the same
    shape.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query, key and value, ``bshd`` ``[b, seq // spatial, local_heads, d]``.
    intra_group : dist.ProcessGroup
        Group the all-to-all runs over.
    ring_core : AttnCore
        Attention over the ring across ``intra_group`` instances.
    qkv_stack : torch.Tensor, optional
        ``q``/``k``/``v`` already stacked, forwarded to :func:`ulysses_attention`.

    Returns
    -------
    torch.Tensor
        Attention output, ``bshd``, same shape as ``q``.
    """
    return ulysses_attention(q, k, v, intra_group, ring_core, qkv_stack=qkv_stack)
