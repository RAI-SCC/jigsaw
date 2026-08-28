"""Ring attention forward implementation.

A single driver loop (:func:`ring_forward`) handles every communication strategy
by delegating to a :class:`~.forward_chunk_provider._ForwardChunkProvider`,
mirroring how :func:`.backward.ring_backward` delegates to a
:class:`~.chunk_provider._BackwardChunkProvider`.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

from jigsaw.modules.attention.config import RingAttentionConfig

from .forward_chunk_provider import build_forward_chunk_provider


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

    kj/vj may be views into a KV buffer; flash attention requires contiguous
    inputs, so they're materialized here.
    """
    out_j, l_j = ring_config.attn_core_forward_fn(
        q, kj.contiguous(), vj.contiguous(), scale, dropout_p
    )

    return _online_softmax_merge(out, lse, out_j, l_j)


# --------------------------------------------------------------------------- #
# forward driver
# --------------------------------------------------------------------------- #
def build_ring_forward(
    ring_config: RingAttentionConfig,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor, tuple]]:
    """Return the ring-attention forward.

    Kept for backwards compatibility: the concrete communication strategy is now
    chosen per call by :func:`~.forward_chunk_provider.build_forward_chunk_provider`,
    so this always returns the one :func:`ring_forward` driver.
    """
    return ring_forward


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
    """Forward pass for ring attention.

    One loop over ``world_size`` ring steps: the
    :class:`~.forward_chunk_provider._ForwardChunkProvider` for
    ``ring_config.send_op`` makes each chunk current (``get``), records it for
    the backward pass (``save_for_backward``), and progresses communication
    (``advance``); this function only runs the attention core and the online
    softmax merge.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Local query, key, and value tensors, shaped ``(b, h, n_local, d)``.
    x : torch.Tensor
        Local input sequence chunk.
    w_qkv : torch.Tensor
        Fused query, key, and value projection weights.
    scale : float
        Attention scaling factor.
    dropout_p : float
        Attention dropout probability.
    group : dist.ProcessGroup
        Distributed process group used for ring communication.
    empty : torch.Tensor
        Empty tensor used by the communication helpers.
    local_rank : int
        Rank within ``group``.
    world_size : int
        Number of ranks in ``group``.
    ring_config : RingAttentionConfig
        Ring-attention configuration.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]
        Attention output, log-sum-exp values, and the chunks saved for backward.
    """
    b, heads, n_local, d = q.shape
    out = torch.zeros_like(q)
    lse = torch.full((b, heads, n_local), float("-inf"), device=q.device, dtype=q.dtype)

    provider = build_forward_chunk_provider(
        ring_config, q, k, v, x, w_qkv, group, empty, local_rank, world_size
    )

    for step in range(world_size):
        kj, vj = provider.get(step)
        provider.save_for_backward(step)
        out, lse = _attend_step(ring_config, q, kj, vj, scale, dropout_p, out, lse)
        provider.advance(step, world_size)

    ring_config.wait_offloads()
    out = out.to(q.dtype)
    return out, lse, provider.result()
