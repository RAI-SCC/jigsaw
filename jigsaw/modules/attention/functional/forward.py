"""Ring attention forward implementation.

A single driver loop (:func:`ring_forward`) handles every communication strategy
by delegating to a :class:`~.forward_chunk_provider._ForwardChunkProvider`,
mirroring how :func:`.backward.ring_backward` delegates to a
:class:`~.backward_chunk_provider._BackwardChunkProvider`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional
from typing import Optional

import torch
import torch.distributed as dist

from jigsaw.modules.attention.config import RingAttentionConfig

from .forward_chunk_provider import build_forward_chunk_provider
from .functional import combine_dropout_seed


def _draw_base_seed(device: torch.device) -> int:
    """Draw a fresh per-call dropout base seed from the ambient default RNG.

    Consumed once per ring-attention forward; every per-chunk sub-seed
    (:func:`..functional.combine_dropout_seed`) derives from it deterministically,
    so only this one int needs saving for backward to replay every chunk's mask.
    """
    return int(torch.randint(0, 2**31 - 1, (1,), device=device).item())


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
    causal: bool = False,
    key_mask: Optional[torch.Tensor] = None,
    seed: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Run the attention core on (q, kj, vj) and merge into the running (out, lse).

    kj/vj may be views into a KV buffer; flash attention requires contiguous
    inputs, so they're materialized here. ``causal`` masks the diagonal block
    (see :func:`_causal_step_kind`); it is never set together with a skip.
    ``key_mask`` is this step's ``[B, n_local]`` slice of the global
    key-padding mask (see :func:`ring_forward`).
    inputs, so they're materialized here. Returns the merged ``(out, lse)``
    plus this step's ``rng_state`` (``None`` unless the core is one that
    threads back opaque kernel RNG state instead of using ``seed`` -- see
    :mod:`.cores`).
    """
    out_j, l_j, rng_state_j = ring_config.attn_core_forward_fn(
        q, kj.contiguous(), vj.contiguous(), scale, dropout_p, causal=causal,
        key_mask=key_mask,, seed=seed
    )

    o_new, l_new = _online_softmax_merge(out, lse, out_j, l_j)
    return o_new, l_new, rng_state_j


def _causal_step_kind(kv_global_idx: int, q_global_idx: int) -> Optional[bool]:
    """Classify a ring step's K/V chunk relative to this rank's own Q chunk.

    Both chunks are ``n_local``-sized, contiguous, evenly-split shards of the
    global sequence, so global chunk id alone (not position within the chunk)
    decides visibility:

    * ``kv_global_idx > q_global_idx``: every key is strictly in the future of
      every query in this chunk -- fully masked, so the caller skips the step
      entirely (returns ``None``).
    * ``kv_global_idx == q_global_idx``: the "diagonal" block -- q and k start
      at the same global offset, so the plain top-left triangular mask is the
      causal boundary (returns ``True``).
    * ``kv_global_idx < q_global_idx``: every key is at or before every query
      in this chunk -- fully visible, no mask needed (returns ``False``).
    """
    if kv_global_idx > q_global_idx:
        return None
    return kv_global_idx == q_global_idx


# --------------------------------------------------------------------------- #
# forward driver
# --------------------------------------------------------------------------- #
def build_ring_forward(
    ring_config: RingAttentionConfig,
) -> Callable[
    ...,
    tuple[
        torch.Tensor,
        torch.Tensor,
        tuple,
        Optional[int],
        tuple,
    ],
]:
    """Return the ring-attention forward.

    Kept for backwards compatibility: the concrete communication strategy is now
    chosen per call by :func:`~.forward_chunk_provider.build_forward_chunk_provider`,
    so this always returns the one :func:`ring_forward` driver.
    """
    return ring_forward


def _gather_key_padding_mask(
    key_padding_mask: torch.Tensor, group: dist.ProcessGroup, ring_group_size: int
) -> torch.Tensor:
    """All-gather local ``[B, n_local]`` key-padding shards into ``[B, S_global]``.

    A one-time, small collective (bandwidth negligible next to the K/V ring
    traffic) done once per forward call, so every rank can slice out the
    ``[B, n_local]`` key-validity slab for whatever global chunk it currently
    holds (via ``provider.global_chunk_idx(step)``) without threading a new
    tensor through every :class:`~.forward_chunk_provider._ForwardChunkProvider`
    subclass's communication path.
    """
    shards = [torch.empty_like(key_padding_mask) for _ in range(ring_group_size)]
    dist.all_gather(shards, key_padding_mask.contiguous(), group=group)
    return torch.cat(shards, dim=1)


def ring_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    scale: float,
    dropout_p: float,
    group: dist.ProcessGroup,
    local_rank: int,
    ring_group_size: int,
    ring_config: RingAttentionConfig,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor | None, ...],
    Optional[int],
    tuple[torch.Tensor | None, ...],
]:
    """Forward pass for ring attention.

    One loop over ``ring_group_size`` ring steps: the
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
    local_rank : int
        Rank within ``group``.
    ring_group_size : int
        Number of ranks in ``group``.
    ring_config : RingAttentionConfig
        Ring-attention configuration.
    key_padding_mask : Optional[torch.Tensor]
        This rank's local shard (``[B, n_local]``, bool, ``True`` = attend) of
        a global key-padding mask.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...], int | None, tuple[torch.Tensor | None, ...]]
        Attention output, log-sum-exp values, the chunks saved for backward,
        the dropout base seed (``None`` if ``dropout_p == 0``), and per-global-chunk
        RNG state (see :func:`_attend_step`).
    """
    b, heads, n_local, d = q.shape
    out = torch.zeros_like(q)
    lse = torch.full((b, heads, n_local), float("-inf"), device=q.device, dtype=q.dtype)

    global_key_mask = None
    if key_padding_mask is not None:
        global_key_mask = _gather_key_padding_mask(
            key_padding_mask, group, ring_group_size
        )

    provider = build_forward_chunk_provider(
        ring_config, q, k, v, x, w_qkv, group, local_rank, ring_group_size
    )

    base_seed = _draw_base_seed(q.device) if dropout_p > 0.0 else None
    rng_states: list[Optional[torch.Tensor]] = [None] * ring_group_size

    for step in range(ring_group_size):
        kj, vj = provider.get(step)
        provider.save_for_backward(step)
        chunk_idx = provider.global_chunk_idx(step)
        step_causal = False
        if ring_config.causal:
            kind = _causal_step_kind(chunk_idx, local_rank)
            if kind is None:
                provider.advance(step, ring_group_size)
                continue
            step_causal = kind
        step_key_mask = (
            global_key_mask[:, chunk_idx * n_local : (chunk_idx + 1) * n_local]
            if global_key_mask is not None
            else None
        )
        out, lse = _attend_step(
            ring_config, q, kj, vj, scale, dropout_p, out, lse,
            causal=step_causal, key_mask=step_key_mask,
        )
        global_chunk_idx = provider.global_chunk_idx(step)
        seed = (
            combine_dropout_seed(base_seed, local_rank, global_chunk_idx)
            if base_seed is not None
            else None
        )
        out, lse, rng_state_j = _attend_step(
            ring_config, q, kj, vj, scale, dropout_p, out, lse, seed
        )
        rng_states[global_chunk_idx] = rng_state_j
        provider.advance(step, ring_group_size)

    ring_config.wait_offloads()
    out = out.to(q.dtype)
    return out, lse, provider.result(), base_seed, tuple(rng_states)
