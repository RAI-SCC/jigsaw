"""Ring attention forward implementation.

A single driver loop (:func:`ring_forward`) handles every communication strategy
by delegating to a :class:`~.forward_chunk_provider._ForwardChunkProvider`,
mirroring how :func:`.backward.ring_backward` delegates to a
:class:`~.backward_chunk_provider._BackwardChunkProvider`.
"""

from __future__ import annotations

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
    seed: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Run the attention core on (q, kj, vj) and merge into the running (out, lse).

    kj/vj may be views into a KV buffer; flash attention requires contiguous
    inputs, so they're materialized here. Returns the merged ``(out, lse)``
    plus this step's ``rng_state`` (``None`` unless the core is one that
    threads back opaque kernel RNG state instead of using ``seed`` -- see
    :mod:`.cores`).

    ``kj``/``vj`` are cast to ``q``'s dtype first. ``DistributedAttention``
    downcasts ``q``/``k``/``v`` to the core dtype (bf16 on CUDA) but hands the
    ring its ``x`` / projection weight in the master dtype, so a chunk
    recomputed from ``x`` via ``w_qkv`` (``_compute_kv``) comes back in the
    master dtype -- a mismatch the fused SDPA/TE cores paper over (they force
    their own kernel dtype) but the unfused ``math`` core raises on. This is the
    forward twin of the ``core_dtype`` reconcile in :func:`..backward.ring_backward`.
    """
    out_j, l_j, rng_state_j = ring_config.attn_core_forward_fn(
        q,
        kj.to(q.dtype).contiguous(),
        vj.to(q.dtype).contiguous(),
        scale,
        dropout_p,
        seed=seed,
    )

    o_new, l_new = _online_softmax_merge(out, lse, out_j, l_j)
    return o_new, l_new, rng_state_j


# --------------------------------------------------------------------------- #
# forward driver
# --------------------------------------------------------------------------- #
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

    provider = build_forward_chunk_provider(
        ring_config, q, k, v, x, w_qkv, group, local_rank, ring_group_size
    )

    base_seed = _draw_base_seed(q.device) if dropout_p > 0.0 else None
    rng_states: list[Optional[torch.Tensor]] = [None] * ring_group_size

    for step in range(ring_group_size):
        kj, vj = provider.get(step)
        provider.save_for_backward(step)
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
