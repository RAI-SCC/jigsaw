"""Ring attention forward pass."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from jigsaw.modules.attention.config import RingAttentionConfig

from .forward_chunk_provider import build_forward_chunk_provider
from .functional import combine_dropout_seed


def _draw_base_seed(device: torch.device) -> int:
    """Draw a dropout base seed from the default RNG."""
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
    o1, o2 : torch.Tensor
        Outputs of the two partial results.
    l1, l2 : torch.Tensor
        Log-sum-exp values of the two partial results.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Merged output and log-sum-exp.
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
    """Attend to one chunk and merge it into the running output.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        Merged output and log-sum-exp, and the ``rng_state`` of the core.
    """
    out_j, l_j, rng_state_j = ring_config.attn_core_forward_fn(
        q, kj.contiguous(), vj.contiguous(), scale, dropout_p, seed=seed
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
    """Run the forward pass of ring attention.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Local query, key, and value tensors, shaped ``(b, h, n_local, d)``.
    x : torch.Tensor
        Local input sequence chunk.
    w_qkv : torch.Tensor
        Fused query, key, and value projection weights.
    scale : float
        Softmax scale.
    dropout_p : float
        Dropout probability.
    group : dist.ProcessGroup
        Process group of the ring.
    local_rank : int
        Rank within ``group``.
    ring_group_size : int
        Number of ranks in ``group``.
    ring_config : RingAttentionConfig
        Ring-attention configuration.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...], int | None, tuple[torch.Tensor | None, ...]]
        Output, log-sum-exp, chunks saved for backward, dropout base seed
        (``None`` without dropout) and per-chunk RNG states.
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
