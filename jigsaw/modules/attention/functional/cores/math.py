"""Unfused ``softmax(QK^T)V`` reference core for the ring inner step.

Runs for any device / shape / dtype and is what the SDPA selectors fall back to
on CPU. The backward takes the ring's globally normalised ``out`` / ``logsumexp``
and recomputes ``dQ/dK/dV`` from them directly.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def _apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """Mask out ``j > i`` (top-left aligned) for a same-position diagonal block.

    Only used by the ring for the one step per rank where the K/V chunk in hand
    starts at the same global offset as the local Q chunk, so ``i``/``j`` index
    the same absolute positions and the plain (unshifted) triangular mask is
    the correct causal boundary.
    """
    sq, sk = scores.shape[-2:]
    mask = torch.triu(
        torch.ones(sq, sk, dtype=torch.bool, device=scores.device), diagonal=1
    )
    return scores.masked_fill(mask, float("-inf"))


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = 0.0,
    dropout_p: float = 0.0,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vanilla attention forward pass. Returns (out, logsumexp)."""
    scale = scale or (q.size(-1) ** -0.5)
    scores = (q @ k.mT) * scale
    if causal:
        scores = _apply_causal_mask(scores)
    logsumexp = scores.logsumexp(dim=-1)  # (B, H, Sq)
    out = torch.softmax(scores, dim=-1) @ v
    return out, logsumexp


def backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    scale: float,
    *args: Any,
    causal: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vanilla attention backward pass using saved (global) LSE. Returns (dq, dk, dv)."""
    attn = (q @ k.mT) * scale  # (B, H, Sq, Sk)
    if causal:
        attn = _apply_causal_mask(attn)
    p = torch.exp(attn - logsumexp.unsqueeze(-1))  # global norm via saved lse

    dv = p.mT @ grad_out

    dp = grad_out @ v.mT
    d = (grad_out * out).sum(dim=-1, keepdim=True)  # GLOBAL row-sum, from saved out
    d_attn = p * (dp - d) * scale

    dq = d_attn @ k
    dk = d_attn.mT @ q
    return dq, dk, dv
