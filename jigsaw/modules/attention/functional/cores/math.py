"""Unfused ``softmax(QK^T)V`` reference core for the ring inner step.

Runs for any device / shape / dtype and is what the SDPA selectors fall back to
on CPU. The backward takes the ring's globally normalised ``out`` / ``logsumexp``
and recomputes ``dQ/dK/dV`` from them directly.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def _build_exclude_mask(
    sq: int,
    sk: int,
    causal: bool,
    key_mask: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Boolean exclude-mask (``True`` = blank out), broadcastable to ``(..., sq, sk)``.

    OR-combination of the causal triangle (see the old ``_apply_causal_mask``,
    only used by the ring for the one step per rank where the K/V chunk in
    hand starts at the same global offset as the local Q chunk, so the plain
    unshifted triangular mask is the correct causal boundary) and the negated
    ``key_mask`` (``[B, sk]``, ``True`` = attend), broadcast over heads and
    query positions. Returns ``None`` when neither applies.
    """
    exclude = None
    if causal:
        exclude = torch.triu(torch.ones(sq, sk, dtype=torch.bool, device=device), diagonal=1)
    if key_mask is not None:
        pad_exclude = (~key_mask)[:, None, None, :]  # (B, 1, 1, sk)
        exclude = pad_exclude if exclude is None else (exclude | pad_exclude)
    return exclude


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = 0.0,
    dropout_p: float = 0.0,
    causal: bool = False,
    key_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vanilla attention forward pass. Returns (out, logsumexp).

    ``key_mask`` (``[B, Sk]``, bool, ``True`` = attend) masks out padding key
    positions; it composes with ``causal`` (a position is excluded if either
    applies). A query row with every key excluded gets ``logsumexp = -inf``
    (correct: ``logsumexp`` of an all ``-inf`` row is ``-inf``, not ``nan``)
    and an explicit zero output row -- softmax of an all ``-inf`` row is
    ``nan``, and this step's contribution must still be a well-defined zero
    for the ring's online-softmax merge (:func:`..forward._online_softmax_merge`)
    to not get poisoned by a ``0 * nan``.
    """
    scale = scale or (q.size(-1) ** -0.5)
    scores = (q @ k.mT) * scale
    exclude = _build_exclude_mask(scores.shape[-2], scores.shape[-1], causal, key_mask, scores.device)
    dead_row = None
    if exclude is not None:
        scores = scores.masked_fill(exclude, float("-inf"))
        dead_row = exclude.all(dim=-1, keepdim=True)  # broadcasts against (B, H, Sq, 1)
    logsumexp = scores.logsumexp(dim=-1)  # (B, H, Sq)
    probs = torch.softmax(scores, dim=-1)
    if dead_row is not None:
        probs = torch.where(dead_row, torch.zeros_like(probs), probs)
    out = probs @ v
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
    key_mask: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vanilla attention backward pass using saved (global) LSE. Returns (dq, dk, dv).

    ``logsumexp`` is the *global* (whole-ring) value; a query row that never
    attended anywhere (globally dead: every key, across every ring step, was
    excluded) has ``logsumexp = -inf``. Subtracting that from this step's
    (also all ``-inf``, by the same global exclusion) ``attn`` row would give
    ``-inf - -inf = nan``. Since that row's saved global ``out`` is exactly
    zero (see :func:`forward`'s ``dead_row`` handling), substituting a finite
    dummy (``0``) for a dead row's ``lse`` makes ``exp(attn - 0) = exp(-inf) =
    0`` -- the same zero row the true (but nan-producing) subtraction would
    give -- without the ``nan``, and without perturbing any live row.
    """
    attn = (q @ k.mT) * scale  # (B, H, Sq, Sk)
    exclude = _build_exclude_mask(attn.shape[-2], attn.shape[-1], causal, key_mask, attn.device)
    if exclude is not None:
        attn = attn.masked_fill(exclude, float("-inf"))
    lse = logsumexp.unsqueeze(-1)
    dead = torch.isneginf(lse)
    safe_lse = torch.where(dead, torch.zeros_like(lse), lse)
    p = torch.exp(attn - safe_lse)  # global norm via saved lse

    dv = p.mT @ grad_out

    dp = grad_out @ v.mT
    d = (grad_out * out).sum(dim=-1, keepdim=True)  # GLOBAL row-sum, from saved out
    d_attn = p * (dp - d) * scale

    dq = d_attn @ k
    dk = d_attn.mT @ q
    return dq, dk, dv
