"""Unfused reference attention core for the ring's inner attention.

Runs for any device / shape / dtype and is what the SDPA selectors fall back to
on CPU. The backward takes the ring's globally normalised ``out`` / ``logsumexp``
and recomputes ``dQ/dK/dV`` from them directly.

Dropout is applied post-softmax (inverted dropout: kept elements scaled by
``1 / (1 - dropout_p)``), matching ``torch.nn.functional.scaled_dot_product_attention``'s
convention. ``logsumexp`` is computed from the *pre*-dropout scores -- dropout
doesn't change the normalising constant, and the ring's online-softmax merge
(:func:`..forward._online_softmax_merge_`) needs the undropped value to combine
chunks correctly.

Since this core has no opaque kernel-internal RNG state to thread back (unlike
:mod:`.sdpa` / :mod:`.te`), it reconstructs its dropout mask from ``seed``: an
int the ring independently recomputes identically in the forward and backward
loops (see :func:`..functional._combine_dropout_seed`), rather than a saved
tensor. A dedicated ``torch.Generator`` seeded from it keeps the draw
reproducible without disturbing the ambient default RNG stream.
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
        exclude = torch.triu(
            torch.ones(sq, sk, dtype=torch.bool, device=device), diagonal=1
        )
    if key_mask is not None:
        pad_exclude = (~key_mask)[:, None, None, :]  # (B, 1, 1, sk)
        exclude = pad_exclude if exclude is None else (exclude | pad_exclude)
    return exclude


def _dropout_mask(shape: torch.Size, device: torch.device, seed: int) -> torch.Tensor:
    """Return a boolean dropout keep-mask that is reproducible from ``seed``."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(shape, generator=generator, device=device, dtype=torch.float32)


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = 0.0,
    dropout_p: float = 0.0,
    causal: bool = False,
    key_mask: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    *,
    layout: str = "bhsd",
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Compute attention.

    ``key_mask`` (``[B, Sk]``, bool, ``True`` = attend) masks out padding key
    positions; it composes with ``causal`` (a position is excluded if either
    applies). A query row with every key excluded gets ``logsumexp = -inf``
    (correct: ``logsumexp`` of an all ``-inf`` row is ``-inf``, not ``nan``)
    and an explicit zero output row -- softmax of an all ``-inf`` row is
    ``nan``, and this step's contribution must still be a well-defined zero
    for the ring's online-softmax merge (:func:`..forward._online_softmax_merge`)
    to not get poisoned by a ``0 * nan``.

    ``rng_state`` is always ``None`` -- this core replays dropout from ``seed``
    instead (see module docstring).

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query, key and value, shaped ``(b, h, s, d)``.
    scale : float, optional
        Softmax scale. Defaults to ``d ** -0.5``.
    dropout_p : float
        Dropout probability.
    causal : bool
        Whether to apply the causal triangular mask.
    key_mask : torch.Tensor, optional
        Boolean key-padding mask, ``[B, Sk]``, ``True`` = attend.
    seed : int, optional
        Seed of the dropout mask.
    layout : str
        Kept for interface parity with the other cores; math only accepts
        ``"bhsd"``.

    Returns
    -------
    out : torch.Tensor
        Attention output.
    logsumexp : torch.Tensor
        Log-sum-exp of the scores.
    rng_state : None
        Always ``None``.
    """
    assert layout == "bhsd", (
        f"math core is bhsd-only; got layout={layout!r}. The bshd ring is TE-only."
    )
    scale = scale or (q.size(-1) ** -0.5)
    scores = (q @ k.mT) * scale
    exclude = _build_exclude_mask(
        scores.shape[-2], scores.shape[-1], causal, key_mask, scores.device
    )
    dead_row = None
    if exclude is not None:
        scores = scores.masked_fill(exclude, float("-inf"))
        dead_row = exclude.all(dim=-1, keepdim=True)  # broadcasts against (B, H, Sq, 1)
    logsumexp = scores.logsumexp(dim=-1)  # (B, H, Sq)
    p = torch.softmax(scores, dim=-1)
    if dead_row is not None:
        p = torch.where(dead_row, torch.zeros_like(p), p)
    if dropout_p > 0.0:
        assert seed is not None, "dropout_p > 0 requires a seed"
        keep = _dropout_mask(p.shape, p.device, seed) >= dropout_p
        p = p * keep / (1.0 - dropout_p)
    out = p @ v
    return out, logsumexp, None


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
    dropout_p: float = 0.0,
    seed: Optional[int] = None,
    rng_state: Optional[torch.Tensor] = None,  # noqa: ARG001 - unused, interface parity
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the attention gradients from the saved output and log-sum-exp.

    ``logsumexp`` is the *global* (whole-ring) value; a query row that never
    attended anywhere (globally dead: every key, across every ring step, was
    excluded) has ``logsumexp = -inf``. Subtracting that from this step's
    (also all ``-inf``, by the same global exclusion) ``attn`` row would give
    ``-inf - -inf = nan``. Since that row's saved global ``out`` is exactly
    zero (see :func:`forward`'s ``dead_row`` handling), substituting a finite
    dummy (``0``) for a dead row's ``lse`` makes ``exp(attn - 0) = exp(-inf) =
    0`` -- the same zero row the true (but nan-producing) subtraction would
    give -- without the ``nan``, and without perturbing any live row.

    Parameters
    ----------
    grad_out : torch.Tensor
        Gradient of the output.
    q, k, v : torch.Tensor
        Query, key and value, shaped ``(b, h, s, d)``.
    out : torch.Tensor
        Saved attention output.
    logsumexp : torch.Tensor
        Saved log-sum-exp.
    scale : float
        Softmax scale.
    causal : bool
        Whether the causal triangular mask was applied in the forward.
    key_mask : torch.Tensor, optional
        Boolean key-padding mask, ``[B, Sk]``, ``True`` = attend.
    dropout_p : float
        Dropout probability.
    seed : int, optional
        Seed of the dropout mask.
    rng_state : torch.Tensor, optional
        Unused.
    *args : Any
        Unused.
    **kwargs : Any
        Unused.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Gradients ``(dq, dk, dv)``.
    """
    attn = (q @ k.mT) * scale  # (B, H, Sq, Sk)
    exclude = _build_exclude_mask(
        attn.shape[-2], attn.shape[-1], causal, key_mask, attn.device
    )
    if exclude is not None:
        attn = attn.masked_fill(exclude, float("-inf"))
    # The ring's accumulator keeps `logsumexp` in at least fp32 (see
    # `_online_softmax_merge_`) regardless of q/k/v's working dtype; cast it
    # down first so the subtraction doesn't promote `p` (and everything built
    # from it below) to fp32, which the final matmuls against bf16/fp16
    # grad_out/k/q would then reject as a dtype mismatch.
    lse = logsumexp.unsqueeze(-1).to(attn.dtype)
    dead = torch.isneginf(lse)
    safe_lse = torch.where(dead, torch.zeros_like(lse), lse)
    p = torch.exp(attn - safe_lse)  # global norm via saved lse

    keep = None
    if dropout_p > 0.0:
        assert seed is not None, "dropout_p > 0 requires a seed"
        keep = (_dropout_mask(p.shape, p.device, seed) >= dropout_p).to(p.dtype)

    p_dropped = p if keep is None else p * keep / (1.0 - dropout_p)
    dv = p_dropped.mT @ grad_out

    dp = grad_out @ v.mT  # dPd = dO @ V^T
    if keep is not None:
        dp = dp * keep / (1.0 - dropout_p)  # dP = dPd * mask / (1 - p)
    d = (grad_out * out).sum(dim=-1, keepdim=True)  # GLOBAL row-sum, from saved out
    d_attn = p * (dp - d) * scale

    dq = d_attn @ k
    dk = d_attn.mT @ q
    return dq, dk, dv
