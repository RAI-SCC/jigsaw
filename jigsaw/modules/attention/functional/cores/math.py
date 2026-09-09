"""Unfused ``softmax(QK^T)V`` reference core for the ring inner step.

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
loops (see :func:`..functional.combine_dropout_seed`), rather than a saved
tensor. A dedicated ``torch.Generator`` seeded from it keeps the draw
reproducible without disturbing the ambient default RNG stream.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def _dropout_mask(shape: torch.Size, device: torch.device, seed: int) -> torch.Tensor:
    """Boolean keep-mask for ``shape``, reproducible given ``seed``.

    Drawn in float32 via a dedicated ``torch.Generator`` (not the ambient
    default RNG), so forward and backward can each build the identical mask
    from nothing but ``seed`` -- no mask/state tensor needs saving.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(shape, generator=generator, device=device, dtype=torch.float32)


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = 0.0,
    dropout_p: float = 0.0,
    seed: Optional[int] = None,
    *,
    layout: str = "bhsd",
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Vanilla attention forward pass. Returns ``(out, logsumexp, rng_state)``.

    ``rng_state`` is always ``None`` -- this core replays dropout from ``seed``
    instead (see module docstring).
    """
    assert layout == "bhsd", (
        f"math core is bhsd-only; got layout={layout!r}. The bshd ring is TE-only."
    )
    scale = scale or (q.size(-1) ** -0.5)
    scores = (q @ k.mT) * scale
    logsumexp = scores.logsumexp(dim=-1)  # (B, H, Sq)
    p = torch.softmax(scores, dim=-1)
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
    dropout_p: float = 0.0,
    seed: Optional[int] = None,
    rng_state: Optional[torch.Tensor] = None,  # noqa: ARG001 - unused, interface parity
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vanilla attention backward pass using saved (global) LSE. Returns (dq, dk, dv).

    With dropout, ``p`` (the un-dropped, globally-normalised softmax share
    recomputed below) needs converting to the *dropped* probabilities actually
    multiplied by ``v`` in forward before it can be used for ``dv``, and the
    upstream gradient needs converting the other way (``dPd -> dP``) before the
    softmax-Jacobian step; both conversions are the same ``mask / (1 - p)``
    dropout replayed from ``seed``. The row-sum correction term ``d`` is
    unaffected by dropout (it uses the actual, already-dropped, saved ``out``).
    """
    attn = (q @ k.mT) * scale  # (B, H, Sq, Sk)
    p = torch.exp(attn - logsumexp.unsqueeze(-1))  # global norm via saved lse

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
