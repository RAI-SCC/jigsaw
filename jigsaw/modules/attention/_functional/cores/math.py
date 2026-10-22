"""Unfused reference attention core for the ring's inner attention.

Runs on any device, shape and dtype, and is the SDPA selectors' fallback on
CPU. The backward recomputes ``dQ/dK/dV`` from the ring's globally normalised
``out`` and ``logsumexp``.

Dropout is inverted dropout applied after the softmax, as in
``torch.nn.functional.scaled_dot_product_attention``. ``logsumexp`` comes from
the pre-dropout scores, which the ring's online-softmax merge
(:func:`..forward._online_softmax_merge_`) needs to combine chunks.

The dropout mask is rebuilt from ``seed``, which the ring recomputes
identically in forward and backward (see
:func:`..ring._combine_dropout_seed`). A dedicated ``torch.Generator``
keeps the draw from disturbing the default RNG stream.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def _dropout_mask(shape: torch.Size, device: torch.device, seed: int) -> torch.Tensor:
    """Return uniform draws for the dropout mask, reproducible from ``seed``."""
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
    """Compute attention.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query, key and value, shaped ``(b, h, s, d)``.
    scale : float, optional
        Softmax scale. Defaults to ``d ** -0.5``.
    dropout_p : float
        Dropout probability.
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
    """Compute the attention gradients from the saved output and log-sum-exp.

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
    *args : Any
        Unused.
    dropout_p : float
        Dropout probability.
    seed : int, optional
        Seed of the dropout mask.
    rng_state : torch.Tensor, optional
        Unused.
    **kwargs : Any
        Unused.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Gradients ``(dq, dk, dv)``.
    """
    attn = (q @ k.mT) * scale  # (B, H, Sq, Sk)
    # The ring keeps `logsumexp` in at least fp32 (see `_online_softmax_merge_`).
    # Cast it to the scores' dtype so `p` stays bf16/fp16; otherwise the
    # matmuls against grad_out/k/q below reject the mixed dtypes.
    p = torch.exp(
        attn - logsumexp.unsqueeze(-1).to(attn.dtype)
    )  # global norm via saved lse

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
