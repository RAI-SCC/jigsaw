"""Flash-attention core for the ring's inner attention (CUDA).

Wraps ``torch.ops.aten._scaled_dot_product_flash_attention`` and its backward,
which return the log-sum-exp needed by the ring.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import math as math_core


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float,
    causal: bool = False,
    key_mask: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    *,
    layout: str = "bhsd",
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Compute flash attention.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query, key and value, shaped ``(b, h, s, d)``.
    scale : float
        Softmax scale.
    dropout_p : float
        Dropout probability.
    causal : bool
        Whether to apply the causal triangular mask.
    key_mask : torch.Tensor, optional
        Boolean key-padding mask, ``[B, Sk]``, ``True`` = attend.
    seed : int, optional
        Unused by the flash path; forwarded to :mod:`.math` when ``key_mask``
        is given (unused there too -- dropout isn't exercised on that path --
        kept for interface parity).
    layout : str
        Kept for interface parity with the other cores; sdpa only accepts
        ``"bhsd"``.

    * ``q``/``k``/``v``/``out``/``grad_out`` are ``(B, H, S, d)`` -- heads
      *before* sequence -- and contiguous. The public ``(B, S, H, d)`` layout
      must be transposed first (ring attention already works in ``(B, H, S, d)``).
    * forward returns ``(out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
      philox_seed, philox_offset, debug_attn_mask)``; ``philox_seed`` /
      ``philox_offset`` are the kernel's own dropout RNG state -- opaque and
      generated internally (the op takes no seed input)
    * ``max_q`` / ``max_k`` (backward) are the query / key sequence lengths.
    * ``cum_seq_q`` / ``cum_seq_k`` (backward) are for nested/ragged batches
      only and stay ``None`` here.
    * backward returns ``(dq, dk, dv)`` in the same ``(B, H, S, d)`` layout.

    ``causal`` is only ever set for the ring's diagonal block (q/k chunks
    starting at the same global offset), where the kernel's top-left-aligned
    triangular mask is the correct causal boundary; every other block is
    either fully visible (``causal=False``) or skipped by the caller entirely.

    ``key_mask`` (``[B, Sk]``, bool, ``True`` = attend): delegates to
    :mod:`.math` instead of the flash op. An earlier version of this core used
    ``_scaled_dot_product_efficient_attention``/its backward (an additive
    ``attn_bias``, returning its own ``logsumexp``) for the masked path, but
    that op's backward does not tolerate being fed the ring's *merged*
    (multi-chunk) ``out``/``logsumexp`` -- which is exactly what
    :func:`..backward.ring_backward` does for every chunk -- even though its
    forward is fine in isolation: confirmed on GPU with a 2-chunk, single-
    process replica of the ring's merge-then-recompute pattern (``dv`` came
    back exact, ``dq``/``dk`` did not, matching the ring tests' failure
    signature exactly). :mod:`.math` implements the same merge-tolerant
    recompute from first principles (explicit ``p = exp(attn - lse)``) and is
    correctness-first by design, at the cost of materializing the full
    ``(b, h, sq, sk)`` score matrix instead of using a fused kernel -- an
    acceptable trade for the less-common masked path; the plain flash path
    above is untouched and stays fused.

    Returns
    -------
    out : torch.Tensor
        Attention output.
    logsumexp : torch.Tensor
        Log-sum-exp of the scores.
    rng_state : torch.Tensor | None
        Dropout RNG state of the kernel, ``None`` without dropout.
    """
    assert layout == "bhsd", (
        f"sdpa core is bhsd-only; got layout={layout!r}. The bshd ring is TE-only."
    )
    if key_mask is not None:
        return math_core.forward(
            q, k, v, scale, dropout_p, causal=causal, key_mask=key_mask, seed=seed
        )
    q, k, v = [x.to(torch.float16) for x in (q, k, v)]
    (
        out,
        logsumexp,
        _cum_seq_q,
        _cum_seq_k,
        _max_q,
        _max_k,
        philox_seed,
        philox_offset,
        _debug_attn_mask,
    ) = torch.ops.aten._scaled_dot_product_flash_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=scale,
        return_debug_mask=False,
    )
    # ``philox_seed`` / ``philox_offset`` are the flash kernel's opaque
    # dropout RNG state. Their shapes are build-dependent and need not match
    # (this torch build returns ``(2,)`` and ``()``), so flatten and
    # concatenate into a single 1-D tensor -- the ring stores ``rng_state``
    # via ``ctx.save_for_backward``, which takes tensors, not tuples -- and
    # split it back in :func:`backward`. ``philox_offset`` is always last.
    rng_state = (
        torch.cat([philox_seed.reshape(-1), philox_offset.reshape(-1)])
        if dropout_p > 0.0
        else None
    )
    return out, logsumexp, rng_state


def backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    scale: float,
    max_q: float,
    max_k: float,
    causal: bool = False,
    key_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    seed: Optional[int] = None,
    rng_state: Optional[torch.Tensor] = None,
    layout: str = "bhsd",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the flash-attention gradients.

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
    max_q, max_k : float
        Query and key sequence lengths.
    causal : bool
        Whether the causal triangular mask was applied in the forward.
    key_mask : torch.Tensor, optional
        Boolean key-padding mask, ``[B, Sk]``, ``True`` = attend.
    dropout_p : float
        Dropout probability.
    seed : int, optional
        Unused.
    rng_state : torch.Tensor, optional
        Dropout RNG state returned by :func:`forward`.
    layout : str
        Kept for interface parity with the other cores; sdpa only accepts
        ``"bhsd"``.

    See :func:`forward` for the ``(B, H, S, d)`` layout, the meaning of
    ``max_q`` / ``max_k`` / ``philox_*`` / ``cum_seq_*``, ``causal``, and the
    ``key_mask`` mask-aware fork onto :mod:`.math`.

    ``q``/``k``/``v``/``out``/``grad_out`` are cast to fp16 to match
    :func:`forward`, and ``logsumexp`` to fp32: the flash backward kernel reads
    it as fp32 and returns garbage for a 16-bit ``logsumexp`` (the ring's
    :func:`..backward.ring_backward` hands it back downcast to the model dtype).
    Gradients are returned in the input dtype.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Gradients ``(dq, dk, dv)`` in the input dtype.
    """
    assert layout == "bhsd", (
        f"sdpa core is bhsd-only; got layout={layout!r}. The bshd ring is TE-only."
    )
    if key_mask is not None:
        return math_core.backward(
            grad_out,
            q,
            k,
            v,
            out,
            logsumexp,
            scale,
            causal=causal,
            key_mask=key_mask,
            dropout_p=dropout_p,
            seed=seed,
        )
    in_dtype = grad_out.dtype
    grad_out, q, k, v, out = (t.to(torch.float16) for t in (grad_out, q, k, v, out))
    logsumexp = logsumexp.to(torch.float32)

    if rng_state is not None:
        philox_seed, philox_offset = rng_state[:-1], rng_state[-1]
    else:
        philox_seed = torch.zeros(1, dtype=torch.long, device=q.device)
        philox_offset = torch.zeros(1, dtype=torch.long, device=q.device)

    dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        grad_out,
        q,
        k,
        v,
        out,
        logsumexp,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=scale,
        cum_seq_q=None,
        cum_seq_k=None,
        max_q=max_q,
        max_k=max_k,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
    )
    return dq.to(in_dtype), dk.to(in_dtype), dv.to(in_dtype)
