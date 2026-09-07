"""SDPA flash-attention core for the ring inner step (CUDA).

Wraps the two undocumented
``torch.ops.aten._scaled_dot_product_flash_attention{,_backward}`` ops, which --
unlike ``F.scaled_dot_product_attention`` -- return the softmax log-sum-exp the
ring's online-softmax merge and explicit-LSE backward need.

On CPU the selectors fall back to :mod:`.math`; see
:func:`..config._select_attn_core_forward_fn`.

NOTE: the dropout path below is untested -- it needs CUDA, which this repo's
test environment doesn't have. It follows the same replay strategy already
used for the (tested) no-dropout path, just threading through the two extra
tensors the kernel needs to reproduce its mask; treat it as implemented but
unverified until it's run on a GPU.
"""

from __future__ import annotations

from typing import Optional

from typing import Optional

import torch


def _build_bias(
    sq: int,
    sk: int,
    causal: bool,
    key_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Additive attention bias for ``_scaled_dot_product_efficient_attention``.

    Excluded positions get ``torch.finfo(dtype).min`` rather than literal
    ``-inf``: a fully-excluded row (e.g. a padding query under causal+padding,
    or a batch entry with no valid keys) would otherwise make the kernel's
    internal softmax subtract ``-inf - -inf = nan`` (the same failure mode
    :func:`..cores.math._build_exclude_mask`'s caller works around explicitly);
    a large-but-finite bias is the standard fix framework attention kernels use
    for additive masks and needs no cooperation from the (opaque, CUDA-only)
    kernel. Returns ``None`` when neither ``causal`` nor ``key_mask`` applies.
    """
    if not causal and key_mask is None:
        return None
    neg = torch.finfo(dtype).min
    bias = torch.zeros(
        key_mask.shape[0] if key_mask is not None else 1, 1, sq, sk, dtype=dtype, device=device
    )
    if causal:
        causal_exclude = torch.triu(
            torch.ones(sq, sk, dtype=torch.bool, device=device), diagonal=1
        )
        bias = bias.masked_fill(causal_exclude, neg)
    if key_mask is not None:
        pad_exclude = (~key_mask)[:, None, None, :]
        bias = bias.masked_fill(pad_exclude, neg)
    return bias


def forward(
    q: torch.Tensor,
   
    k: torch.Tensor,
   
    v: torch.Tensor,
   
    scale: float,
   
    dropout_p: float,
    causal: bool = False,
    key_mask: Optional[torch.Tensor] = None,,
    seed: Optional[int] = None,  # noqa: ARG001 - unused, interface parity (see .math)
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Flash attention forward pass. Returns ``(out, logsumexp, rng_state)``.

    Calling convention for the two undocumented
    ``torch.ops.aten._scaled_dot_product_flash_attention{,_backward}`` ops,
    worked out against a naive ``softmax(q @ k.mT * scale) @ v`` autograd
    reference and ``F.scaled_dot_product_attention``:

    * ``q``/``k``/``v``/``out``/``grad_out`` are ``(B, H, S, d)`` -- heads
      *before* sequence -- and contiguous. The public ``(B, S, H, d)`` layout
      must be transposed first (ring attention already works in ``(B, H, S, d)``).
    * forward returns ``(out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
      philox_seed, philox_offset, debug_attn_mask)``; ``philox_seed`` /
      ``philox_offset`` are the kernel's own dropout RNG state -- opaque and
      generated internally (the op takes no seed input), so unlike :mod:`.math`
      this core can't rederive its mask from an int and instead threads the
      real state back as ``rng_state`` (``None`` when ``dropout_p == 0``, since
      the mask is never drawn).
    * ``max_q`` / ``max_k`` (backward) are the query / key sequence lengths.
    * ``cum_seq_q`` / ``cum_seq_k`` (backward) are for nested/ragged batches
      only and stay ``None`` here.
    * backward returns ``(dq, dk, dv)`` in the same ``(B, H, S, d)`` layout.

    ``causal`` is only ever set for the ring's diagonal block (q/k chunks
    starting at the same global offset), where the kernel's top-left-aligned
    triangular mask is the correct causal boundary; every other block is
    either fully visible (``causal=False``) or skipped by the caller entirely.

    ``key_mask`` (``[B, Sk]``, bool, ``True`` = attend): the flash op above
    only takes ``is_causal``, no arbitrary mask, so when a ``key_mask`` is
    given this switches to ``_scaled_dot_product_efficient_attention``
    instead, which accepts an additive ``attn_bias`` and also returns
    ``logsumexp``; ``causal`` is folded into that bias too (see
    :func:`_build_bias`) rather than passed as the op's own ``is_causal``, so
    the two masks compose. CUDA-only, like the flash op; unverified against a
    GPU in this repo (no CUDA device available in the environment this was
    written in) -- verify before relying on it in production.
    """
    q, k, v = [x.to(torch.float16) for x in (q, k, v)]
    if key_mask is None:
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
        return out, logsumexp

    bias = _build_bias(q.shape[-2], k.shape[-2], causal, key_mask, q.dtype, q.device)
    out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, bias, True, dropout_p, False, scale=scale
    )
    # ``philox_seed`` / ``philox_offset`` are the flash kernel's opaque dropout
    # RNG state. Their shapes are build-dependent and need not match (this torch
    # build returns ``(2,)`` and ``()``), so flatten and concatenate into a
    # single 1-D tensor -- the ring stores ``rng_state`` via
    # ``ctx.save_for_backward``, which takes tensors, not tuples -- and split it
    # back in :func:`backward`. ``philox_offset`` is always the last element.
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
    seed: Optional[int] = None,  # noqa: ARG001 - unused, interface parity (see .math)
    rng_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward call to flash attention. Returns (dq, dk, dv).

    ``rng_state`` -- the flattened ``[*philox_seed, philox_offset]`` tensor
    :func:`forward` packed for this same chunk -- is split back here (offset is
    the last element) and replayed so the kernel reconstructs the identical
    dropout mask it applied in forward; without it (``None``, the
    ``dropout_p == 0`` case) any same-device int64 tensors work, hence zeros.

    See :func:`forward` for the ``(B, H, S, d)`` layout, the meaning of
    ``max_q`` / ``max_k`` / ``cum_seq_*``, ``causal``, and the
    ``key_mask`` mask-aware fork onto ``_scaled_dot_product_efficient_attention``
    (mirrored here via its ``_backward`` counterpart).
    """
    if key_mask is None:
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
        return dq, dk, dv

    bias = _build_bias(q.shape[-2], k.shape[-2], causal, key_mask, q.dtype, q.device)
    dq, dk, dv, _ = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        grad_out,
        q,
        k,
        v,
        bias,
        out,
        logsumexp,
        torch.zeros(1, dtype=torch.long, device=q.device),
        torch.zeros(1, dtype=torch.long, device=q.device),
        0.0,
        [True, True, True, False],
        is_causal=False,
        scale=scale,
    )
    return dq, dk, dv
