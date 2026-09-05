"""SDPA flash-attention core for the ring inner step (CUDA).

Wraps the two undocumented
``torch.ops.aten._scaled_dot_product_flash_attention{,_backward}`` ops, which --
unlike ``F.scaled_dot_product_attention`` -- return the softmax log-sum-exp the
ring's online-softmax merge and explicit-LSE backward need.

On CPU the selectors fall back to :mod:`.math`; see
:func:`..config._select_attn_core_forward_fn`.
"""

from __future__ import annotations

import torch


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flash attention forward pass. Returns (out, logsumexp).

    Calling convention for the two undocumented
    ``torch.ops.aten._scaled_dot_product_flash_attention{,_backward}`` ops,
    worked out against a naive ``softmax(q @ k.mT * scale) @ v`` autograd
    reference and ``F.scaled_dot_product_attention``:

    * ``q``/``k``/``v``/``out``/``grad_out`` are ``(B, H, S, d)`` -- heads
      *before* sequence -- and contiguous. The public ``(B, S, H, d)`` layout
      must be transposed first (ring attention already works in ``(B, H, S, d)``).
    * forward returns ``(out, logsumexp, *rng_and_debug)``; keep the first two.
      ``logsumexp`` is ``(B, H, S)`` and is fed back into backward unchanged.
    * ``max_q`` / ``max_k`` (backward) are the query / key sequence lengths.
    * ``philox_seed`` / ``philox_offset`` (backward) are the dropout RNG state;
      with ``dropout_p == 0`` any same-device int64 tensors work, hence zeros.
    * ``cum_seq_q`` / ``cum_seq_k`` (backward) are for nested/ragged batches
      only and stay ``None`` here.
    * backward returns ``(dq, dk, dv)`` in the same ``(B, H, S, d)`` layout.

    ``causal`` is only ever set for the ring's diagonal block (q/k chunks
    starting at the same global offset), where the kernel's top-left-aligned
    triangular mask is the correct causal boundary; every other block is
    either fully visible (``causal=False``) or skipped by the caller entirely.
    """
    q, k, v = [x.to(torch.float16) for x in (q, k, v)]
    out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=scale,
        return_debug_mask=False,
    )

    return out, logsumexp


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward call to flash attention. Returns (dq, dk, dv).

    See :func:`forward` for the ``(B, H, S, d)`` layout, the meaning of
    ``max_q`` / ``max_k`` / ``philox_*`` / ``cum_seq_*``, and ``causal``.
    """
    dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        grad_out,
        q,
        k,
        v,
        out,
        logsumexp,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
        cum_seq_q=None,
        cum_seq_k=None,
        max_q=max_q,
        max_k=max_k,
        philox_seed=torch.zeros(1, dtype=torch.long, device=q.device),
        philox_offset=torch.zeros(1, dtype=torch.long, device=q.device),
    )
    return dq, dk, dv
