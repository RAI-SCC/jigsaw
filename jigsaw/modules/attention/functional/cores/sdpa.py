"""SDPA flash-attention core for the ring inner step (CUDA).

Wraps the two undocumented
``torch.ops.aten._scaled_dot_product_flash_attention{,_backward}`` ops, which --
unlike ``F.scaled_dot_product_attention`` -- return the softmax log-sum-exp the
ring's online-softmax merge and explicit-LSE backward need.

On CPU the selectors fall back to :mod:`.math`; see
:func:`..config._select_attn_core_forward_fn`.

The dropout path threads through the two extra tensors the flash kernel needs
to reproduce its mask, on top of the replay strategy the no-dropout path
already uses. It needs CUDA, so it is skipped where no GPU is present;
``tests/modules/attention/test_dropout.py::TestSdpaTeCoreDropoutReplay``
recovers the opaque mask from the forward output and checks that
:func:`backward` replays it (verified on an NVIDIA A100).
"""

from __future__ import annotations

from typing import Optional

import torch


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float,
    seed: Optional[int] = None,  # noqa: ARG001 - unused, interface parity (see .math)
    *,
    layout: str = "bhsd",
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
      generated internally (the op takes no seed input)
    * ``max_q`` / ``max_k`` (backward) are the query / key sequence lengths.
    * ``cum_seq_q`` / ``cum_seq_k`` (backward) are for nested/ragged batches
      only and stay ``None`` here.
    * backward returns ``(dq, dk, dv)`` in the same ``(B, H, S, d)`` layout.
    """
    assert layout == "bhsd", (
        f"sdpa core is bhsd-only; got layout={layout!r}. The bshd ring is TE-only."
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
        is_causal=False,
        scale=scale,
        return_debug_mask=False,
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
    dropout_p: float = 0.0,
    seed: Optional[int] = None,  # noqa: ARG001 - unused, interface parity (see .math)
    rng_state: Optional[torch.Tensor] = None,
    layout: str = "bhsd",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward call to flash attention. Returns (dq, dk, dv).

    ``rng_state`` -- the flattened ``[*philox_seed, philox_offset]`` tensor
    :func:`forward` packed for this same chunk -- is split back here (offset is
    the last element) and replayed so the kernel reconstructs the identical
    dropout mask it applied in forward; without it (``None``, the
    ``dropout_p == 0`` case) any same-device int64 tensors work, hence zeros.

    See :func:`forward` for the ``(B, H, S, d)`` layout and the meaning of
    ``max_q`` / ``max_k`` / ``cum_seq_*``.

    ``q``/``k``/``v``/``out``/``grad_out`` are cast to fp16 to match
    :func:`forward`, and ``logsumexp`` to fp32: the flash backward kernel reads
    it as fp32 and returns garbage for a 16-bit ``logsumexp`` (the ring's
    :func:`..backward.ring_backward` hands it back downcast to the model dtype).
    Gradients are returned in the input dtype.
    """
    assert layout == "bhsd", (
        f"sdpa core is bhsd-only; got layout={layout!r}. The bshd ring is TE-only."
    )
    if rng_state is not None:
        philox_seed, philox_offset = rng_state[:-1], rng_state[-1]
    else:
        philox_seed = torch.zeros(1, dtype=torch.long, device=q.device)
        philox_offset = torch.zeros(1, dtype=torch.long, device=q.device)

    in_dtype = grad_out.dtype
    grad_out, q, k, v, out = (
        t.to(torch.float16) for t in (grad_out, q, k, v, out)
    )
    logsumexp = logsumexp.to(torch.float32)

    dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        grad_out,
        q,
        k,
        v,
        out,
        logsumexp,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale,
        cum_seq_q=None,
        cum_seq_k=None,
        max_q=max_q,
        max_k=max_k,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
    )
    return dq.to(in_dtype), dk.to(in_dtype), dv.to(in_dtype)
