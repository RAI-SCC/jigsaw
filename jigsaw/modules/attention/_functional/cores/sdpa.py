"""Flash-attention core for the ring's inner attention (CUDA).

Wraps ``torch.ops.aten._scaled_dot_product_flash_attention`` and its backward,
which return the log-sum-exp needed by the ring.
"""

from __future__ import annotations

import torch


def _flash_dtype(q: torch.Tensor) -> torch.dtype:
    """Return the dtype the flash kernel runs in: ``q``'s if half, else bf16."""
    return q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float,
    seed: int | None = None,  # noqa: ARG001 - unused, interface parity (see .math)
    *,
    layout: str = "bhsd",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute flash attention.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query, key and value, shaped ``(b, h, s, d)``.
    scale : float
        Softmax scale.
    dropout_p : float
        Dropout probability.
    seed : int, optional
        Unused.
    layout : str
        Kept for interface parity with the other cores; sdpa only accepts
        ``"bhsd"``.

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
    dtype = _flash_dtype(q)
    q, k, v = [x.to(dtype) for x in (q, k, v)]
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

    # Pack the flash kernel's dropout RNG state (build-dependent shapes) into one
    # 1-D tensor, since ``ctx.save_for_backward`` takes tensors, not tuples.
    # :func:`backward` splits it; ``philox_offset`` is the last element.
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
    seed: int | None = None,  # noqa: ARG001 - unused, interface parity (see .math)
    rng_state: torch.Tensor | None = None,
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
    dropout_p : float
        Dropout probability.
    seed : int, optional
        Unused.
    rng_state : torch.Tensor, optional
        Dropout RNG state returned by :func:`forward`.
    layout : str
        Kept for interface parity with the other cores; sdpa only accepts
        ``"bhsd"``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Gradients ``(dq, dk, dv)`` in the input dtype.
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
    dtype = _flash_dtype(q)
    grad_out, q, k, v, out = (t.to(dtype) for t in (grad_out, q, k, v, out))
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
