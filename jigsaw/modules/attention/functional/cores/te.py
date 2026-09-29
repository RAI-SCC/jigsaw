"""TransformerEngine fused-attention core for the ring's inner attention.

Calls ``fused_attn_fwd`` and ``fused_attn_bwd`` directly to obtain the
log-sum-exp. Supports no bias, no mask and no FP8.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import transformer_engine.pytorch  # noqa: F401  (loads the C++ extension)
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        AttnBiasType,
        AttnMaskType,
        FusedAttnBackend,
        QKVLayout,
        SoftmaxType,
        fused_attn_bwd,
        fused_attn_fwd,
    )

    TE_CORE_AVAILABLE = True
except Exception:  # pragma: no cover - TE not installed (no container)
    TE_CORE_AVAILABLE = False


_LAYOUT = "bshd_bshd_bshd"


def _full_cu_seqlens(batch: int, seqlen: int, device: torch.device) -> torch.Tensor:
    """Return the ``int32`` cumulative sequence lengths of an unpadded batch."""
    return torch.arange(
        0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device=device
    )


def _fused_attn_backend(
    is_training: bool,
    dtype: torch.dtype,
    n_heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    head_dim: int,
    dropout_p: float,
) -> "tex.NVTE_Fused_Attn_Backend":
    """Return the cuDNN fused-attention sub-backend for a problem shape."""
    backend = tex.get_fused_attn_backend(
        is_training,
        TE_DType[dtype],
        TE_DType[dtype],
        QKVLayout[_LAYOUT],
        AttnBiasType["no_bias"],
        AttnMaskType["no_mask"],
        SoftmaxType["vanilla"],
        dropout_p,
        n_heads,
        n_heads,  # num_gqa_groups
        seqlen_q,
        seqlen_kv,
        head_dim,
        head_dim,
        -1,  # window_size left
        -1,  # window_size right
        False,  # return_max_logit
        False,  # cuda_graph
        False,  # deterministic
    )
    if backend == FusedAttnBackend["No_Backend"]:
        raise RuntimeError(
            "TransformerEngine has no fused-attention backend for this shape "
            f"(heads={n_heads}, seqlen_q={seqlen_q}, seqlen_kv={seqlen_kv}, "
            f"head_dim={head_dim}, dtype={dtype})."
        )
    return backend


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float,
    seed: Optional[int] = None,  # noqa: ARG001 - unused, interface parity (see .math)
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Compute TE fused attention.

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

    Returns
    -------
    out : torch.Tensor
        Attention output.
    logsumexp : torch.Tensor
        Log-sum-exp of the scores.
    rng_state : torch.Tensor | None
        Dropout RNG state of the kernel, ``None`` without dropout.
    """
    b, h, s_q, d = q.shape
    s_kv = k.shape[2]
    q_b, k_b, v_b = (t.transpose(1, 2).contiguous() for t in (q, k, v))  # -> bshd
    backend = _fused_attn_backend(True, q_b.dtype, h, s_q, s_kv, d, dropout_p)
    cu_q = _full_cu_seqlens(b, s_q, q.device)
    cu_kv = _full_cu_seqlens(b, s_kv, q.device)

    out, aux, *_ = fused_attn_fwd(
        True,  # is_training
        s_q,
        s_kv,
        cu_q,
        cu_kv,
        q_b,
        k_b,
        v_b,
        q_b.dtype,  # fake_dtype
        backend,
        None,  # attn_bias
        None,  # cu_seqlens_q_padded
        None,  # cu_seqlens_kv_padded
        None,  # page_table_k
        None,  # page_table_v
        None,  # s_quantizer
        None,  # o_quantizer
        scale,
        dropout_p,
        True,  # fast_zero_fill
        _LAYOUT,
        "no_bias",
        "no_mask",
        "vanilla",
        (-1, -1),
        None,  # bottom_right_diagonal
        None,  # rng_gen
        None,  # softmax_offset
        False,  # return_max_logit
        False,  # cuda_graph
    )
    # cuDNN returns LSE as (b, h, s_q, 1); the ring wants (b, h, s_q).
    lse = aux[0].squeeze(-1)
    rng_state = aux[1] if dropout_p > 0.0 else None
    return out.transpose(1, 2).contiguous(), lse, rng_state


def backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    scale: float,
    *_: object,
    dropout_p: float = 0.0,
    seed: Optional[int] = None,  # noqa: ARG001 - unused, interface parity (see .math)
    rng_state: Optional[torch.Tensor] = None,
    **__: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the TE fused-attention gradients.

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
    dropout_p : float
        Dropout probability.
    seed : int, optional
        Unused.
    rng_state : torch.Tensor, optional
        Dropout RNG state returned by :func:`forward`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Gradients ``(dq, dk, dv)``.
    """
    b, h, s_q, d = q.shape
    s_kv = k.shape[2]
    q_b, k_b, v_b, o_b, do_b = (
        t.transpose(1, 2).contiguous() for t in (q, k, v, out, grad_out)
    )
    backend = _fused_attn_backend(True, q_b.dtype, h, s_q, s_kv, d, dropout_p)
    cu_q = _full_cu_seqlens(b, s_q, q.device)
    cu_kv = _full_cu_seqlens(b, s_kv, q.device)
    # cuDNN's bwd wants the LSE shaped as its fwd emitted it, (b, h, s_q, 1),
    # and in fp32 -- a 16-bit LSE (the ring hands it back downcast to the model
    # dtype) makes the kernel recompute the softmax wrong.
    lse = logsumexp.to(torch.float32)
    if lse.dim() == 3:
        lse = lse.unsqueeze(-1)
    aux = [
        lse.contiguous(),
        rng_state
        if rng_state is not None
        else torch.zeros(2, dtype=torch.int64, device=q.device),
    ]

    dq, dk, dv, *_rest = fused_attn_bwd(
        s_q,
        s_kv,
        cu_q,
        cu_kv,
        q_b,
        k_b,
        v_b,
        o_b,
        do_b,
        q_b.dtype,  # fake_dtype
        TE_DType[q_b.dtype],  # dqkv_dtype
        aux,
        backend,
        None,  # cu_seqlens_q_padded
        None,  # cu_seqlens_kv_padded
        None,  # s_quantizer
        None,  # dp_quantizer
        None,  # dqkv_quantizer
        scale,
        dropout_p,
        True,  # fast_zero_fill
        _LAYOUT,
        "no_bias",
        "no_mask",
        "vanilla",
        (-1, -1),
        None,  # bottom_right_diagonal
        False,  # deterministic
        False,  # cuda_graph
    )
    return (
        dq.transpose(1, 2).contiguous(),
        dk.transpose(1, 2).contiguous(),
        dv.transpose(1, 2).contiguous(),
    )
