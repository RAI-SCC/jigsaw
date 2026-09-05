"""TransformerEngine fused-attention core for the ring inner step.

The ring's online-softmax merge (:func:`..forward._attend_step`) needs a per-chunk
log-sum-exp from every inner attention call, and its backward
(:func:`..backward.ring_backward`) recomputes ``dQ/dK/dV`` from the *global* output
and LSE. ``te.DotProductAttention`` returns only the output, so this module drives
TransformerEngine's cuDNN fused-attention kernels directly via
``fused_attn_fwd`` / ``fused_attn_bwd``, whose ``aux_ctx_tensors[0]`` is the
softmax LSE.

Scope: non-FP8, no bias, no mask, ``bshd`` layout. Tensors follow the ring's
``(b, h, s, d)`` convention on the way in and out; LSE is ``(b, h, s)``.

Untested on GPU in this repo yet - covered by the ``te_required`` +
``gpu_required`` ring parity test.
"""

from __future__ import annotations

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
    """``[0, s, 2s, ..., b*s]`` int32 offsets for an unpadded ``bshd`` batch."""
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
    mask_type: str = "no_mask",
) -> "tex.NVTE_Fused_Attn_Backend":
    """Ask cuDNN which fused-attention sub-backend serves this problem shape."""
    backend = tex.get_fused_attn_backend(
        is_training,
        TE_DType[dtype],
        TE_DType[dtype],
        QKVLayout[_LAYOUT],
        AttnBiasType["no_bias"],
        AttnMaskType[mask_type],
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
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TE fused attention forward. Returns ``(out (b,h,s,d), logsumexp (b,h,s))``.

    ``causal`` is only ever set for the ring's diagonal block (q/k chunks
    starting at the same global offset, so ``s_q == s_kv``), where TE's
    top-left-aligned ``"causal"`` mask is the correct causal boundary; every
    other block is either fully visible (``causal=False``) or skipped entirely
    by the caller.
    """
    mask_type = "causal" if causal else "no_mask"
    b, h, s_q, d = q.shape
    s_kv = k.shape[2]
    q_b, k_b, v_b = (t.transpose(1, 2).contiguous() for t in (q, k, v))  # -> bshd
    backend = _fused_attn_backend(
        True, q_b.dtype, h, s_q, s_kv, d, dropout_p, mask_type
    )
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
        mask_type,
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
    return out.transpose(1, 2).contiguous(), lse


def backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    scale: float,
    *_: object,
    causal: bool = False,
    **__: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TE fused attention backward from the saved (global) ``out`` / ``logsumexp``.

    ``fused_attn_bwd`` recomputes ``S = scale * QK^T``, ``P = exp(S - lse)`` and
    the dot-product-attention gradients using the passed ``o`` and ``lse``; with
    the ring's globally normalised ``out`` / ``lse`` this yields the correct
    per-chunk ``dQ/dK/dV``. Dropout is forced off, so the ``rng_state`` slot of
    ``aux_ctx_tensors`` is a placeholder. See :func:`forward` for ``causal``.
    """
    mask_type = "causal" if causal else "no_mask"
    b, h, s_q, d = q.shape
    s_kv = k.shape[2]
    q_b, k_b, v_b, o_b, do_b = (
        t.transpose(1, 2).contiguous() for t in (q, k, v, out, grad_out)
    )
    backend = _fused_attn_backend(True, q_b.dtype, h, s_q, s_kv, d, 0.0, mask_type)
    cu_q = _full_cu_seqlens(b, s_q, q.device)
    cu_kv = _full_cu_seqlens(b, s_kv, q.device)
    # cuDNN's bwd wants the LSE shaped as its fwd emitted it: (b, h, s_q, 1).
    lse = logsumexp
    if lse.dim() == 3:
        lse = lse.unsqueeze(-1)
    aux = [
        lse.contiguous(),
        torch.zeros(2, dtype=torch.int64, device=q.device),  # rng_state, unused
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
        0.0,  # dropout
        True,  # fast_zero_fill
        _LAYOUT,
        "no_bias",
        mask_type,
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
