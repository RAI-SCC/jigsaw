"""TransformerEngine's cuDNN fused-attention core (``te.cudnn``).

The Hopper-and-newer arm of ``attn_backend="te"`` (:mod:`.flash` is the Ampere
arm; see :mod:`..te` for the rule). ``te.DotProductAttention`` returns only the
output, but the ring needs a per-chunk LSE for its online-softmax merge
(:func:`...forward._attend_step`) and recomputes ``dQ/dK/dV`` from the
*global* output/LSE in backward (:func:`...backward.ring_backward`). So this
module drives ``fused_attn_fwd``/``fused_attn_bwd`` directly, reading the LSE
from ``aux_ctx_tensors[0]``.

The non-ring core
(:meth:`~jigsaw.modules.attention.attention.DistributedAttention._transformer_attn_core`)
goes through here too with ``layout="bshd"`` and just discards the LSE (cuDNN
computes it regardless as its own backward's normaliser, so it's a free
extra). What direct kernel calls buy there is the explicit
``fused_attn_bwd``, which ``te.DotProductAttention`` hides behind its own
``autograd.Function`` -- see
:class:`~jigsaw.modules.attention.attention.TransformerEngineAttention`.

Also the only TE path that can ever carry FP8 (not yet implemented here).

Scope: non-FP8, no bias, no mask, no GQA. ``layout`` is the caller's tensor
convention -- ``"bhsd"`` (ring) or ``"bshd"`` (non-ring, cuDNN's native
layout, no transposes); LSE is ``(b, h, s)`` either way.

Dropout replay mirrors :mod:`..sdpa`: cuDNN generates its own opaque RNG
state (``aux_ctx_tensors[1]``) rather than taking a caller seed, so it's
threaded back verbatim as ``rng_state`` for :func:`backward` to replay,
instead of rederiving a mask from an int like :mod:`..math` does.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from ._layout import Layout, from_bshd as _from_bshd

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

    CUDNN_CORE_AVAILABLE = True
except Exception:  # pragma: no cover - TE not installed (no container)
    CUDNN_CORE_AVAILABLE = False


_LAYOUT = "bshd_bshd_bshd"


def _to_bshd(t: torch.Tensor, layout: Layout) -> torch.Tensor:
    """Bring ``t`` into the contiguous ``bshd`` the fused kernels require."""
    return t.contiguous() if layout == "bshd" else t.transpose(1, 2).contiguous()


def _shape(q: torch.Tensor, k: torch.Tensor, layout: Layout) -> tuple[int, int, int, int, int]:
    """``(batch, heads, seqlen_q, seqlen_kv, head_dim)`` for either layout."""
    if layout == "bshd":
        b, s_q, h, d = q.shape
        return b, h, s_q, k.shape[1], d
    b, h, s_q, d = q.shape
    return b, h, s_q, k.shape[2], d


def _full_cu_seqlens(batch: int, seqlen: int, device: torch.device) -> torch.Tensor:
    """``[0, s, 2s, ..., b*s]`` int32 offsets for an unpadded ``bshd`` batch."""
    return torch.arange(
        0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device=device
    )


@functools.lru_cache(maxsize=None)
def _fused_attn_backend(
    is_training: bool,
    dtype: torch.dtype,
    n_heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    head_dim: int,
    dropout_p: float,
) -> "tex.NVTE_Fused_Attn_Backend":
    """Ask cuDNN which fused-attention sub-backend serves this problem shape.

    Cached on its arguments: the ring calls it once per chunk per ring step, and
    the answer only depends on the (shape, dtype, dropout) problem description.
    """
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
    *,
    layout: Layout = "bhsd",
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """TE fused attention forward.

    Returns ``(out, logsumexp (b,h,s), rng_state)``. ``out`` comes back in the
    same ``layout`` ``q``/``k``/``v`` were given in -- ``(b,h,s,d)`` by default
    (the ring's convention), ``(b,s,h,d)`` for ``layout="bshd"``, which skips
    the transposes entirely. ``logsumexp`` is ``(b,h,s)`` either way.
    ``rng_state`` is cuDNN's own opaque dropout RNG state (``None`` when
    ``dropout_p == 0``), threaded back for :func:`backward` to replay (see
    module docstring).
    """
    b, h, s_q, s_kv, d = _shape(q, k, layout)
    q_b, k_b, v_b = (_to_bshd(t, layout) for t in (q, k, v))
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
    return _from_bshd(out, layout), lse, rng_state


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
    layout: Layout = "bhsd",
    **__: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TE fused attention backward from the saved (global) ``out`` / ``logsumexp``.

    ``fused_attn_bwd`` recomputes ``S = scale * QK^T``, ``P = exp(S - lse)`` and
    the dot-product-attention gradients using the passed ``o`` and ``lse``; with
    the ring's globally normalised ``out`` / ``lse`` this yields the correct
    per-chunk ``dQ/dK/dV``. ``rng_state`` -- cuDNN's own dropout RNG state,
    threaded back from this same chunk's :func:`forward` call -- reproduces the
    identical dropout mask; ``None`` (the ``dropout_p == 0`` case) falls back
    to a placeholder, matching cuDNN's own "unused" convention for no-dropout.

    ``layout`` describes ``q``/``k``/``v``/``out``/``grad_out`` and the returned
    gradients, exactly as in :func:`forward`; ``logsumexp`` is ``(b,h,s)``
    regardless.
    """
    b, h, s_q, s_kv, d = _shape(q, k, layout)
    q_b, k_b, v_b, o_b, do_b = (
        _to_bshd(t, layout) for t in (q, k, v, out, grad_out)
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
        _from_bshd(dq, layout),
        _from_bshd(dk, layout),
        _from_bshd(dv, layout),
    )
