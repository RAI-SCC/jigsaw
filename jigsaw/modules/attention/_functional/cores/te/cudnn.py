"""TransformerEngine's cuDNN fused-attention core (``te.cudnn``), for sm90+.

Calls ``fused_attn_fwd``/``fused_attn_bwd`` directly, since
``te.DotProductAttention`` does not expose the LSE the ring needs. Dropout is
replayed from cuDNN's opaque ``rng_state``. No FP8, bias, mask or GQA.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional

import torch

from ._layout import Layout
from ._layout import from_bshd as _from_bshd

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


# --- transformer-engine ABI compatibility ---------------------------------
#
# The private ``fused_attn_fwd``/``bwd`` signatures change between TE releases,
# so every argument goes by keyword. Meaningful ones are required (missing ->
# error); ones restating a TE default are dropped if the installed TE lacks them.
#
# Parameter names TE has used for the same logical argument, best first.
_TE_ALIASES: dict[str, tuple[str, ...]] = {
    # the mandatory block
    "is_training": ("is_training",),
    "max_seqlen_q": ("max_seqlen_q",),
    "max_seqlen_kv": ("max_seqlen_kv",),
    "cu_seqlens_q": ("cu_seqlens_q",),
    "cu_seqlens_kv": ("cu_seqlens_kv",),
    "q": ("q",),
    "k": ("k",),
    "v": ("v",),
    "o": ("o",),
    "d_o": ("d_o", "do"),
    "fake_dtype": ("fake_dtype", "qkv_dtype"),
    "aux_ctx_tensors": ("aux_ctx_tensors",),
    "fused_attention_backend": ("fused_attention_backend",),
    # the semantic tail
    "attn_scale": ("attn_scale", "scale"),
    "dropout": ("dropout", "dropout_p"),
    "qkv_layout": ("qkv_layout",),
    "attn_bias_type": ("attn_bias_type",),
    "attn_mask_type": ("attn_mask_type",),
    "window_size": ("window_size",),
}


@functools.lru_cache(maxsize=None)
def _te_params(fn: Callable) -> frozenset[str]:
    """Return the parameter names of ``fn``."""
    return frozenset(inspect.signature(fn).parameters)


def _te_call(
    fn: Callable,
    required: dict[str, Any],
    optional: dict[str, Any],
) -> Any:
    """Call a TE fused-attn entry point by keyword, across TE versions.

    ``required`` must exist in the installed signature (under one of its
    :data:`_TE_ALIASES`); ``optional`` is filtered to what it accepts. Raises
    :class:`RuntimeError` if the signature lacks a required argument or needs
    one not supplied.
    """
    accepted = _te_params(fn)

    kwargs: dict[str, Any] = {}
    for logical, value in required.items():
        for candidate in _TE_ALIASES[logical]:
            if candidate in accepted:
                kwargs[candidate] = value
                break
        else:
            raise RuntimeError(
                f"transformer-engine's {fn.__name__} has no parameter for "
                f"'{logical}' (tried {_TE_ALIASES[logical]}). The installed TE "
                f"is not compatible with this module; signature:\n"
                f"  {fn.__name__}{inspect.signature(fn)}"
            )
    kwargs.update({k: v for k, v in optional.items() if k in accepted})

    # Any mandatory parameter left unsupplied is a genuine incompatibility; say so
    # with the signature rather than letting Python raise a bare TypeError.
    missing = [
        name
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is inspect.Parameter.empty and name not in kwargs
    ]
    if missing:
        raise RuntimeError(
            f"transformer-engine's {fn.__name__} requires {missing}, which this "
            f"core does not supply. The installed TE is not compatible with "
            f"this module; signature:\n  {fn.__name__}{inspect.signature(fn)}"
        )

    return fn(**kwargs)


def _to_bshd(t: torch.Tensor, layout: Layout) -> torch.Tensor:
    """Bring ``t`` into the contiguous ``bshd`` the fused kernels require."""
    return t.contiguous() if layout == "bshd" else t.transpose(1, 2).contiguous()


def _shape(
    q: torch.Tensor, k: torch.Tensor, layout: Layout
) -> tuple[int, int, int, int, int]:
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
    Raises :class:`RuntimeError` if no sub-backend serves it.
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
    seed: Optional[int] = None,  # noqa: ARG001 - unused, interface parity (see ..math)
    *,
    layout: Layout = "bhsd",
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """TE fused attention forward.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query, key and value in ``layout``.
    scale : float
        Softmax scale.
    dropout_p : float
        Dropout probability.
    seed : int, optional
        Unused; the kernel draws its own RNG state.
    layout : {"bhsd", "bshd"}, optional
        Layout of ``q``/``k``/``v`` and of the returned ``out``. Default ``"bhsd"``.

    Returns
    -------
    out : torch.Tensor
        Attention output in ``layout``.
    logsumexp : torch.Tensor
        Log-sum-exp, ``(b, h, s)``.
    rng_state : torch.Tensor or None
        Kernel dropout RNG state for :func:`backward`; ``None`` without dropout.
    """
    b, h, s_q, s_kv, d = _shape(q, k, layout)
    q_b, k_b, v_b = (_to_bshd(t, layout) for t in (q, k, v))
    backend = _fused_attn_backend(True, q_b.dtype, h, s_q, s_kv, d, dropout_p)
    cu_q = _full_cu_seqlens(b, s_q, q.device)
    cu_kv = _full_cu_seqlens(b, s_kv, q.device)

    out, aux, *_ = _te_call(
        fused_attn_fwd,
        required={
            "is_training": True,
            "max_seqlen_q": s_q,
            "max_seqlen_kv": s_kv,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_kv": cu_kv,
            "q": q_b,
            "k": k_b,
            "v": v_b,
            "fake_dtype": q_b.dtype,
            "fused_attention_backend": backend,
            "attn_scale": scale,
            "dropout": dropout_p,
            "qkv_layout": _LAYOUT,
            "attn_bias_type": "no_bias",
            "attn_mask_type": "no_mask",
            "window_size": (-1, -1),
        },
        optional={
            "attn_bias": None,
            "cu_seqlens_q_padded": None,
            "cu_seqlens_kv_padded": None,
            "page_table_k": None,
            "page_table_v": None,
            "s_quantizer": None,
            "o_quantizer": None,
            "fast_zero_fill": True,
            # TE >= 2.16 defaults the output to sbhd; older TE follows qkv_layout.
            "o_format": "bshd",
            "softmax_type": "vanilla",
            "bottom_right_diagonal": None,
            "rng_gen": None,
            "softmax_offset": None,
            "return_max_logit": False,
            "cuda_graph": False,
        },
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
    seed: Optional[int] = None,  # noqa: ARG001 - unused, interface parity (see ..math)
    rng_state: Optional[torch.Tensor] = None,
    layout: Layout = "bhsd",
    **__: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TE fused attention backward from the saved (global) ``out`` / ``logsumexp``.

    With the ring's globally normalized ``out`` / ``logsumexp`` this yields the
    correct per-chunk ``dQ/dK/dV``.

    Parameters
    ----------
    grad_out : torch.Tensor
        Gradient of the output, in ``layout``.
    q, k, v : torch.Tensor
        Query, key and value, in ``layout``.
    out : torch.Tensor
        Saved (globally normalized) attention output, in ``layout``.
    logsumexp : torch.Tensor
        Saved (global) log-sum-exp, ``(b, h, s)``.
    scale : float
        Softmax scale.
    *_ : object
        Unused; interface parity with the other cores.
    dropout_p : float, optional
        Dropout probability. Default 0.
    seed : int, optional
        Unused; ``rng_state`` replays the mask.
    rng_state : torch.Tensor, optional
        Dropout RNG state from this chunk's :func:`forward`.
    layout : {"bhsd", "bshd"}, optional
        Layout of the tensors and of the returned gradients. Default ``"bhsd"``.
    **__ : object
        Unused; interface parity with the other cores.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Gradients ``(dq, dk, dv)`` in ``layout``.
    """
    b, h, s_q, s_kv, d = _shape(q, k, layout)
    q_b, k_b, v_b, o_b, do_b = (_to_bshd(t, layout) for t in (q, k, v, out, grad_out))
    backend = _fused_attn_backend(True, q_b.dtype, h, s_q, s_kv, d, dropout_p)
    cu_q = _full_cu_seqlens(b, s_q, q.device)
    cu_kv = _full_cu_seqlens(b, s_kv, q.device)
    # cuDNN's bwd needs the LSE as (b, h, s_q, 1) in fp32; 16-bit gives wrong grads.
    lse = logsumexp.to(torch.float32)
    if lse.dim() == 3:
        lse = lse.unsqueeze(-1)
    aux = [
        lse.contiguous(),
        rng_state
        if rng_state is not None
        else torch.zeros(2, dtype=torch.int64, device=q.device),
    ]

    dq, dk, dv, *_rest = _te_call(
        fused_attn_bwd,
        required={
            "max_seqlen_q": s_q,
            "max_seqlen_kv": s_kv,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_kv": cu_kv,
            "q": q_b,
            "k": k_b,
            "v": v_b,
            "o": o_b,
            "d_o": do_b,
            "fake_dtype": q_b.dtype,
            "aux_ctx_tensors": aux,
            "fused_attention_backend": backend,
            "attn_scale": scale,
            "dropout": dropout_p,
            "qkv_layout": _LAYOUT,
            "attn_bias_type": "no_bias",
            "attn_mask_type": "no_mask",
            "window_size": (-1, -1),
        },
        optional={
            # TE < 2.16 wants the dQ/dK/dV dtype spelled out; 2.16 dropped it.
            "dqkv_dtype": TE_DType[q_b.dtype],
            "cu_seqlens_q_padded": None,
            "cu_seqlens_kv_padded": None,
            "s_quantizer": None,
            "dp_quantizer": None,
            "dqkv_quantizer": None,
            "fast_zero_fill": True,
            # TE >= 2.16 defaults these to sbhd; see the forward.
            "o_format": "bshd",
            "do_format": "bshd",
            "dqkv_layout": _LAYOUT,
            "softmax_type": "vanilla",
            "bottom_right_diagonal": None,
            "deterministic": False,
            "cuda_graph": False,
        },
    )
    return (
        _from_bshd(dq, layout),
        _from_bshd(dk, layout),
        _from_bshd(dv, layout),
    )
