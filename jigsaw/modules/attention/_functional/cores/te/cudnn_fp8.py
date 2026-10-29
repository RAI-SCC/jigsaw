"""TransformerEngine's cuDNN *FP8* fused-attention core (``te.cudnn_fp8``), sm90+.

Same contract as :mod:`.cudnn` (so the ring, Ulysses and plain paths use it
unchanged) but the two attention GEMMs run on FP8 tensor cores. Select it with
``JIGSAW_TE_CORE=cudnn_fp8`` together with ``attn_backend="te"``; the optional
``JIGSAW_TE_FP8_FORMAT`` picks the QKV format (see below).

This mirrors what ``te.DotProductAttention`` does under ``fp8_autocast`` with a
``Float8CurrentScaling`` recipe (TE's table in ``FusedAttention``):

====================  ===================================  ============  ==========
direction             inputs                               intermediate  outputs
====================  ===================================  ============  ==========
forward               Q/K/V: E4M3, current scaling         S: E4M3 (DS)  O: F16
backward              Q/K/V (re-quantized), dO: E5M2 (CS)  dP: E5M2 (DS) dQ/dK/dV: F16
====================  ===================================  ============  ==========

* **Stateless per-call scaling for Q/K/V/O/dO/dQKV.** Current scaling derives the
  scale from the tensor it is quantizing, so the core needs no amax history
  shared across layers or ring steps. The backward re-quantizes the same bf16
  chunk it quantized in the forward, which gives the same FP8 values.
* **S and dP use delayed scaling**, which the cuDNN FP8 kernel requires: a
  module-level scale/amax pair per device. S is a softmax probability, so its
  scale is fixed at the E4M3 maximum; dP's scale follows the amax the kernel
  records in the previous backward call (the first call uses scale 1).
  The state is shared by every layer on the device.
* Inputs and outputs stay bf16/fp16: the quantization is internal, so the ring's
  LSE merge and dK/dV accumulation are unchanged.

Not supported: dropout, bias, mask, GQA (as for :mod:`.cudnn`), and FP8 on
anything below sm90.

Bound to a TE release: see :data:`TE_FP8_VERSION`. The calls below follow the
FP8 path of ``te.pytorch.attention.dot_product_attention.backends`` of the TE
release the pin names and are not exercised in CI (no GPU / TE there); run
``tests/modules/attention/test_te_fp8_core.py`` in the container after changing
the TE version.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import torch

from . import cudnn
from ._checks import check_inputs as _check_inputs
from ._layout import Layout
from ._layout import from_bshd as _from_bshd

#: ``major.minor`` of the TE release this core is validated against; ``None``
#: until it has been run in the container (then a warning names the TE in use).
TE_FP8_VERSION: str | None = None

CUDNN_FP8_CORE_AVAILABLE = cudnn.CUDNN_CORE_AVAILABLE

if CUDNN_FP8_CORE_AVAILABLE:
    try:
        import transformer_engine as _te
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.attention.dot_product_attention.utils import (
            combine_and_quantize,
        )
        from transformer_engine.pytorch.cpp_extensions.fused_attn import (
            fused_attn_bwd,
            fused_attn_fwd,
        )
        from transformer_engine.pytorch.tensor.float8_tensor import (
            Float8CurrentScalingQuantizer,
            Float8Quantizer,
        )
    except Exception:  # pragma: no cover - TE too old/new for the FP8 path
        CUDNN_FP8_CORE_AVAILABLE = False

check_head_dim = cudnn.check_head_dim

_LAYOUT = cudnn._LAYOUT
_E4M3_MAX = 448.0
_E5M2_MAX = 57344.0
_warned_version = False


def _check_version() -> None:
    """Fail on a TE other than the pinned one; warn once while nothing is pinned."""
    global _warned_version
    version = ".".join(_te.__version__.split(".")[:2])
    if TE_FP8_VERSION is not None:
        if version != TE_FP8_VERSION:
            raise RuntimeError(
                f"cudnn_fp8 core is validated against TE {TE_FP8_VERSION}.x, found "
                f"{_te.__version__}. Install the pinned release or update "
                f"TE_FP8_VERSION after re-running tests/modules/attention/test_te_fp8_core.py."
            )
    elif not _warned_version:
        _warned_version = True
        warnings.warn(
            f"cudnn_fp8 core has not been validated against any TE release "
            f"(running TE {_te.__version__}); check the output against the bf16 core.",
            stacklevel=3,
        )


class _State:
    """Delayed-scaling state of the S and dP quantizers for one device."""

    def __init__(self, device: torch.device) -> None:
        def t(v: float) -> torch.Tensor:
            return torch.full((1,), v, dtype=torch.float32, device=device)

        self.s_scale, self.s_amax = t(_E4M3_MAX), t(0.0)
        self.dp_scale, self.dp_amax = t(1.0), t(0.0)
        self.device = device

    def s_quantizer(self) -> Any:
        q = Float8Quantizer(self.s_scale, self.s_amax, tex.DType.kFloat8E4M3)
        q.internal = True
        q.set_usage(rowwise=True, columnwise=False)
        return q

    def dp_quantizer(self) -> Any:
        q = Float8Quantizer(self.dp_scale, self.dp_amax, tex.DType.kFloat8E5M2)
        q.internal = True
        q.set_usage(rowwise=True, columnwise=False)
        return q

    def update_dp(self) -> None:
        """Set dP's scale from the amax the backward just recorded (no host sync)."""
        amax = self.dp_amax.clamp(min=torch.finfo(torch.float32).tiny)
        new = _E5M2_MAX / (2.0 * amax)  # 1 bit of headroom
        self.dp_scale.copy_(torch.where(self.dp_amax > 0, new, self.dp_scale))
        self.dp_amax.zero_()


_STATES: dict[torch.device, _State] = {}


def _state(device: torch.device) -> _State:
    st = _STATES.get(device)
    if st is None:
        st = _STATES[device] = _State(device)
    return st


def _fp8_dtypes() -> tuple[Any, Any]:
    """``(qkv_dtype, grad_dtype)`` from ``JIGSAW_TE_FP8_FORMAT``.

    ``hybrid`` (default): E4M3 forward, E5M2 gradients (TE's default recipe).
    ``e4m3``: E4M3 for the gradients too.
    """
    fmt = os.environ.get("JIGSAW_TE_FP8_FORMAT", "hybrid")
    if fmt == "hybrid":
        return tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2
    if fmt == "e4m3":
        return tex.DType.kFloat8E4M3, tex.DType.kFloat8E4M3
    raise ValueError(f"JIGSAW_TE_FP8_FORMAT={fmt!r}; expected 'hybrid' or 'e4m3'.")


def _cs_quantizer(dtype: Any, device: torch.device) -> Any:
    q = Float8CurrentScalingQuantizer(dtype, device)
    q.internal = False
    q.set_usage(rowwise=True, columnwise=False)
    return q


def _dequant(t: Any, like: torch.Tensor) -> torch.Tensor:
    """Return a kernel output as a plain tensor of ``like``'s dtype and shape."""
    if hasattr(t, "dequantize"):
        t = t.dequantize(dtype=like.dtype)
    out: torch.Tensor = t.view(like.shape)
    return out


def _quantize_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, device: torch.device
) -> tuple[Any, Any, Any, Any, Any]:
    qkv_dtype, _ = _fp8_dtypes()
    quantizer = _cs_quantizer(qkv_dtype, device)
    q8, k8, v8, layout, scale_inv_format = combine_and_quantize(
        _LAYOUT, q, k, v, quantizer, used_in_backward=True
    )
    return q8, k8, v8, layout, scale_inv_format


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float,
    seed: int | None = None,  # noqa: ARG001 - interface parity
    *,
    layout: Layout = "bhsd",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """FP8 fused attention forward; see :func:`.cudnn.forward` for the contract.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query, key and value in fp16/bf16 and ``layout``. They are quantized to FP8
        inside the call.
    scale : float
        Softmax scale.
    dropout_p : float
        Must be 0.
    seed : int, optional
        Unused.
    layout : {"bhsd", "bshd"}, optional
        Layout of the inputs and of the returned ``out``.

    Returns
    -------
    out : torch.Tensor
        Attention output in ``layout``, in the dtype of ``q``.
    logsumexp : torch.Tensor
        Log-sum-exp, ``(b, h, s)``, fp32.
    rng_state : None
        Always ``None``: there is no dropout.
    """
    _check_version()
    assert dropout_p == 0.0, "cudnn_fp8 core does not support dropout."
    _check_inputs(q, k, layout)
    b, h, s_q, s_kv, d = cudnn._shape(q, k, layout)
    assert s_q > 1 or s_kv > 1, "attn_backend='te' (cuDNN core) needs seq len > 1."
    q_b, k_b, v_b = (cudnn._to_bshd(t, layout) for t in (q, k, v))
    device = q.device
    st = _state(device)

    q8, k8, v8, qkv_layout, scale_inv_format = _quantize_qkv(q_b, k_b, v_b, device)
    qkv_dtype, _ = _fp8_dtypes()
    backend = cudnn._fused_attn_backend(True, qkv_dtype, h, s_q, s_kv, d, dropout_p)
    cu_q = cudnn._full_cu_seqlens(b, s_q, device)
    cu_kv = cudnn._full_cu_seqlens(b, s_kv, device)

    out, aux, *_ = cudnn._te_call(
        fused_attn_fwd,
        required={
            "is_training": True,
            "max_seqlen_q": s_q,
            "max_seqlen_kv": s_kv,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_kv": cu_kv,
            "q": q8,
            "k": k8,
            "v": v8,
            "fake_dtype": q_b.dtype,
            "fused_attention_backend": backend,
            "attn_scale": scale,
            "dropout": dropout_p,
            "qkv_layout": qkv_layout,
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
            "s_quantizer": st.s_quantizer(),
            "o_quantizer": _cs_quantizer(qkv_dtype, device),
            "fast_zero_fill": True,
            "o_format": "bshd",
            "qkv_scale_inv_format": scale_inv_format,
            "softmax_type": "vanilla",
            "bottom_right_diagonal": None,
            "rng_gen": None,
            "softmax_offset": None,
            "return_max_logit": False,
            "cuda_graph": False,
        },
    )
    out = _dequant(out, q_b)
    # cuDNN returns the stats as (b, h, s_q, 1); the ring wants (b, h, s_q).
    lse = aux[0].squeeze(-1)
    return _from_bshd(out, layout), lse, None


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
    seed: int | None = None,  # noqa: ARG001 - interface parity
    rng_state: torch.Tensor | None = None,  # noqa: ARG001
    layout: Layout = "bhsd",
    **__: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP8 fused attention backward from the saved (global) ``out`` / ``logsumexp``.

    Same contract as :func:`.cudnn.backward`: with the ring's globally
    normalized ``out`` / ``logsumexp`` this yields the per-chunk dQ/dK/dV.
    ``q``/``k``/``v`` are re-quantized, and ``out`` stays in 16 bit (TE's default
    ``NVTE_DPA_FP8CS_O_in_F16=1``; with ``0`` it is quantized like Q/K/V).

    Parameters
    ----------
    grad_out : torch.Tensor
        Gradient of the output, in ``layout``.
    q, k, v : torch.Tensor
        Query, key and value in fp16/bf16 and ``layout``.
    out : torch.Tensor
        Saved (globally normalized) attention output, in ``layout``.
    logsumexp : torch.Tensor
        Saved (global) log-sum-exp, ``(b, h, s)``.
    scale : float
        Softmax scale.
    *_ : object
        Unused; interface parity with the other cores.
    dropout_p : float, optional
        Must be 0.
    seed : int, optional
        Unused.
    rng_state : torch.Tensor, optional
        Unused: there is no dropout.
    layout : {"bhsd", "bshd"}, optional
        Layout of the tensors and of the returned gradients. Default ``"bhsd"``.
    **__ : object
        Unused; interface parity with the other cores.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Gradients ``(dq, dk, dv)`` in ``layout``.
    """
    _check_version()
    assert dropout_p == 0.0, "cudnn_fp8 core does not support dropout."
    b, h, s_q, s_kv, d = cudnn._shape(q, k, layout)
    q_b, k_b, v_b, o_b, do_b = (
        cudnn._to_bshd(t, layout) for t in (q, k, v, out, grad_out)
    )
    device = q.device
    st = _state(device)
    qkv_dtype, grad_dtype = _fp8_dtypes()

    q8, k8, v8, qkv_layout, scale_inv_format = _quantize_qkv(q_b, k_b, v_b, device)
    do8 = _cs_quantizer(grad_dtype, device)(do_b)
    if os.environ.get("NVTE_DPA_FP8CS_O_in_F16", "1") != "1":
        o_arg: Any = _cs_quantizer(qkv_dtype, device)(o_b)
    else:
        o_arg = o_b

    backend = cudnn._fused_attn_backend(True, qkv_dtype, h, s_q, s_kv, d, dropout_p)
    cu_q = cudnn._full_cu_seqlens(b, s_q, device)
    cu_kv = cudnn._full_cu_seqlens(b, s_kv, device)
    # cuDNN's bwd needs the LSE as (b, h, s_q, 1) in fp32.
    lse = logsumexp.to(torch.float32)
    if lse.dim() == 3:
        lse = lse.unsqueeze(-1)
    aux = [lse.contiguous(), torch.zeros(2, dtype=torch.int64, device=device)]

    dq, dk, dv, *_rest = cudnn._te_call(
        fused_attn_bwd,
        required={
            "max_seqlen_q": s_q,
            "max_seqlen_kv": s_kv,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_kv": cu_kv,
            "q": q8,
            "k": k8,
            "v": v8,
            "o": o_arg,
            "d_o": do8,
            "fake_dtype": q_b.dtype,
            "aux_ctx_tensors": aux,
            "fused_attention_backend": backend,
            "attn_scale": scale,
            "dropout": dropout_p,
            "qkv_layout": qkv_layout,
            "attn_bias_type": "no_bias",
            "attn_mask_type": "no_mask",
            "window_size": (-1, -1),
        },
        optional={
            "dqkv_dtype": cudnn.TE_DType[q_b.dtype],
            "cu_seqlens_q_padded": None,
            "cu_seqlens_kv_padded": None,
            "s_quantizer": st.s_quantizer(),
            "dp_quantizer": st.dp_quantizer(),
            "dqkv_quantizer": _cs_quantizer(grad_dtype, device),
            "fast_zero_fill": True,
            "o_format": "bshd",
            "do_format": "bshd",
            "dqkv_layout": _LAYOUT,
            "qkv_scale_inv_format": scale_inv_format,
            "do_scale_inv_format": None,
            "softmax_type": "vanilla",
            "bottom_right_diagonal": None,
            "deterministic": False,
            "cuda_graph": False,
        },
    )
    st.update_dp()
    return (
        _from_bshd(_dequant(dq, q_b), layout),
        _from_bshd(_dequant(dk, k_b), layout),
        _from_bshd(_dequant(dv, v_b), layout),
    )
