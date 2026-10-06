"""The ``flash_attn`` (FA2) attention core (``te.flash``).

The Ampere arm of ``attn_backend="te"`` (:mod:`.cudnn` is the Hopper-and-newer
arm; see :mod:`..te` for the rule) -- the same kernel ``te.DotProductAttention``
itself picks over cuDNN on Ampere for non-FP8, no-mask, no-bias shapes, kept
here so it still exposes the LSE the ring needs.

Not a TransformerEngine API: TE's own ``FlashAttention`` wrapper throws the LSE
away, so this module calls the ``flash_attn`` package directly instead. Also
not :mod:`..sdpa`, which wraps PyTorch's separately-built vendored FA2 fork.

Entry points: the public ``flash_attn_func`` only yields the LSE via
``return_attn_probs=True``, which under dropout also materialises a full
``(b, h, s, s)`` probability tensor and is measurably slower. So this core
calls the underlying ``_flash_attn_forward``/``_flash_attn_backward`` ops with
``return_softmax=False`` instead, returning just the LSE and the kernel's own
dropout ``rng_state`` -- the same posture :mod:`..sdpa` takes with torch's
equivalent undocumented ops.

Scope: non-FP8, no bias, no key-padding mask, no GQA, fp16/bf16 (``causal`` is
supported natively). ``layout`` is the caller's tensor convention --
``"bhsd"`` (ring) or ``"bshd"`` (non-ring, FA2's native layout); LSE is
``(b, h, s)`` either way, FA2's native shape.

Dropout replay follows :mod:`..sdpa`/:mod:`.cudnn`: the kernel draws its own
mask from an internal philox state, threaded back verbatim as ``rng_state``
for :func:`backward` to replay.
"""

from __future__ import annotations

from typing import Optional

import torch

from ._layout import Layout
from ._layout import from_bshd as _from_bshd

try:
    import flash_attn  # noqa: F401  (loads the CUDA extension)
    from flash_attn.flash_attn_interface import (
        _flash_attn_backward,
        _flash_attn_forward,
    )

    FLASH_CORE_AVAILABLE = True
except Exception:  # pragma: no cover - flash_attn not installed (no container)
    FLASH_CORE_AVAILABLE = False


#: FA2 requires the head dimension to be a multiple of 8 and at most 256; the
#: package's own ``FlashAttnFunc`` zero-pads up to the next multiple rather than
#: rejecting the shape, and so do we (:func:`_pad_head_dim`). Zero padding is
#: exact here: the padded lanes contribute 0 to every ``q @ k^T``, so the scores
#: -- and therefore the LSE -- are unchanged, and the padded lanes of ``out``
#: are 0 because ``v``'s are.
_HEAD_DIM_MULTIPLE = 8


def _pad_head_dim(t: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Zero-pad the trailing (head) dimension up to a multiple of 8."""
    pad = -head_dim % _HEAD_DIM_MULTIPLE
    return t if pad == 0 else torch.nn.functional.pad(t, [0, pad])


def _to_bshd(t: torch.Tensor, layout: Layout) -> torch.Tensor:
    """View ``t`` as ``bshd`` for FA2 (see :func:`..cudnn._to_bshd`).

    No ``.contiguous()``: FA2's ``_flash_attn_forward`` / ``_flash_attn_backward``
    read the batch/seq/head strides in-kernel and only copy an input whose *last*
    dim isn't unit-stride (their internal ``maybe_contiguous``). The ring's
    ``bhsd`` tensor transposed to ``bshd`` keeps ``stride(-1) == 1``, so it is
    consumed as a view -- forcing it contiguous here was up to 8 full
    ``(b, h, s, d)`` copies per ring step (3 fwd inputs + 5 bwd inputs) for
    nothing. cuDNN (:mod:`.cudnn`) is stricter and keeps its own copy.
    """
    return t if layout == "bshd" else t.transpose(1, 2)


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float,
    causal: bool = False,
    key_mask: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,  # noqa: ARG001 - unused, interface parity (see ..math)
    *,
    layout: Layout = "bhsd",
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """FA2 forward. Returns ``(out, logsumexp (b,h,s), rng_state)``.

    ``out`` comes back in the same ``layout`` ``q``/``k``/``v`` were given in.
    ``rng_state`` is the kernel's own opaque philox state (``None`` when
    ``dropout_p == 0``), threaded back for :func:`backward` to replay.

    ``causal`` is only ever set for the ring's diagonal block (see
    :func:`..cudnn.forward`); FA2 takes it natively. ``key_mask`` (key-padding
    masking) is not supported by this core and is rejected by the caller
    (``DistributedAttention._ring_attn_core``) before this is ever reached
    with a non-``None`` value.
    """
    if key_mask is not None:
        raise NotImplementedError(
            "key-padding masks are not supported by the TE flash core."
        )
    head_dim = q.shape[-1]
    q_b, k_b, v_b = (_pad_head_dim(_to_bshd(t, layout), head_dim) for t in (q, k, v))

    out, lse, _s_dmask, rng_state = _flash_attn_forward(
        q_b,
        k_b,
        v_b,
        dropout_p,
        scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )
    # FA2 emits the LSE as (b, h, s_q) already -- the ring's convention.
    return (
        _from_bshd(out[..., :head_dim].contiguous(), layout),
        lse,
        rng_state if dropout_p > 0.0 else None,
    )


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
    dropout_p: float = 0.0,
    seed: Optional[int] = None,  # noqa: ARG001 - unused, interface parity (see ..math)
    rng_state: Optional[torch.Tensor] = None,
    layout: Layout = "bhsd",
    **__: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FA2 backward from the saved (global) ``out`` / ``logsumexp``.

    FA2 recomputes ``S = scale * QK^T`` and ``P = exp(S - lse)`` from the passed
    ``out`` and ``lse`` rather than from a stored probability matrix, so handing
    it the ring's *globally* normalised pair yields the correct per-chunk
    ``dQ/dK/dV`` -- the same property :mod:`..sdpa` and :mod:`.cudnn` rely on.

    ``rng_state`` -- this chunk's philox state from :func:`forward` -- reproduces
    the identical dropout mask. FA2 writes the gradients into buffers we
    allocate (its op declares ``dq``/``dk``/``dv`` as mutated), and ``logsumexp``
    must be fp32: the ring hands it back downcast to the model dtype.
    """
    head_dim = q.shape[-1]
    q_b, k_b, v_b, o_b, do_b = (
        _pad_head_dim(_to_bshd(t, layout), head_dim) for t in (q, k, v, out, grad_out)
    )
    # Contiguous bshd output buffers: q_b/k_b/v_b are now bare transpose *views*
    # (``_to_bshd`` no longer copies), and FA2 writes the grads with whatever
    # strides these buffers carry -- a preserved-format ``empty_like`` would hand
    # it non-contiguous targets and just move the copy into ``_from_bshd``.
    dq, dk, dv = (
        torch.empty_like(t, memory_format=torch.contiguous_format)
        for t in (q_b, k_b, v_b)
    )

    _flash_attn_backward(
        do_b,
        q_b,
        k_b,
        v_b,
        o_b,
        logsumexp.to(torch.float32).contiguous(),
        dq,
        dk,
        dv,
        dropout_p,
        scale,
        causal,
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap
        None,  # alibi_slopes
        False,  # deterministic
        rng_state,
    )
    return tuple(  # type: ignore[return-value]
        _from_bshd(d[..., :head_dim].contiguous(), layout) for d in (dq, dk, dv)
    )
