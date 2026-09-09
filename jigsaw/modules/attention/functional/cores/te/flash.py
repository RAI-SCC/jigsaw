"""The ``flash_attn`` (FA2) attention core (``te.flash``).

One of the two kernels ``attn_backend="te"`` dispatches to -- the Ampere arm;
:mod:`.cudnn` is the Hopper-and-newer arm. See :mod:`..te` for the rule.

This is the kernel ``te.DotProductAttention`` itself runs on Ampere: TE's
backend selector picks ``flash_attn`` over its own cuDNN kernels for every
non-FP8, no-mask, no-bias shape measured on an A100 (see
``experiments/probe_selector.py``). Selecting it here keeps ``attn_backend="te"``
meaning "what TransformerEngine would have run", while still exposing the
softmax LSE the ring needs.

Note what this module is *not*: it is not a TransformerEngine API. TE's own
``FlashAttention`` wrapper (``attention/dot_product_attention/backends.py``)
returns only ``out`` and throws the LSE away, so the ring cannot use it. This
module therefore calls the ``flash_attn`` package directly. Nor is it
:mod:`..sdpa`, which wraps ``torch.ops.aten._scaled_dot_product_flash_attention``
-- PyTorch's own vendored FA2 fork, a separate build from the Dao-AILab package
(verified: the aten op never loads ``flash_attn_2_cuda``).

Entry points: ``flash_attn``'s public ``flash_attn_func`` only yields the LSE
via ``return_attn_probs=True``, which under dropout also materialises
``S_dmask``, a ``(b, h, s, s)`` tensor -- measured at 20.3 ms vs 7.8 ms for the
same problem without it. So this core calls the underlying
``_flash_attn_forward`` / ``_flash_attn_backward`` custom ops with
``return_softmax=False``, which return the LSE and the kernel's own dropout
``rng_state`` and nothing else. That is the same posture as :mod:`..sdpa`,
which wraps torch's equivalent undocumented ops.

Scope: non-FP8, no bias, no mask, no GQA, fp16/bf16. ``layout`` selects the
caller's tensor convention (``"bhsd"`` for the ring, ``"bshd"`` for the
non-ring core -- FA2 speaks only the latter, so ``"bhsd"`` is transposed in and
out); LSE is ``(b, h, s)`` regardless, which is FA2's native shape.

Dropout replay follows :mod:`..sdpa` / :mod:`.cudnn`: the kernel draws its own
mask from an internally generated philox state, so :func:`forward` returns that
``rng_state`` verbatim and the ring threads it back into :func:`backward`.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

try:
    import flash_attn  # noqa: F401  (loads the CUDA extension)
    from flash_attn.flash_attn_interface import (
        _flash_attn_backward,
        _flash_attn_forward,
    )

    FLASH_CORE_AVAILABLE = True
except Exception:  # pragma: no cover - flash_attn not installed (no container)
    FLASH_CORE_AVAILABLE = False


#: See :data:`..cudnn.Layout`.
Layout = Literal["bhsd", "bshd"]

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


def _from_bshd(t: torch.Tensor, layout: Layout) -> torch.Tensor:
    """Inverse of :func:`_to_bshd`."""
    return t if layout == "bshd" else t.transpose(1, 2).contiguous()


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
    """FA2 forward. Returns ``(out, logsumexp (b,h,s), rng_state)``.

    ``out`` comes back in the same ``layout`` ``q``/``k``/``v`` were given in.
    ``rng_state`` is the kernel's own opaque philox state (``None`` when
    ``dropout_p == 0``), threaded back for :func:`backward` to replay.
    """
    head_dim = q.shape[-1]
    q_b, k_b, v_b = (_pad_head_dim(_to_bshd(t, layout), head_dim) for t in (q, k, v))

    out, lse, _s_dmask, rng_state = _flash_attn_forward(
        q_b,
        k_b,
        v_b,
        dropout_p,
        scale,
        causal=False,
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
        _pad_head_dim(_to_bshd(t, layout), head_dim)
        for t in (q, k, v, out, grad_out)
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
        False,  # causal
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
