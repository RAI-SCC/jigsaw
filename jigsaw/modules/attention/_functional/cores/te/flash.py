"""The ``flash_attn`` (FA2) attention core (``te.flash``), for sm80-89 and ROCm.

Calls the ``flash_attn`` package's private ``_flash_attn_forward``/``_backward``
ops directly, since TE's wrapper and the public ``flash_attn_func`` do not
expose the LSE cheaply. Dropout is replayed from the kernel's ``rng_state``.
fp16/bf16 only; no FP8, bias, mask or GQA.
"""

from __future__ import annotations

import torch

from ._checks import check_inputs as _check_inputs
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


#: FA2 needs the head dim to be a multiple of 8 and at most 256; zero-padding
#: up to the next multiple is exact.
_HEAD_DIM_MULTIPLE = 8
_MAX_HEAD_DIM = 256


def check_head_dim(head_dim: int, device: torch.device) -> None:  # noqa: ARG001
    """Assert FA2 can serve ``head_dim``; ``device`` is for parity with :mod:`.cudnn`.

    Parameters
    ----------
    head_dim : int
        Per-head embedding dimension.
    device : torch.device
        Unused.
    """
    assert head_dim <= _MAX_HEAD_DIM, (
        f"attn_backend='te' (flash core) needs embed_dim // heads to be at most "
        f"{_MAX_HEAD_DIM}, got {head_dim}."
    )


def _pad_head_dim(t: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Zero-pad the trailing (head) dimension up to a multiple of 8."""
    pad = -head_dim % _HEAD_DIM_MULTIPLE
    return t if pad == 0 else torch.nn.functional.pad(t, [0, pad])


def _to_bshd(t: torch.Tensor, layout: Layout) -> torch.Tensor:
    """View ``t`` as ``bshd`` for FA2, without a copy.

    FA2 only needs ``stride(-1) == 1``, which the transpose keeps.
    """
    return t if layout == "bshd" else t.transpose(1, 2)


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float,
    seed: int | None = None,  # noqa: ARG001 - unused, interface parity (see ..math)
    *,
    layout: Layout = "bhsd",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """FA2 forward.

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
    _check_inputs(q, k, layout)
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
    # FA2 emits the LSE as (b, h, s_q) already, the ring's convention.
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
    seed: int | None = None,  # noqa: ARG001 - unused, interface parity (see ..math)
    rng_state: torch.Tensor | None = None,
    layout: Layout = "bhsd",
    **__: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FA2 backward from the saved (global) ``out`` / ``logsumexp``.

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
    head_dim = q.shape[-1]
    q_b, k_b, v_b, o_b, do_b = (
        _pad_head_dim(_to_bshd(t, layout), head_dim) for t in (q, k, v, out, grad_out)
    )
    # Contiguous grad buffers: q_b/k_b/v_b may be transposed views.
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
