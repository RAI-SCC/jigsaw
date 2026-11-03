"""Input checks shared by the TE cores, run before a kernel launch.

Both cores call the kernels' private entry points, which trust their inputs:
validation lives in the public wrappers (``te.DotProductAttention``,
``flash_attn_func``) they bypass. Inputs outside these limits make the kernels
fail with an opaque cuDNN error, poison the CUDA context (FA2's grid limit), or
silently compute garbage (mismatched ``k``/``v``, a missing dropout RNG state).
"""

from __future__ import annotations

import torch

from ._layout import Layout

#: Both kernels launch one grid row per batch entry and per head (CUDA grid
#: dims y/z), so neither may exceed 2**16 - 1.
MAX_GRID_DIM = 65535


def _seq_dim(layout: Layout) -> int:
    return 1 if layout == "bshd" else 2


def _without_seq(t: torch.Tensor, layout: Layout) -> tuple[int, ...]:
    shape = list(t.shape)
    del shape[_seq_dim(layout)]
    return tuple(shape)


def check_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: Layout
) -> None:
    """Check ``q``/``k``/``v`` are inputs the TE kernels can serve.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query, key and value in ``layout``.
    layout : {"bhsd", "bshd"}
        Layout of ``q``, ``k`` and ``v``.

    Raises
    ------
    ValueError
        If the shapes, dtypes or devices are outside the kernels' limits.
    """
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dim() != 4:
            raise ValueError(
                f"attn_backend='te' needs 4-D q/k/v ({layout}), got {name} of shape "
                f"{tuple(t.shape)}; pass x as (batch, seq, channels) and fold any "
                "other leading dims into batch."
            )
    # The kernels take batch, heads and head dim from q and the kv length from
    # k, and read v with k's: a mismatch reads out of bounds rather than failing.
    if k.shape != v.shape or _without_seq(q, layout) != _without_seq(k, layout):
        raise ValueError(
            f"attn_backend='te' needs k and v of equal shape, matching q in all but "
            f"seq len (no GQA), got q {tuple(q.shape)}, k {tuple(k.shape)}, "
            f"v {tuple(v.shape)} ({layout})."
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"attn_backend='te' needs fp16/bf16 q/k/v, got {q.dtype}.")
    if not q.dtype == k.dtype == v.dtype:
        raise ValueError(
            f"attn_backend='te' needs q/k/v of one dtype, got {q.dtype}, "
            f"{k.dtype}, {v.dtype}."
        )
    if not q.is_cuda:
        raise ValueError(f"attn_backend='te' needs CUDA tensors, got q on {q.device}.")
    if not q.device == k.device == v.device:
        raise ValueError(
            f"attn_backend='te' needs q/k/v on one device, got {q.device}, "
            f"{k.device}, {v.device}."
        )
    if q.numel() == 0 or k.numel() == 0:
        raise ValueError(
            f"attn_backend='te' needs non-empty q/k/v, got q {tuple(q.shape)}, "
            f"k {tuple(k.shape)}."
        )
    b, h = q.shape[0], q.shape[2 if layout == "bshd" else 1]
    if b > MAX_GRID_DIM or h > MAX_GRID_DIM:
        raise ValueError(
            f"attn_backend='te' supports at most {MAX_GRID_DIM} batch entries and "
            f"{MAX_GRID_DIM} heads per call, got batch={b}, heads={h}."
        )


def check_backward(
    q: torch.Tensor,
    logsumexp: torch.Tensor,
    dropout_p: float,
    rng_state: torch.Tensor | None,
    layout: Layout,
) -> None:
    """Check the saved forward state lets the backward replay it.

    Parameters
    ----------
    q : torch.Tensor
        Query in ``layout``.
    logsumexp : torch.Tensor
        Saved log-sum-exp, ``(b, h, s_q)`` or cuDNN's ``(b, h, s_q, 1)``.
    dropout_p : float
        Dropout probability.
    rng_state : torch.Tensor or None
        Dropout RNG state from the matching forward.
    layout : {"bhsd", "bshd"}
        Layout of ``q``.

    Raises
    ------
    ValueError
        If dropout lacks ``rng_state`` or ``logsumexp`` has the wrong shape.
    """
    # Without it the kernel replays a different dropout mask: wrong grads, no error.
    if dropout_p != 0.0 and rng_state is None:
        raise ValueError(
            "attn_backend='te' backward with dropout needs the forward's rng_state."
        )
    b, s_q = q.shape[0], q.shape[_seq_dim(layout)]
    h = q.shape[2 if layout == "bshd" else 1]
    lse_shape = tuple(logsumexp.shape)
    if lse_shape not in ((b, h, s_q), (b, h, s_q, 1)):
        raise ValueError(
            f"attn_backend='te' backward needs logsumexp of shape {(b, h, s_q)}, "
            f"got {lse_shape}."
        )
