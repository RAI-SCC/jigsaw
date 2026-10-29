"""Input checks shared by the TE cores, run before a kernel launch.

Inputs outside these limits make the kernels fail with an opaque cuDNN error
or, for FA2's grid limit, a CUDA error that poisons the context.
"""

from __future__ import annotations

import torch

from ._layout import Layout

#: Both kernels launch one grid row per batch entry and per head (CUDA grid
#: dims y/z), so neither may exceed 2**16 - 1.
MAX_GRID_DIM = 65535


def check_inputs(q: torch.Tensor, k: torch.Tensor, layout: Layout) -> None:
    """Assert ``q``/``k`` are inputs the TE kernels can serve.

    Parameters
    ----------
    q, k : torch.Tensor
        Query and key in ``layout``.
    layout : {"bhsd", "bshd"}
        Layout of ``q`` and ``k``.
    """
    assert q.dim() == 4 and k.dim() == 4, (
        f"attn_backend='te' needs 4-D q/k/v ({layout}), got q of shape "
        f"{tuple(q.shape)}; pass x as (batch, seq, channels) and fold any "
        "other leading dims into batch."
    )
    assert q.is_cuda, f"attn_backend='te' needs CUDA tensors, got q on {q.device}."
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"attn_backend='te' needs fp16/bf16 q/k/v, got {q.dtype}."
    )
    assert q.numel() > 0 and k.numel() > 0, (
        f"attn_backend='te' needs non-empty q/k/v, got q {tuple(q.shape)}, "
        f"k {tuple(k.shape)}."
    )
    b, h = q.shape[0], q.shape[2 if layout == "bshd" else 1]
    assert b <= MAX_GRID_DIM and h <= MAX_GRID_DIM, (
        f"attn_backend='te' supports at most {MAX_GRID_DIM} batch entries and "
        f"{MAX_GRID_DIM} heads per call, got batch={b}, heads={h}."
    )
