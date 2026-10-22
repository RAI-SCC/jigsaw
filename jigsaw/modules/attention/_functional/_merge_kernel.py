"""Fused Triton kernel for the ring-attention online-softmax merge.

Computes, in place on ``o1``::

    w = sigmoid(l2 - l1)
    o1 = o1 + w * (o2 - o1)
    l_new = logaddexp(l1, l2)

in one launch instead of four eager kernels.
:func:`fused_online_softmax_merge_` returns ``None`` whenever Triton or the
inputs are unsupported, and the caller falls back to the eager merge.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - triton missing / broken toolchain
    _TRITON_AVAILABLE = False

#: Disabled after the first failed launch, or by ``JIGSAW_DISABLE_TRITON_MERGE=1``.
_KERNEL_USABLE = _TRITON_AVAILABLE and os.environ.get(
    "JIGSAW_DISABLE_TRITON_MERGE", ""
).strip().lower() not in ("1", "true", "yes")


if _TRITON_AVAILABLE:

    @triton.jit
    def _merge_kernel(  # type: ignore[no-untyped-def]
        o1_ptr,  # (n_rows, D), mutated in place; any float dtype
        o2_ptr,  # (n_rows, D), this step's partial output
        l1_ptr,  # (n_rows,) fp32, running log-sum-exp, contiguous (b, h, n)
        l2_ptr,  # (n_rows,) fp32, this step's log-sum-exp, contiguous (b, h, n)
        lnew_ptr,  # (n_rows,) fp32, output log-sum-exp
        n_rows,
        N,
        H,
        D,
        IS_BSHD: tl.constexpr,  # o rows run (b, n, h) rather than (b, h, n)
        BLOCK_M: tl.constexpr,  # o rows per program
        BLOCK_D: tl.constexpr,  # >= D, power of two
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < n_rows

        # Map each flat ``o`` row to its (b, h, n) log-sum-exp slot. bhsd rows
        # already run (b, h, n) so the row index *is* the LSE index; bshd rows
        # run (b, n, h) and have to be unpacked.
        if IS_BSHD:
            b = rows // (N * H)
            rem = rows % (N * H)
            n = rem // H
            h = rem % H
            l_off = (b * H + h) * N + n
        else:
            l_off = rows

        l1 = tl.load(l1_ptr + l_off, mask=rmask)
        l2 = tl.load(l2_ptr + l_off, mask=rmask)

        # w = sigmoid(l2 - l1) = exp(l2 - logaddexp(l1, l2)); at the seed state
        # l1 = -inf so w = sigmoid(+inf) = 1 and o1 := o2, l_new := l2.
        w = tl.sigmoid(l2 - l1)
        mx = tl.maximum(l1, l2)
        l_new = mx + tl.log(1.0 + tl.exp(-tl.abs(l1 - l2)))
        tl.store(lnew_ptr + l_off, l_new, mask=rmask)

        cols = tl.arange(0, BLOCK_D)
        p = rows[:, None] * D + cols[None, :]
        pmask = rmask[:, None] & (cols[None, :] < D)
        a = tl.load(o1_ptr + p, mask=pmask, other=0.0).to(tl.float32)
        b_ = tl.load(o2_ptr + p, mask=pmask, other=0.0).to(tl.float32)
        tl.store(o1_ptr + p, a + w[:, None] * (b_ - a), mask=pmask)


_SUPPORTED_O_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def fused_online_softmax_merge_(
    o1: torch.Tensor,
    l1: torch.Tensor,
    o2: torch.Tensor,
    l2: torch.Tensor,
    layout: str,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Triton fast path for :func:`.forward._online_softmax_merge_`.

    Mutates ``o1`` in place; ``l_new`` is a fresh fp32 ``(b, h, n)`` tensor.

    Parameters
    ----------
    o1 : torch.Tensor
        Running output, updated in place.
    l1 : torch.Tensor
        Running log-sum-exp.
    o2 : torch.Tensor
        Output of the new block.
    l2 : torch.Tensor
        Log-sum-exp of the new block.
    layout : str
        Layout of ``o1`` / ``o2``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] or None
        ``(o1, l_new)``, or ``None`` if the kernel does not support the inputs.
    """
    global _KERNEL_USABLE
    if not _KERNEL_USABLE or not o1.is_cuda:
        return None
    if layout not in ("bhsd", "bshd"):
        return None
    # ``o1`` is the ring's own fp32/-64 accumulator; ``o2`` is the core's output
    # in the model dtype. Both are freshly produced and packed in ``layout``.
    if o1.dtype not in _SUPPORTED_O_DTYPES or o2.dtype not in _SUPPORTED_O_DTYPES:
        return None
    if o1.shape != o2.shape or not o1.is_contiguous() or not o2.is_contiguous():
        return None
    if l1.shape != l2.shape or l1.dim() != 3:
        return None
    if not l1.is_contiguous() or not l2.is_contiguous():
        return None

    is_bshd = layout == "bshd"
    b, s1, s2, d = o1.shape
    n = s1 if is_bshd else s2
    h = s2 if is_bshd else s1
    if l1.shape != (b, h, n):
        return None

    l_new = torch.empty_like(l1, dtype=torch.float32)
    n_rows = b * h * n
    o1f = o1.view(n_rows, d)
    o2f = o2.view(n_rows, d)
    block_d = triton.next_power_of_2(d)
    block_m = max(1, 2048 // block_d)  # ~2K elements/program
    grid = (triton.cdiv(n_rows, block_m),)
    try:
        # num_warps is a launcher meta-arg; constexprs take plain Python values.
        _merge_kernel[grid](  # type: ignore[call-arg]
            o1f,
            o2f,
            l1,
            l2.to(torch.float32) if l2.dtype != torch.float32 else l2,
            l_new,
            n_rows,
            n,
            h,
            d,
            IS_BSHD=is_bshd,  # type: ignore[arg-type]
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            num_warps=4,
        )
    except Exception as exc:  # broken runtime-build toolchain, etc.
        _KERNEL_USABLE = False
        warnings.warn(
            f"fused online-softmax merge kernel failed ({exc!r}); falling back "
            "to the eager merge for the rest of this run.",
            stacklevel=2,
        )
        return None
    return o1, l_new
