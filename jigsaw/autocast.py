"""Autocast opt-in helpers shared by the distributed autograd Functions.

A :class:`torch.autograd.Function` is opaque to autocast's op-level casting
policy: once the distributed forward is wrapped in a custom Function, autocast
never sees the inner ``F.conv*`` / matmul / ``layer_norm`` call, so the Function
has to apply autocast's dtype rule itself. These two primitives are the two
directions of that rule:

* :func:`cast_to_autocast_dtype` -- cast *down* to the autocast dtype, for ops
  on autocast's lower-precision list (distributed conv and linear).
* :func:`promote_to_fp32_under_autocast` -- promote *up* to fp32, for ops on
  autocast's fp32 list (distributed LayerNorm and softmax, whose reductions are
  too precision sensitive for bf16/fp16).

Deliberately not here:
:func:`jigsaw.modules.attention.attention.attn_core_dtype` picks a
flash-kernel-eligible dtype even *outside* autocast, and
:func:`jigsaw.comm.resolve_matmul_dtype` predicts a matmul's output dtype
*before* it runs so a receive buffer can be sized -- both answer a different
question and stay module-local.
"""

from __future__ import annotations

from typing import Optional, TypeVar, overload

import torch

_T0 = TypeVar("_T0", bound=Optional[torch.Tensor])
_T1 = TypeVar("_T1", bound=Optional[torch.Tensor])
_T2 = TypeVar("_T2", bound=Optional[torch.Tensor])


def _reference_device_type(
    tensors: tuple[Optional[torch.Tensor], ...],
) -> Optional[str]:
    """Device type of the first non-``None`` tensor, or ``None`` if there is none."""
    for t in tensors:
        if t is not None:
            return t.device.type
    return None


@overload
def cast_to_autocast_dtype(t0: _T0, /) -> tuple[_T0]: ...
@overload
def cast_to_autocast_dtype(t0: _T0, t1: _T1, /) -> tuple[_T0, _T1]: ...
@overload
def cast_to_autocast_dtype(t0: _T0, t1: _T1, t2: _T2, /) -> tuple[_T0, _T1, _T2]: ...


def cast_to_autocast_dtype(
    *tensors: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], ...]:
    """Cast each tensor to the active autocast dtype; pass through if disabled.

    ``None`` entries are preserved (a Function's optional bias stays optional).
    Mirrors what autocast would do to a lower-precision op (conv, matmul) that a
    custom Function hides from it, so the whole distributed pipeline -- the
    channel-gather / halo buffers the kernels size from these tensors included --
    runs in the low-precision dtype and the output carries it. Backward
    reconciles the saved activation to ``weight.dtype`` (master precision) to
    match ``grad_output``.
    """
    device_type = _reference_device_type(tensors)
    if device_type is None or not torch.is_autocast_enabled(device_type):
        return tensors
    dtype = torch.get_autocast_dtype(device_type)
    return tuple(None if t is None else t.to(dtype) for t in tensors)


@overload
def promote_to_fp32_under_autocast(t0: _T0, /) -> tuple[_T0]: ...
@overload
def promote_to_fp32_under_autocast(t0: _T0, t1: _T1, /) -> tuple[_T0, _T1]: ...
@overload
def promote_to_fp32_under_autocast(
    t0: _T0, t1: _T1, t2: _T2, /
) -> tuple[_T0, _T1, _T2]: ...


def promote_to_fp32_under_autocast(
    *tensors: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], ...]:
    """Promote each tensor to (at least) fp32 while autocast is active.

    fp16/bf16 -> fp32; fp32/fp64 unchanged; ``None`` preserved. Mirrors how
    autocast keeps ``layer_norm`` / ``softmax`` in fp32 by op-level policy, and
    makes their output land in fp32 the way stock AMP does. Outside autocast the
    operands pass straight through.
    """
    device_type = _reference_device_type(tensors)
    if device_type is None or not torch.is_autocast_enabled(device_type):
        return tensors
    return tuple(
        None if t is None else t.to(torch.promote_types(t.dtype, torch.float32))
        for t in tensors
    )
