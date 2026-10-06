"""Autocast helpers for custom autograd Functions.

Autocast does not apply to ops inside a :class:`torch.autograd.Function`, so
these helpers apply its dtype policy explicitly.

* :func:`cast_to_autocast_dtype` casts down to the autocast dtype.
* :func:`promote_to_fp32_under_autocast` promotes up to fp32.
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
    """Cast tensors to the active autocast dtype.

    ``None`` entries are preserved. Tensors pass through unchanged when autocast
    is disabled.

    Parameters
    ----------
    *tensors : torch.Tensor | None
        Tensors to convert.

    Returns
    -------
    tuple[torch.Tensor | None, ...]
        Converted tensors.
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
    """Promote tensors to at least fp32 while autocast is active.

    fp16 and bf16 tensors are cast to fp32, wider dtypes and ``None`` are kept.
    Tensors pass through unchanged when autocast is disabled.

    Parameters
    ----------
    *tensors : torch.Tensor | None
        Tensors to convert.

    Returns
    -------
    tuple[torch.Tensor | None, ...]
        Converted tensors.
    """
    device_type = _reference_device_type(tensors)
    if device_type is None or not torch.is_autocast_enabled(device_type):
        return tensors
    return tuple(
        None if t is None else t.to(torch.promote_types(t.dtype, torch.float32))
        for t in tensors
    )
