"""Autocast helpers for custom autograd Functions.

Autocast does not apply to ops inside a :class:`torch.autograd.Function`, so
these helpers apply its dtype policy explicitly.

* :func:`cast_to_autocast_dtype` casts down to the autocast dtype.
* :func:`promote_to_fp32_under_autocast` promotes up to fp32.
* :func:`resolve_matmul_dtype` gives the dtype a matmul produces.
"""

from __future__ import annotations

from typing import TypeVar, overload

import torch

_T0 = TypeVar("_T0", bound=torch.Tensor | None)
_T1 = TypeVar("_T1", bound=torch.Tensor | None)
_T2 = TypeVar("_T2", bound=torch.Tensor | None)


def _reference_device_type(
    tensors: tuple[torch.Tensor | None, ...],
) -> str | None:
    """Device type of the first non-``None`` tensor, or ``None`` if there is none."""
    for t in tensors:
        if t is not None:
            return t.device.type
    return None


@overload
def cast_to_autocast_dtype(t0: _T0, /) -> tuple[_T0]: ...  # numpydoc ignore=GL08
@overload
def cast_to_autocast_dtype(
    t0: _T0, t1: _T1, /
) -> tuple[_T0, _T1]: ...  # numpydoc ignore=GL08
@overload
def cast_to_autocast_dtype(
    t0: _T0, t1: _T1, t2: _T2, /
) -> tuple[_T0, _T1, _T2]: ...  # numpydoc ignore=GL08


def cast_to_autocast_dtype(
    *tensors: torch.Tensor | None,
) -> tuple[torch.Tensor | None, ...]:
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
def promote_to_fp32_under_autocast(
    t0: _T0, /
) -> tuple[_T0]: ...  # numpydoc ignore=GL08
@overload
def promote_to_fp32_under_autocast(
    t0: _T0, t1: _T1, /
) -> tuple[_T0, _T1]: ...  # numpydoc ignore=GL08
@overload
def promote_to_fp32_under_autocast(  # numpydoc ignore=GL08
    t0: _T0, t1: _T1, t2: _T2, /
) -> tuple[_T0, _T1, _T2]: ...


def promote_to_fp32_under_autocast(
    *tensors: torch.Tensor | None,
) -> tuple[torch.Tensor | None, ...]:
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


def resolve_matmul_dtype(device: torch.device, *operands: torch.Tensor) -> torch.dtype:
    """Return the dtype a ``matmul`` over ``operands`` produces on ``device``.

    This is the autocast dtype if autocast is active, otherwise the promoted
    dtype of the operands.

    Parameters
    ----------
    device : torch.device
        Device the matmul runs on.
    *operands : torch.Tensor
        Matmul operands.

    Returns
    -------
    torch.dtype
        Result dtype.
    """
    if torch.is_autocast_enabled(device.type):
        return torch.get_autocast_dtype(device.type)
    dtype = operands[0].dtype
    for operand in operands[1:]:
        dtype = torch.promote_types(dtype, operand.dtype)
    return dtype
