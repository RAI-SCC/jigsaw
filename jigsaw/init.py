"""Shard-aware versions of :mod:`torch.nn.init`.

The initializers take the shape of the unsharded parameter (``global_shape``)
or explicit fans, so that the fans do not depend on the local shard size.
Without these arguments they match their :mod:`torch.nn.init` namesakes.
"""

import math
from collections.abc import Sequence
from typing import Any, Optional, Union, cast

import torch
import torch.nn.init as _torch_init

calculate_gain = _torch_init.calculate_gain

# Per-element initializers: the value distribution does not depend on the tensor
# shape, so filling a shard is identical to filling the whole parameter -- these
# are re-exported verbatim from ``torch.nn.init`` (each already wraps the write
# in ``no_grad``).
constant_ = _torch_init.constant_
normal_ = _torch_init.normal_
ones_ = _torch_init.ones_
trunc_normal_ = _torch_init.trunc_normal_
uniform_ = _torch_init.uniform_
zeros_ = _torch_init.zeros_

__all__ = [
    "calculate_gain",
    "constant_",
    "normal_",
    "ones_",
    "trunc_normal_",
    "uniform_",
    "zeros_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "xavier_uniform_",
    "xavier_normal_",
]

ShapeLike = Union[torch.Tensor, torch.Size, Sequence[int]]


def _as_shape(shape: ShapeLike) -> tuple[int, ...]:
    """Convert a tensor, ``Size`` or int sequence to a shape tuple."""
    if isinstance(shape, torch.Tensor):
        return tuple(shape.shape)
    return tuple(int(s) for s in shape)


def _calculate_fan_in_and_fan_out(shape: ShapeLike) -> tuple[int, int]:
    """Compute ``(fan_in, fan_out)`` of a weight as ``torch.nn.init`` does.

    Parameters
    ----------
    shape : torch.Tensor | torch.Size | Sequence[int]
        Weight or its shape.

    Returns
    -------
    tuple[int, int]
        ``(fan_in, fan_out)``.
    """
    dims = _as_shape(shape)
    if len(dims) < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than "
            "2 dimensions"
        )
    num_input_fmaps = dims[1]
    num_output_fmaps = dims[0]
    receptive_field_size = 1
    for s in dims[2:]:
        receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _resolve_fans(
    tensor: torch.Tensor,
    global_shape: Optional[ShapeLike],
    fan_in: Optional[int],
    fan_out: Optional[int],
) -> tuple[int, int]:
    """Resolve the fans from explicit arguments, ``global_shape`` or ``tensor``, in that order."""
    if global_shape is not None:
        gi, go = _calculate_fan_in_and_fan_out(global_shape)
    else:
        gi, go = _calculate_fan_in_and_fan_out(tensor)
    if fan_in is not None:
        gi = fan_in
    if fan_out is not None:
        go = fan_out
    return gi, go


def _calculate_correct_fan(
    tensor: torch.Tensor,
    mode: str,
    global_shape: Optional[ShapeLike] = None,
    fan_in: Optional[int] = None,
    fan_out: Optional[int] = None,
) -> int:
    """Return the fan selected by ``mode``."""
    mode = mode.lower()
    valid_modes = ("fan_in", "fan_out")
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")
    gi, go = _resolve_fans(tensor, global_shape, fan_in, fan_out)
    return gi if mode == "fan_in" else go


def kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: Optional[torch.Generator] = None,
    *,
    global_shape: Optional[ShapeLike] = None,
    fan_in: Optional[int] = None,
    fan_out: Optional[int] = None,
) -> torch.Tensor:
    """Fill a tensor with the Kaiming-uniform distribution.

    Parameters
    ----------
    tensor : torch.Tensor
        Local tensor (shard) to fill in place.
    a : float
        Negative slope of the rectifier used after this layer.
    mode : {'fan_in', 'fan_out'}
        Fan that preserves the variance in the forward or backward pass.
    nonlinearity : str
        Nonlinearity name for :func:`torch.nn.init.calculate_gain`.
    generator : torch.Generator, optional
        RNG to draw from.
    global_shape : torch.Tensor | torch.Size | Sequence[int], optional
        Shape of the unsharded weight. The fans are computed from it instead of
        from ``tensor``.
    fan_in, fan_out : int, optional
        Explicit fans, overriding ``global_shape`` and ``tensor``.

    Returns
    -------
    torch.Tensor
        ``tensor``, filled in place.
    """
    if tensor.numel() == 0:
        return tensor
    fan = _calculate_correct_fan(tensor, mode, global_shape, fan_in, fan_out)
    gain = calculate_gain(cast(Any, nonlinearity), a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: Optional[torch.Generator] = None,
    *,
    global_shape: Optional[ShapeLike] = None,
    fan_in: Optional[int] = None,
    fan_out: Optional[int] = None,
) -> torch.Tensor:
    """Fill a tensor with the Kaiming-normal distribution.

    Parameters
    ----------
    tensor : torch.Tensor
        Local tensor (shard) to fill in place.
    a : float
        Negative slope of the rectifier used after this layer.
    mode : {'fan_in', 'fan_out'}
        Fan that preserves the variance in the forward or backward pass.
    nonlinearity : str
        Nonlinearity name for :func:`torch.nn.init.calculate_gain`.
    generator : torch.Generator, optional
        RNG to draw from.
    global_shape : torch.Tensor | torch.Size | Sequence[int], optional
        Shape of the unsharded weight. The fans are computed from it instead of
        from ``tensor``.
    fan_in, fan_out : int, optional
        Explicit fans, overriding ``global_shape`` and ``tensor``.

    Returns
    -------
    torch.Tensor
        ``tensor``, filled in place.
    """
    if tensor.numel() == 0:
        return tensor
    fan = _calculate_correct_fan(tensor, mode, global_shape, fan_in, fan_out)
    gain = calculate_gain(cast(Any, nonlinearity), a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0.0, std, generator=generator)


def xavier_uniform_(
    tensor: torch.Tensor,
    gain: float = 1.0,
    generator: Optional[torch.Generator] = None,
    *,
    global_shape: Optional[ShapeLike] = None,
    fan_in: Optional[int] = None,
    fan_out: Optional[int] = None,
) -> torch.Tensor:
    """Fill a tensor with the Xavier-uniform distribution.

    Parameters
    ----------
    tensor : torch.Tensor
        Local tensor (shard) to fill in place.
    gain : float
        Scaling factor.
    generator : torch.Generator, optional
        RNG to draw from.
    global_shape : torch.Tensor | torch.Size | Sequence[int], optional
        Shape of the unsharded weight. The fans are computed from it instead of
        from ``tensor``.
    fan_in, fan_out : int, optional
        Explicit fans, overriding ``global_shape`` and ``tensor``.

    Returns
    -------
    torch.Tensor
        ``tensor``, filled in place.
    """
    if tensor.numel() == 0:
        return tensor
    fi, fo = _resolve_fans(tensor, global_shape, fan_in, fan_out)
    std = gain * math.sqrt(2.0 / float(fi + fo))
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


def xavier_normal_(
    tensor: torch.Tensor,
    gain: float = 1.0,
    generator: Optional[torch.Generator] = None,
    *,
    global_shape: Optional[ShapeLike] = None,
    fan_in: Optional[int] = None,
    fan_out: Optional[int] = None,
) -> torch.Tensor:
    """Fill a tensor with the Xavier-normal distribution.

    Parameters
    ----------
    tensor : torch.Tensor
        Local tensor (shard) to fill in place.
    gain : float
        Scaling factor.
    generator : torch.Generator, optional
        RNG to draw from.
    global_shape : torch.Tensor | torch.Size | Sequence[int], optional
        Shape of the unsharded weight. The fans are computed from it instead of
        from ``tensor``.
    fan_in, fan_out : int, optional
        Explicit fans, overriding ``global_shape`` and ``tensor``.

    Returns
    -------
    torch.Tensor
        ``tensor``, filled in place.
    """
    if tensor.numel() == 0:
        return tensor
    fi, fo = _resolve_fans(tensor, global_shape, fan_in, fan_out)
    std = gain * math.sqrt(2.0 / float(fi + fo))
    with torch.no_grad():
        return tensor.normal_(0.0, std, generator=generator)
