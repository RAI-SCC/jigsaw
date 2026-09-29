"""Shard-aware re-implementation of :mod:`torch.nn.init`.

Every distributed jigsaw layer holds only a *slice* of its weight: the linear
layer shards both weight axes over the row/column groups, the convolutions shard
the output-channel axis over the channel group (and replicate over the spatial
groups). ``torch.nn.init`` infers the fan from ``tensor.shape``, so calling it on
such a slice keys the distribution off the local shard size and produces a layer
whose statistics drift from the equivalent non-distributed module.

The initializers here take an explicit ``global_shape`` (or a raw ``fan_in`` /
``fan_out``) describing the *whole*, unsharded parameter. The fan is computed
from that; the actual draw still fills only the local tensor. Because the
per-element distribution is what the fan controls, the concatenation of every
rank's shard is then distributed exactly as the non-distributed parameter would
be.

With no ``global_shape`` / ``fan_in`` / ``fan_out`` argument every function is a
drop-in for its :mod:`torch.nn.init` namesake, so a model can call these
unconditionally whether or not it ends up sharded.

A jigsaw module exposes ``global_weight_shape`` / ``global_bias_shape`` and a
``reset_parameters`` method; override that method (or call these helpers with
``global_shape=module.global_weight_shape``) to change a layer's init the way
``torch.nn.Module.reset_parameters`` is overridden for ``torch.nn`` layers.
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
    """Normalize a tensor / ``Size`` / int-sequence to a plain shape tuple."""
    if isinstance(shape, torch.Tensor):
        return tuple(shape.shape)
    return tuple(int(s) for s in shape)


def _calculate_fan_in_and_fan_out(shape: ShapeLike) -> tuple[int, int]:
    """
    Fan-in / fan-out of a weight, computed exactly as ``torch.nn.init`` does.

    Parameters
    ----------
    shape : torch.Tensor | torch.Size | Sequence[int]
        The weight, or its shape. Pass the *global* (unsharded) shape to get the
        fan of the whole parameter rather than of a local shard.

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
    """Pick the fans from (in order of precedence) explicit args, ``global_shape``, ``tensor``."""
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
    """``fan_in`` or ``fan_out`` per ``mode`` ('fan_in' / 'fan_out'), fan source resolved as in :func:`_resolve_fans`."""
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
    """
    Kaiming-uniform init, drawing from a distribution keyed off the *global* fan.

    Identical to :func:`torch.nn.init.kaiming_uniform_` except for the three
    keyword-only arguments. The tensor is filled in place; only its own elements
    are written, so on a shard the result is the matching slice of the global
    Kaiming-uniform draw.

    Parameters
    ----------
    tensor : torch.Tensor
        Local tensor (shard) to fill in place.
    a : float
        Negative slope of the rectifier used after this layer (``leaky_relu``).
    mode : {'fan_in', 'fan_out'}
        Whether to preserve the variance in the forward (``fan_in``) or backward
        (``fan_out``) pass.
    nonlinearity : str
        Nonlinearity name for :func:`torch.nn.init.calculate_gain`.
    generator : torch.Generator, optional
        RNG to draw from.
    global_shape : torch.Tensor | torch.Size | Sequence[int], optional
        Shape of the whole, unsharded weight. When given, the fan is computed
        from it instead of from ``tensor``.
    fan_in, fan_out : int, optional
        Explicit fans, overriding both ``global_shape`` and ``tensor``.

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
    """Kaiming-normal counterpart of :func:`kaiming_uniform_`; see it for the extra arguments."""
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
    """
    Xavier/Glorot-uniform init keyed off the *global* fans.

    Identical to :func:`torch.nn.init.xavier_uniform_` bar the keyword-only
    ``global_shape`` / ``fan_in`` / ``fan_out`` (see :func:`kaiming_uniform_`).
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
    """Xavier-normal counterpart of :func:`xavier_uniform_`."""
    if tensor.numel() == 0:
        return tensor
    fi, fo = _resolve_fans(tensor, global_shape, fan_in, fan_out)
    std = gain * math.sqrt(2.0 / float(fi + fo))
    with torch.no_grad():
        return tensor.normal_(0.0, std, generator=generator)
