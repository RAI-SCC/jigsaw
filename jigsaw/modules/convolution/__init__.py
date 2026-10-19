"""Distributed convolution and transpose convolution."""

from jigsaw._core.public import set_module as _set_module

from .conv import (
    ConvNdConfig,
    DistributedConv1d,
    DistributedConv2d,
    DistributedConv3d,
)
from .conv_transpose import (
    ConvTransposeNdConfig,
    DistributedConvTranspose1d,
    DistributedConvTranspose2d,
    DistributedConvTranspose3d,
)

__all__ = [
    "DistributedConv1d",
    "DistributedConv2d",
    "DistributedConv3d",
    "DistributedConvTranspose1d",
    "DistributedConvTranspose2d",
    "DistributedConvTranspose3d",
    "ConvNdConfig",
    "ConvTransposeNdConfig",
]

_set_module(globals(), __all__)
