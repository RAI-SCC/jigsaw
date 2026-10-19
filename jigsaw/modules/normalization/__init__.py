"""Distributed normalization layers."""

from jigsaw._core.public import set_module as _set_module

from .group_norm import DistributedGroupNorm
from .layer_norm import DistributedLayerNorm

__all__ = ["DistributedGroupNorm", "DistributedLayerNorm"]

_set_module(globals(), __all__)
