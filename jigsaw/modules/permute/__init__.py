"""Distributed permutation."""

from jigsaw._core.public import set_module as _set_module

from .distributed_permute import distributed_permute

__all__ = ["distributed_permute"]

_set_module(globals(), __all__)
