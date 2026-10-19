"""Distributed activation layers."""

from jigsaw._core.public import set_module as _set_module

from ._activation import DistributedSoftmax
from ._functional import distributed_argmax, distributed_log_softmax

__all__ = ["DistributedSoftmax", "distributed_log_softmax", "distributed_argmax"]

_set_module(globals(), __all__)
