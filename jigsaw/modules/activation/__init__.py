"""Distributed activation layers."""

from .activation import DistributedSoftmax
from .functional import distributed_argmax, distributed_log_softmax

__all__ = ["DistributedSoftmax", "distributed_log_softmax", "distributed_argmax"]
