"""Distributed activation layers."""

from ._activation import DistributedSoftmax
from ._functional import distributed_argmax, distributed_log_softmax

__all__ = ["DistributedSoftmax", "distributed_log_softmax", "distributed_argmax"]
