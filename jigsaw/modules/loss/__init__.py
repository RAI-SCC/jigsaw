"""Distributed classification and regression losses."""

from .classification import (
    DistributedBCELoss,
    DistributedBCEWithLogitsLoss,
    DistributedCrossEntropyLoss,
    DistributedNLLLoss,
    distributed_binary_cross_entropy,
    distributed_binary_cross_entropy_with_logits,
    distributed_cross_entropy,
    distributed_nll_loss,
)
from .common import redundant_all_reduce
from .regression import (
    DistributedL1Loss,
    DistributedMSELoss,
    DistributedSmoothL1Loss,
    distributed_l1_loss,
    distributed_mse_loss,
    distributed_smooth_l1_loss,
)

__all__ = [
    "DistributedNLLLoss",
    "DistributedCrossEntropyLoss",
    "DistributedBCELoss",
    "DistributedBCEWithLogitsLoss",
    "DistributedMSELoss",
    "DistributedL1Loss",
    "DistributedSmoothL1Loss",
    "distributed_nll_loss",
    "distributed_cross_entropy",
    "distributed_binary_cross_entropy",
    "distributed_binary_cross_entropy_with_logits",
    "distributed_mse_loss",
    "distributed_l1_loss",
    "distributed_smooth_l1_loss",
    "redundant_all_reduce",
]
