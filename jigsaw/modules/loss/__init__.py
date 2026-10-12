"""Distributed classification and regression losses."""

from .classification import (
    DistributedBCELoss,
    DistributedBCEWithLogitsLoss,
    DistributedCrossEntropyLoss,
    DistributedNLLLoss,
    distributed_bce_with_logits,
    distributed_binary_cross_entropy,
    distributed_cross_entropy,
    distributed_nll_loss,
)
from .common import distributed_argmax, distributed_log_softmax, redundant_all_reduce
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
    "distributed_bce_with_logits",
    "distributed_mse_loss",
    "distributed_l1_loss",
    "distributed_smooth_l1_loss",
    "distributed_argmax",
    "distributed_log_softmax",
    "redundant_all_reduce",
]
