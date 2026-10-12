"""Distributed classification losses."""

from .functional import (
    distributed_bce_with_logits,
    distributed_binary_cross_entropy,
    distributed_cross_entropy,
    distributed_nll_loss,
)
from .loss import (
    DistributedBCELoss,
    DistributedBCEWithLogitsLoss,
    DistributedCrossEntropyLoss,
    DistributedNLLLoss,
)

__all__ = [
    "DistributedNLLLoss",
    "DistributedCrossEntropyLoss",
    "DistributedBCELoss",
    "DistributedBCEWithLogitsLoss",
    "distributed_nll_loss",
    "distributed_cross_entropy",
    "distributed_binary_cross_entropy",
    "distributed_bce_with_logits",
]
