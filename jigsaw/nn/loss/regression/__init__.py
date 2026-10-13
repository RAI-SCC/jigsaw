"""Distributed regression losses."""

from .functional import (
    distributed_l1_loss,
    distributed_mse_loss,
    distributed_smooth_l1_loss,
)
from .loss import DistributedL1Loss, DistributedMSELoss, DistributedSmoothL1Loss

__all__ = [
    "DistributedMSELoss",
    "DistributedL1Loss",
    "DistributedSmoothL1Loss",
    "distributed_mse_loss",
    "distributed_l1_loss",
    "distributed_smooth_l1_loss",
]
