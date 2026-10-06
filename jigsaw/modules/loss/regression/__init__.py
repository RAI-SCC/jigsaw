"""Distributed regression losses."""

from .functional import (
    distributed_l1_loss as distributed_l1_loss,
)
from .functional import (
    distributed_mse_loss as distributed_mse_loss,
)
from .functional import (
    distributed_smooth_l1_loss as distributed_smooth_l1_loss,
)
from .loss import DistributedL1Loss as DistributedL1Loss
from .loss import DistributedMSELoss as DistributedMSELoss
from .loss import DistributedSmoothL1Loss as DistributedSmoothL1Loss
