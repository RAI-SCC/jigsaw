"""Distributed regression losses: L2 (MSE), L1 and smooth-L1 (Huber).

* :mod:`~jigsaw.modules.loss.regression.functional` -- functional forms;
* :mod:`~jigsaw.modules.loss.regression.loss` -- ``torch.nn`` module wrappers.
"""

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
