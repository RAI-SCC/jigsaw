"""Expose distributed loss functions to the public interface.

The implementation is split by problem type:

* :mod:`jigsaw.modules.loss.common` -- shared collectives / autograd primitives;
* :mod:`jigsaw.modules.loss.classification` -- cross-entropy, NLL, BCE
  (``functional`` forms and ``torch.nn`` module wrappers);
* :mod:`jigsaw.modules.loss.regression` -- L2 (MSE), L1, smooth-L1
  (``functional`` forms and ``torch.nn`` module wrappers).

Distributed segmentation metrics live in :mod:`jigsaw.modules.metrics`.
"""

from .classification import (
    DistributedBCELoss as DistributedBCELoss,
)
from .classification import (
    DistributedBCEWithLogitsLoss as DistributedBCEWithLogitsLoss,
)
from .classification import (
    DistributedCrossEntropyLoss as DistributedCrossEntropyLoss,
)
from .classification import (
    DistributedNLLLoss as DistributedNLLLoss,
)
from .classification import (
    distributed_bce_with_logits as distributed_bce_with_logits,
)
from .classification import (
    distributed_binary_cross_entropy as distributed_binary_cross_entropy,
)
from .classification import (
    distributed_cross_entropy as distributed_cross_entropy,
)
from .classification import (
    distributed_nll_loss as distributed_nll_loss,
)
from .common import (
    distributed_argmax as distributed_argmax,
)
from .common import (
    distributed_log_softmax as distributed_log_softmax,
)
from .common import (
    redundant_all_reduce as redundant_all_reduce,
)
from .regression import (
    DistributedL1Loss as DistributedL1Loss,
)
from .regression import (
    DistributedMSELoss as DistributedMSELoss,
)
from .regression import (
    DistributedSmoothL1Loss as DistributedSmoothL1Loss,
)
from .regression import (
    distributed_l1_loss as distributed_l1_loss,
)
from .regression import (
    distributed_mse_loss as distributed_mse_loss,
)
from .regression import (
    distributed_smooth_l1_loss as distributed_smooth_l1_loss,
)
