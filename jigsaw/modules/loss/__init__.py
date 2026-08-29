"""Expose distributed loss functions to the public interface.

The implementation is split by problem type:

* :mod:`jigsaw.modules.loss.common` -- shared collectives / autograd primitives;
* :mod:`jigsaw.modules.loss.functional_classification` and
  :mod:`jigsaw.modules.loss.loss_classification` -- cross-entropy, NLL, BCE;
* :mod:`jigsaw.modules.loss.functional_regression` and
  :mod:`jigsaw.modules.loss.loss_regression` -- L2 (MSE), L1, smooth-L1;
* :mod:`jigsaw.modules.loss.metrics_segmentation` -- distributed confusion
  matrix and the segmentation metrics derived from it.
"""

from .common import (
    distributed_argmax as distributed_argmax,
)
from .common import (
    distributed_log_softmax as distributed_log_softmax,
)
from .common import (
    redundant_all_reduce as redundant_all_reduce,
)
from .functional_classification import (
    distributed_bce_with_logits as distributed_bce_with_logits,
)
from .functional_classification import (
    distributed_binary_cross_entropy as distributed_binary_cross_entropy,
)
from .functional_classification import (
    distributed_cross_entropy as distributed_cross_entropy,
)
from .functional_classification import (
    distributed_nll_loss as distributed_nll_loss,
)
from .functional_regression import (
    distributed_l1_loss as distributed_l1_loss,
)
from .functional_regression import (
    distributed_mse_loss as distributed_mse_loss,
)
from .functional_regression import (
    distributed_smooth_l1_loss as distributed_smooth_l1_loss,
)
from .loss_classification import DistributedBCELoss as DistributedBCELoss
from .loss_classification import (
    DistributedBCEWithLogitsLoss as DistributedBCEWithLogitsLoss,
)
from .loss_classification import (
    DistributedCrossEntropyLoss as DistributedCrossEntropyLoss,
)
from .loss_classification import DistributedNLLLoss as DistributedNLLLoss
from .loss_regression import DistributedL1Loss as DistributedL1Loss
from .loss_regression import DistributedMSELoss as DistributedMSELoss
from .loss_regression import DistributedSmoothL1Loss as DistributedSmoothL1Loss
from .metrics_segmentation import (
    DistributedSegmentationMetrics as DistributedSegmentationMetrics,
)
from .metrics_segmentation import (
    confusion_to_metrics as confusion_to_metrics,
)
from .metrics_segmentation import (
    distributed_confusion_matrix as distributed_confusion_matrix,
)
