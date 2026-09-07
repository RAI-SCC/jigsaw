"""Distributed evaluation metrics.

* :mod:`jigsaw.modules.metrics.segmentation` -- distributed confusion matrix and
  the semantic-segmentation metrics (mIoU, Dice, pixel accuracy, ...) derived
  from it.
"""

from .segmentation import (
    DistributedSegmentationMetrics as DistributedSegmentationMetrics,
)
from .segmentation import (
    confusion_to_metrics as confusion_to_metrics,
)
from .segmentation import (
    distributed_confusion_matrix as distributed_confusion_matrix,
)
