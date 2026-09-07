"""Distributed classification losses: cross-entropy, NLL and binary cross-entropy.

* :mod:`~jigsaw.modules.loss.classification.functional` -- functional forms;
* :mod:`~jigsaw.modules.loss.classification.loss` -- ``torch.nn`` module wrappers.
"""

from .functional import (
    distributed_bce_with_logits as distributed_bce_with_logits,
)
from .functional import (
    distributed_binary_cross_entropy as distributed_binary_cross_entropy,
)
from .functional import (
    distributed_cross_entropy as distributed_cross_entropy,
)
from .functional import (
    distributed_nll_loss as distributed_nll_loss,
)
from .loss import DistributedBCELoss as DistributedBCELoss
from .loss import DistributedBCEWithLogitsLoss as DistributedBCEWithLogitsLoss
from .loss import DistributedCrossEntropyLoss as DistributedCrossEntropyLoss
from .loss import DistributedNLLLoss as DistributedNLLLoss
