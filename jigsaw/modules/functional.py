"""Functional forms of the jigsaw layers, the counterpart of :mod:`torch.nn.functional`."""

from jigsaw._core.public import set_module as _set_module

from .activation import distributed_argmax, distributed_log_softmax
from .loss import (
    distributed_binary_cross_entropy,
    distributed_binary_cross_entropy_with_logits,
    distributed_cross_entropy,
    distributed_l1_loss,
    distributed_mse_loss,
    distributed_nll_loss,
    distributed_smooth_l1_loss,
    redundant_all_reduce,
)
from .metrics import confusion_to_metrics, distributed_confusion_matrix
from .permute import distributed_permute

__all__ = [
    # Activations
    "distributed_log_softmax",
    "distributed_argmax",
    # Losses
    "distributed_nll_loss",
    "distributed_cross_entropy",
    "distributed_binary_cross_entropy",
    "distributed_binary_cross_entropy_with_logits",
    "distributed_mse_loss",
    "distributed_l1_loss",
    "distributed_smooth_l1_loss",
    "redundant_all_reduce",
    # Metrics
    "distributed_confusion_matrix",
    "confusion_to_metrics",
    # Permutation
    "distributed_permute",
]

_set_module(globals(), __all__)
