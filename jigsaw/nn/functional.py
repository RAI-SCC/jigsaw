"""Functional forms of the jigsaw layers, the counterpart of :mod:`torch.nn.functional`."""

from .activation import distributed_argmax, distributed_log_softmax
from .attention._functional import hierarchical_ulysses_attention, ulysses_attention
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
from .permute import distributed_permute

__all__ = [
    "distributed_log_softmax",
    "distributed_argmax",
    "ulysses_attention",
    "hierarchical_ulysses_attention",
    "distributed_nll_loss",
    "distributed_cross_entropy",
    "distributed_binary_cross_entropy",
    "distributed_binary_cross_entropy_with_logits",
    "distributed_mse_loss",
    "distributed_l1_loss",
    "distributed_smooth_l1_loss",
    "redundant_all_reduce",
    "distributed_permute",
]
