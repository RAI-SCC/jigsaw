"""Distributed jigsaw layers, the counterpart of :mod:`torch.nn`.

Every layer and loss is exported here (``jigsaw.nn.DistributedLinear``), the
functional forms from :mod:`jigsaw.nn.functional`. Configuration classes and
enums stay in their subpackage (``jigsaw.nn.attention.RingAttentionConfig``).
"""

from . import (
    activation,
    attention,
    convolution,
    functional,
    linear,
    loss,
    metrics,
    normalization,
    permute,
    roll,
)
from .activation import DistributedSoftmax
from .attention import Attention, DistributedAttention
from .convolution import (
    DistributedConv1d,
    DistributedConv2d,
    DistributedConv3d,
    DistributedConvTranspose1d,
    DistributedConvTranspose2d,
    DistributedConvTranspose3d,
)
from .linear import (
    DistributedLinear,
    DistributedMatMulXTW,
    DistributedMatMulXW,
    DistributedMatMulXWT,
    MatMulXTW,
    MatMulXW,
    MatMulXWT,
)
from .loss import (
    DistributedBCELoss,
    DistributedBCEWithLogitsLoss,
    DistributedCrossEntropyLoss,
    DistributedL1Loss,
    DistributedMSELoss,
    DistributedNLLLoss,
    DistributedSmoothL1Loss,
)
from .metrics import DistributedSegmentationMetrics
from .normalization import DistributedGroupNorm, DistributedLayerNorm
from .roll import DistributedRoll, Roll

__all__ = [
    # Layers
    "DistributedLinear",
    "DistributedMatMulXWT",
    "DistributedMatMulXW",
    "DistributedMatMulXTW",
    "MatMulXWT",
    "MatMulXW",
    "MatMulXTW",
    "DistributedConv1d",
    "DistributedConv2d",
    "DistributedConv3d",
    "DistributedConvTranspose1d",
    "DistributedConvTranspose2d",
    "DistributedConvTranspose3d",
    "Attention",
    "DistributedAttention",
    "DistributedLayerNorm",
    "DistributedGroupNorm",
    "DistributedSoftmax",
    "Roll",
    "DistributedRoll",
    # Losses
    "DistributedNLLLoss",
    "DistributedCrossEntropyLoss",
    "DistributedBCELoss",
    "DistributedBCEWithLogitsLoss",
    "DistributedMSELoss",
    "DistributedL1Loss",
    "DistributedSmoothL1Loss",
    # Metrics
    "DistributedSegmentationMetrics",
    # Subpackages
    "activation",
    "attention",
    "convolution",
    "functional",
    "linear",
    "loss",
    "metrics",
    "normalization",
    "permute",
    "roll",
]
