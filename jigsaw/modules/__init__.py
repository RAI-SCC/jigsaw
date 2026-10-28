"""Distributed jigsaw layers, the counterpart of :mod:`torch.nn`.

Every layer, loss and metric module is exported here
(``from jigsaw.modules import DistributedLinear``), together with the config
classes that tune them and :class:`CommOp`, the communication choice shared by
the configs. The functional forms are in :mod:`jigsaw.modules.functional`.
Other enums and helpers stay in their subpackage
(``jigsaw.modules.attention.AttnBackend``).
"""

from jigsaw._core.comm import CommOp
from jigsaw._core.public import set_module as _set_module

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
from .attention import Attention, DistributedAttention, RingAttentionConfig
from .convolution import (
    ConvNdConfig,
    ConvTransposeNdConfig,
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
    LinearConfig,
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
from .roll import DistributedRoll, Roll, RollConfig

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
    # Configs
    "LinearConfig",
    "ConvNdConfig",
    "ConvTransposeNdConfig",
    "RingAttentionConfig",
    "RollConfig",
    # Enums
    "CommOp",
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

_set_module(globals(), __all__)
