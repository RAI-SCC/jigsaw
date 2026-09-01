"""Back-compat shim: the distributed 2-D transpose convolution now lives in ``jigsaw.modules.convolution.conv_transpose``."""

from jigsaw.modules.convolution.conv_transpose.base import _ntuple
from jigsaw.modules.convolution.conv_transpose.conv import (
    DistributedConv2DTranspose as DistributedConv2DTranspose,
)

# historical name for the 2-tuple arg normaliser.
_pair = _ntuple(2)
