"""Back-compat shim: the distributed 1-D transpose convolution now lives in ``jigsaw.modules.convolution.conv_transpose``."""

from jigsaw.modules.convolution.conv_transpose.base import _ntuple
from jigsaw.modules.convolution.conv_transpose.conv import (
    DistributedConv1DTranspose as DistributedConv1DTranspose,
)

# historical name for the 1-tuple arg normaliser.
_single = _ntuple(1)
