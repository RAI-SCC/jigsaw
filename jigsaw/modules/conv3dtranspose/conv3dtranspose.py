"""Back-compat shim: the distributed 3-D transpose convolution now lives in ``jigsaw.modules.convolution.conv_transpose``."""

from jigsaw.modules.convolution.conv_transpose.base import _ntuple
from jigsaw.modules.convolution.conv_transpose.conv import (
    DistributedConv3DTranspose as DistributedConv3DTranspose,
)

# historical name for the 3-tuple arg normaliser.
_triple = _ntuple(3)
