"""Back-compat shim: the distributed 1-D convolution now lives in ``jigsaw.modules.convolution.conv``."""

from jigsaw.modules.convolution.conv.base import _ntuple
from jigsaw.modules.convolution.conv.conv import DistributedConv1D as DistributedConv1D

# historical name for the 1-tuple arg normaliser.
_single = _ntuple(1)
