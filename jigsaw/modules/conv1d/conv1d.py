"""Back-compat shim: the distributed 1-D convolution now lives in ``jigsaw.modules.conv``."""

from jigsaw.modules.conv.base import _ntuple
from jigsaw.modules.conv.conv import DistributedConv1D as DistributedConv1D

# historical name for the 1-tuple arg normaliser.
_single = _ntuple(1)
