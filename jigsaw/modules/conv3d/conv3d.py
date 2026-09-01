"""Back-compat shim: the distributed 3-D convolution now lives in ``jigsaw.modules.convolution.conv``."""

from jigsaw.modules.convolution.conv.base import _ntuple
from jigsaw.modules.convolution.conv.conv import DistributedConv3D as DistributedConv3D

# historical name for the 3-tuple arg normaliser.
_triple = _ntuple(3)
