"""Back-compat shim: the distributed 2-D convolution now lives in ``jigsaw.modules.conv``."""

from jigsaw.modules.conv.base import _ntuple
from jigsaw.modules.conv.conv import DistributedConv2D as DistributedConv2D

# historical name for the 2-tuple arg normaliser (used by the arg-normalisation
# unit tests).
_pair = _ntuple(2)
