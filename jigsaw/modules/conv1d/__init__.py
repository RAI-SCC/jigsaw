"""Back-compat shim: the distributed 1-D convolution now lives in ``jigsaw.modules.conv``."""

from jigsaw.modules.conv import DistributedConv1D as DistributedConv1D
