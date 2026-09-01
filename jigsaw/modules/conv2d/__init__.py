"""Back-compat shim: the distributed 2-D convolution now lives in ``jigsaw.modules.conv``."""

from jigsaw.modules.conv import DistributedConv2D as DistributedConv2D
