"""Back-compat shim: the distributed 2-D convolution now lives in ``jigsaw.modules.convolution.conv``."""

from jigsaw.modules.convolution.conv import DistributedConv2D as DistributedConv2D
