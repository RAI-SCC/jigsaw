"""Back-compat shim: the distributed 3-D convolution now lives in ``jigsaw.modules.convolution.conv``."""

from jigsaw.modules.convolution.conv import DistributedConv3D as DistributedConv3D
