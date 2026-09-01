"""Back-compat shim: the distributed 3-D convolution now lives in ``jigsaw.modules.conv``."""

from jigsaw.modules.conv import DistributedConv3D as DistributedConv3D
