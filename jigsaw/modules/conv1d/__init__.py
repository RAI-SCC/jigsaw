"""Back-compat shim: the distributed 1-D convolution now lives in ``jigsaw.modules.convolution.conv``."""

from jigsaw.modules.convolution.conv import DistributedConv1D as DistributedConv1D
