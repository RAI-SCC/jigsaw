"""Back-compat shim: the distributed 3-D transpose convolution now lives in ``jigsaw.modules.convolution.conv_transpose``."""

from jigsaw.modules.convolution.conv_transpose import (
    DistributedConv3DTranspose as DistributedConv3DTranspose,
)
