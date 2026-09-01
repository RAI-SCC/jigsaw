"""Back-compat shim: the distributed 2-D transpose convolution now lives in ``jigsaw.modules.conv_transpose``."""

from jigsaw.modules.conv_transpose import (
    DistributedConv2DTranspose as DistributedConv2DTranspose,
)
