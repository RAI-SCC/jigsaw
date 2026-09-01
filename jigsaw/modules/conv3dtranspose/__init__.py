"""Back-compat shim: the distributed 3-D transpose convolution now lives in ``jigsaw.modules.conv_transpose``."""

from jigsaw.modules.conv_transpose import (
    DistributedConv3DTranspose as DistributedConv3DTranspose,
)
