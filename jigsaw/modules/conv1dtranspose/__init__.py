"""Back-compat shim: the distributed 1-D transpose convolution now lives in ``jigsaw.modules.convolution.conv_transpose``."""

from jigsaw.modules.convolution.conv_transpose import (
    DistributedConv1DTranspose as DistributedConv1DTranspose,
)
