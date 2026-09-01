"""Distributed 1-/2-/3-D transpose convolutions sharing one implementation.

``base`` holds the dimension-independent module (``_ConvTransposeNd``-style),
``functional`` / ``config`` / ``distributed`` the autograd Function, kernel
selection and distributed kernels, ``channel_comm`` / ``halo`` the
channel-parallel and spatial-split communication, and ``conv`` the three thin
concrete classes. Pieces identical to the plain-conv case live in
:mod:`jigsaw.modules.convolution.common`.
"""

from .base import _DistributedConvTransposeNd as _DistributedConvTransposeNd
from .config import ConvTransposeNdConfig as ConvTransposeNdConfig
from .conv import DistributedConv1DTranspose as DistributedConv1DTranspose
from .conv import DistributedConv2DTranspose as DistributedConv2DTranspose
from .conv import DistributedConv3DTranspose as DistributedConv3DTranspose
from .functional import ConvTransposeNd as ConvTransposeNd
