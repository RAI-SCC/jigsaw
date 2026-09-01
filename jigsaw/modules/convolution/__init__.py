"""Distributed N-D convolution and transpose convolution.

Two sibling packages -- :mod:`~jigsaw.modules.convolution.conv` and
:mod:`~jigsaw.modules.convolution.conv_transpose` -- each unify the 1-/2-/3-D
case behind one dimension-independent implementation (``base`` module,
``functional`` autograd Function, ``config`` kernel selection, ``distributed``
kernels, ``channel_comm`` channel-parallel driver, ``halo`` spatial exchange).
Pieces that do not differ between the forward and transpose case live in
:mod:`~jigsaw.modules.convolution.common`.
"""

from .conv import ConvNd as ConvNd
from .conv import ConvNdConfig as ConvNdConfig
from .conv import DistributedConv1D as DistributedConv1D
from .conv import DistributedConv2D as DistributedConv2D
from .conv import DistributedConv3D as DistributedConv3D
from .conv import _DistributedConvNd as _DistributedConvNd
from .conv_transpose import ConvTransposeNd as ConvTransposeNd
from .conv_transpose import ConvTransposeNdConfig as ConvTransposeNdConfig
from .conv_transpose import DistributedConv1DTranspose as DistributedConv1DTranspose
from .conv_transpose import DistributedConv2DTranspose as DistributedConv2DTranspose
from .conv_transpose import DistributedConv3DTranspose as DistributedConv3DTranspose
from .conv_transpose import _DistributedConvTransposeNd as _DistributedConvTransposeNd
