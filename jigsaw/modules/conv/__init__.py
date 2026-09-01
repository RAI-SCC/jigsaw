"""Distributed 1-/2-/3-D convolutions sharing one implementation.

``base`` holds the dimension-independent module (``_ConvNd``-style),
``functional`` / ``config`` / ``distributed`` the autograd Function, kernel
selection and distributed kernels, and ``conv`` the three thin concrete
classes.
"""

from .base import _DistributedConvNd as _DistributedConvNd
from .config import ConvNdConfig as ConvNdConfig
from .conv import DistributedConv1D as DistributedConv1D
from .conv import DistributedConv2D as DistributedConv2D
from .conv import DistributedConv3D as DistributedConv3D
from .functional import ConvNd as ConvNd
