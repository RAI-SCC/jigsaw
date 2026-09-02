"""Concrete distributed 1-/2-/3-D convolutions.

Each class only pins ``_n_spatial``; every behaviour lives in
:class:`jigsaw.modules.convolution.conv.base._DistributedConvNd` (mirroring how
``torch.nn.Conv{1,2,3}d`` only normalise their tuple args on top of
``_ConvNd``).
"""

from .base import _DistributedConvNd


class DistributedConv1D(_DistributedConvNd):
    """Distributed 1-D convolution.

    input shape of forward pass: ``(N, Cin, Lin)``
    output shape of forward pass: ``(N, Cout, Lout)``
    """

    _n_spatial = 1


class DistributedConv2D(_DistributedConvNd):
    """Distributed 2-D convolution.

    input shape of forward pass: ``(N, Cin, Hin, Win)``
    output shape of forward pass: ``(N, Cout, Hout, Wout)``
    """

    _n_spatial = 2


class DistributedConv3D(_DistributedConvNd):
    """Distributed 3-D convolution.

    input shape of forward pass: ``(N, Cin, Din, Hin, Win)``
    output shape of forward pass: ``(N, Cout, Dout, Hout, Wout)``
    """

    _n_spatial = 3
