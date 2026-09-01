"""Concrete distributed 1-/2-/3-D transpose convolutions.

Each class only pins ``_n_spatial``; every behaviour lives in
:class:`jigsaw.modules.convolution.conv_transpose.base._DistributedConvTransposeNd`
(mirroring how ``torch.nn.ConvTranspose{1,2,3}d`` only normalise their tuple
args on top of ``_ConvTransposeNd``).
"""

from .base import _DistributedConvTransposeNd


class DistributedConv1DTranspose(_DistributedConvTransposeNd):
    """Distributed 1-D transpose convolution.

    input shape of forward pass: ``(N, Cin, Lin)``
    output shape of forward pass: ``(N, Cout, Lout)``
    """

    _n_spatial = 1


class DistributedConv2DTranspose(_DistributedConvTransposeNd):
    """Distributed 2-D transpose convolution.

    input shape of forward pass: ``(N, Cin, Hin, Win)``
    output shape of forward pass: ``(N, Cout, Hout, Wout)``
    """

    _n_spatial = 2


class DistributedConv3DTranspose(_DistributedConvTransposeNd):
    """Distributed 3-D transpose convolution.

    input shape of forward pass: ``(N, Cin, Din, Hin, Win)``
    output shape of forward pass: ``(N, Cout, Dout, Hout, Wout)``
    """

    _n_spatial = 3
