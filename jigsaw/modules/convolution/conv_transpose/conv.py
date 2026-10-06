"""Distributed 1-D, 2-D and 3-D transpose convolution modules."""

from .base import _DistributedConvTransposeNd


class DistributedConv1DTranspose(_DistributedConvTransposeNd):
    """Distributed 1-D transpose convolution.

    Input has shape ``(N, Cin, Lin)`` and output ``(N, Cout, Lout)``.
    """

    _n_spatial = 1


class DistributedConv2DTranspose(_DistributedConvTransposeNd):
    """Distributed 2-D transpose convolution.

    Input has shape ``(N, Cin, Hin, Win)`` and output ``(N, Cout, Hout, Wout)``.
    """

    _n_spatial = 2


class DistributedConv3DTranspose(_DistributedConvTransposeNd):
    """Distributed 3-D transpose convolution.

    Input has shape ``(N, Cin, Din, Hin, Win)`` and output
    ``(N, Cout, Dout, Hout, Wout)``.
    """

    _n_spatial = 3
