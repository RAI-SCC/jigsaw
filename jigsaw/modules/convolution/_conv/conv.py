"""Distributed 1-D, 2-D and 3-D convolution modules."""

from .base import _DistributedConvNd


class DistributedConv1d(_DistributedConvNd):
    """Distributed 1-D convolution.

    Input has shape ``(N, Cin, Lin)`` and output ``(N, Cout, Lout)``.
    """

    _n_spatial = 1


class DistributedConv2d(_DistributedConvNd):
    """Distributed 2-D convolution.

    Input has shape ``(N, Cin, Hin, Win)`` and output ``(N, Cout, Hout, Wout)``.
    """

    _n_spatial = 2


class DistributedConv3d(_DistributedConvNd):
    """Distributed 3-D convolution.

    Input has shape ``(N, Cin, Din, Hin, Win)`` and output
    ``(N, Cout, Dout, Hout, Wout)``.
    """

    _n_spatial = 3
