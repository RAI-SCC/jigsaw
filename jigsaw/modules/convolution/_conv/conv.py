"""Distributed 1-D, 2-D and 3-D convolution modules."""

from .base import _DistributedConvNd


class DistributedConv1d(_DistributedConvNd):
    """Distributed 1-D convolution.

    Input has global shape ``(N, Cin, Lin)`` or ``(Cin, Lin)`` and output
    ``(N, Cout, Lout)`` or ``(Cout, Lout)``.
    """

    _n_spatial = 1


class DistributedConv2d(_DistributedConvNd):
    """Distributed 2-D convolution.

    Input has global shape ``(N, Cin, Hin, Win)`` or ``(Cin, Hin, Win)`` and
    output ``(N, Cout, Hout, Wout)`` or ``(Cout, Hout, Wout)``.
    """

    _n_spatial = 2


class DistributedConv3d(_DistributedConvNd):
    """Distributed 3-D convolution.

    Input has global shape ``(N, Cin, Din, Hin, Win)`` or ``(Cin, Din, Hin, Win)``
    and output ``(N, Cout, Dout, Hout, Wout)`` or ``(Cout, Dout, Hout, Wout)``.
    """

    _n_spatial = 3
