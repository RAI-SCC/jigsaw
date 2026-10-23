"""Kernel selection for the distributed transpose convolution."""

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from jigsaw._core.comm import CommOp
from jigsaw._core.vmap import VmapConfig

from .._common.channel_comm import resolve_channel_comm
from . import distributed as dist_conv

# native forward primitives indexed by ``n_spatial - 1``.
_CONV_TRANSPOSE_FN = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)


class ConvTransposeNdConfig(VmapConfig):
    """Selection of the transpose-convolution kernels of the forward and backward pass.

    Parameters
    ----------
    channel_group : dist.ProcessGroup
        Channel process group.
    bias : torch.Tensor | None
        Bias, if any.
    kernel_size : tuple[int, ...]
        Kernel size. Its length is the number of spatial dimensions.
    stride : tuple[int, ...]
        Stride.
    device : torch.device
        Device of the module.
    channel_comm : CommOp or str
        Channel communication strategy: ``"all_gather"`` or a
        :class:`~jigsaw.modules.CommOp`.
    groups : int
        Number of convolution groups.
    """

    n_spatial: int
    dist_conv_transpose_fn: Callable
    dist_conv_transpose_backward_fn: Callable

    def __init__(
        self,
        channel_group: dist.ProcessGroup,
        bias: Parameter | None,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        device: torch.device,
        channel_comm: CommOp | str = "auto",
        groups: int = 1,
    ):
        """Initialize the configuration."""
        self.n_spatial = len(kernel_size)
        self.channel_comm = resolve_channel_comm(channel_comm, device)
        self.groups = groups
        # ``channel_local``: this rank owns whole conv-groups -> no channel-group
        # exchange, the native transpose conv is called with ``groups_local``
        # groups. (base validated the divisibility.)
        group_size = channel_group.size()
        self.channel_local = groups % group_size == 0
        self.groups_local = groups // group_size if self.channel_local else 1
        self.native_conv_transpose_fn = _CONV_TRANSPOSE_FN[self.n_spatial - 1]
        self.conv_transpose_fn: Callable = self.native_conv_transpose_fn
        self.conv_transpose_grad_fn: Callable = torch.ops.aten.convolution_backward
        self.grad_bias = self._grad_bias_selection(bias)
        # ``stride == kernel_size`` makes each rank's raw output tile the global
        # output with no overlap, so the spatial split needs no communication.
        # Otherwise neighboring ranks' raw outputs overlap and the spatial ranks
        # exchange halos.
        if all(k == s for k, s in zip(kernel_size, stride, strict=True)):
            self.dist_conv_transpose_fn = dist_conv.conv_transpose_nway
            self.dist_conv_transpose_backward_fn = (
                dist_conv.conv_transpose_nway_backward
            )
        else:
            self.dist_conv_transpose_fn = dist_conv.conv_transpose_halo_nway
            self.dist_conv_transpose_backward_fn = (
                dist_conv.conv_transpose_halo_nway_backward
            )

    def _select_vmapped(self) -> None:
        """Swap in the members-axis primitives on the vmapped copy.

        ``x``, the weight and the bias carry a leading members axis; the local
        primitives are vmapped over it. The collectives are members-agnostic.
        """
        self.conv_transpose_fn = torch.vmap(self.native_conv_transpose_fn)
        self.conv_transpose_grad_fn = torch.vmap(
            torch.ops.aten.convolution_backward,
            in_dims=(0, 0, 0, None, None, None, None, None, None, None, None),
            out_dims=(0, 0, None),  # type: ignore[arg-type]
        )

    def _grad_bias_selection(self, bias: torch.Tensor | None) -> Callable:
        """Select the grad-bias function."""
        if bias is None:
            return self._no_bias
        return self._sum_grad_bias

    def _no_bias(self, *args: Any, **kwargs: Any) -> None:
        """No-op."""
        return None

    def _sum_grad_bias(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Sum ``grad_output`` over the batch and spatial axes."""
        n = self.n_spatial
        return grad_output.sum(dim=(-(n + 2), *range(-n, 0)))
