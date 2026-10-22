"""Kernel selection for the distributed convolution."""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.grad import (
    conv1d_input,
    conv1d_weight,
    conv2d_input,
    conv2d_weight,
    conv3d_input,
    conv3d_weight,
)
from torch.nn.parameter import Parameter

from jigsaw._core.vmap import VmapConfig

from .._common.channel_comm import resolve_channel_comm
from . import distributed as dist_conv

# native primitives indexed by ``n_spatial - 1``.
_CONV_FN = (F.conv1d, F.conv2d, F.conv3d)
_CONV_INPUT_FN = (conv1d_input, conv2d_input, conv3d_input)
_CONV_WEIGHT_FN = (conv1d_weight, conv2d_weight, conv3d_weight)


class ConvNdConfig(VmapConfig):
    """Selection of the convolution kernels of the forward and backward pass.

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
    channel_comm : str
        Channel communication strategy: ``"all_gather"``, ``"p2p"``,
        ``"all_to_all"`` or ``"auto"``.
    groups : int
        Number of convolution groups.
    """

    # type hints
    n_spatial: int
    dist_conv_fn: Callable
    dist_conv_backward_fn: Callable

    def __init__(
        self,
        channel_group: dist.ProcessGroup,
        bias: Parameter | None,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        device: torch.device,
        channel_comm: str = "auto",
        groups: int = 1,
    ):
        """Initialize the configuration."""
        self.n_spatial = len(kernel_size)
        self.channel_comm = resolve_channel_comm(channel_comm, device)
        self.groups = groups
        # ``channel_local``: this rank owns whole conv-groups -> no channel-group
        # exchange, the native conv is called with ``groups_local`` groups.
        # (base validated the divisibility; recompute the split here so a config
        # built directly still resolves it.)
        group_size = channel_group.size()
        self.channel_local = groups % group_size == 0
        self.groups_local = groups // group_size if self.channel_local else 1
        self.native_conv_fn = _CONV_FN[self.n_spatial - 1]
        self.conv_fn: Callable = self.native_conv_fn
        self.conv_input_fn: Callable = _CONV_INPUT_FN[self.n_spatial - 1]
        self.conv_weight_fn: Callable = _CONV_WEIGHT_FN[self.n_spatial - 1]
        self.grad_bias = self._grad_bias_selection(bias)
        # ``stride == kernel_size`` tiles the domain with no overlap, so the
        # spatial split needs no communication. Otherwise neighbouring output
        # positions share input positions and the spatial ranks exchange halos.
        if all(k == s for k, s in zip(kernel_size, stride, strict=True)):
            self.dist_conv_fn = dist_conv.conv_nway
            self.dist_conv_backward_fn = dist_conv.conv_nway_backward
        else:
            self.dist_conv_fn = dist_conv.conv_halo_nway
            self.dist_conv_backward_fn = dist_conv.conv_halo_nway_backward

    def _select_vmapped(self) -> None:
        """Swap in the members-axis primitives on the vmapped copy.

        ``x``, the weight and the bias carry a leading members axis; the local
        primitives are vmapped over it. The collectives are members-agnostic.
        """
        self.conv_fn = torch.vmap(self.native_conv_fn)
        self.conv_input_fn = torch.vmap(self.conv_input_fn, in_dims=(None, 0, 0))
        self.conv_weight_fn = torch.vmap(self.conv_weight_fn, in_dims=(0, None, 0))

    def _grad_bias_selection(self, bias: Optional[torch.Tensor]) -> Callable:
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
