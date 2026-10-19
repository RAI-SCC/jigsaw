"""Kernel selection for the distributed transpose convolution."""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ..common.channel_comm import resolve_channel_comm
from . import distributed as dist_conv

# native forward primitives indexed by ``n_spatial - 1``.
_CONV_TRANSPOSE_FN = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)


class ConvTransposeNdConfig:
    """Selection of the transpose-convolution kernels of the forward and backward pass."""

    # type hints
    n_spatial: int
    bayesian: bool
    dist_conv_transpose_fn: Callable
    dist_conv_transpose_backward_fn: Callable

    def __init__(
        self,
        bayesian: bool,
        channel_group: dist.ProcessGroup,
        bias: Parameter | None,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        device: torch.device,
        channel_comm: str = "auto",
        groups: int = 1,
    ):
        """Initialize the configuration.

        Parameters
        ----------
        bayesian : bool
            Whether the model is Bayesian.
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
        self.n_spatial = len(kernel_size)
        self.bayesian = bayesian
        self.channel_comm = resolve_channel_comm(channel_comm, device)
        self.groups = groups
        # ``channel_local``: this rank owns whole conv-groups -> no channel-group
        # exchange, the native transpose conv is called with ``groups_local``
        # groups. (base validated the divisibility.)
        group_size = channel_group.size()
        self.channel_local = groups % group_size == 0
        self.groups_local = groups // group_size if self.channel_local else 1
        self.native_conv_transpose_fn = _CONV_TRANSPOSE_FN[self.n_spatial - 1]
        self.conv_transpose_fn = self._conv_transpose_fn_selection(bayesian)
        self.conv_transpose_grad_fn = self._conv_transpose_grad_fn_selection(bayesian)
        self.weight_reshape = self._reshape_weights_selection(bayesian)
        self.grad_bias = self._grad_bias_selection(bias)
        # ``stride == kernel_size`` makes each rank's raw output tile the global
        # output with no overlap, so the spatial split needs no communication.
        # Otherwise neighbouring ranks' raw outputs overlap and the spatial ranks
        # exchange halos.
        if all(k == s for k, s in zip(kernel_size, stride)):
            self.dist_conv_transpose_fn = dist_conv.conv_transpose_nway
            self.dist_conv_transpose_backward_fn = (
                dist_conv.conv_transpose_nway_backward
            )
        else:
            self.dist_conv_transpose_fn = dist_conv.conv_transpose_halo_nway
            self.dist_conv_transpose_backward_fn = (
                dist_conv.conv_transpose_halo_nway_backward
            )

    def _conv_transpose_fn_selection(self, bayesian: bool) -> Callable:
        """Select the forward primitive."""
        if bayesian:
            return torch.vmap(self.native_conv_transpose_fn)
        return self.native_conv_transpose_fn

    def _conv_transpose_grad_fn_selection(self, bayesian: bool) -> Callable:
        """Select the backward primitive."""
        if bayesian:
            return torch.vmap(
                torch.ops.aten.convolution_backward,
                in_dims=(0, 0, 0, None, None, None, None, None, None, None, None),
                out_dims=(0, 0, None),  # type: ignore[arg-type]
            )
        return torch.ops.aten.convolution_backward

    def _reshape_weights_selection(self, bayesian: bool) -> Callable:
        """Select the weight-reshape function."""
        if bayesian:
            return self._reshape_weights
        return self._no_reshape_weights

    def _reshape_weights(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Broadcast the weights over the vmap dimension."""
        return w.expand(x.shape[0], *w.shape)

    def _no_reshape_weights(self, w: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """No-op."""
        return w

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
