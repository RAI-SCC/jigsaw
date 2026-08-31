"""Config setup for distributed 1-D transpose convolutions.

Mirrors :class:`jigsaw.modules.conv2dtranspose.config.Conv2dTransposeConfig` for
``F.conv_transpose1d``.
"""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from . import distributed_conv1dtranspose as dist_conv


class Conv1dTransposeConfig:
    """Selection strategy for the 1-D transpose-conv kernels used in fwd/bwd."""

    # type hints
    dist_conv_transpose1d_fn: Callable
    dist_conv_transpose1d_backward_fn: Callable

    def __init__(
        self,
        bayesian: bool,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bias: Optional[Parameter],
        kernel_size: tuple[int],
        stride: tuple[int],
        device: torch.device,
    ):
        """
        Initialize selection strategy for conv transpose 1d.

        bayesian: bool
            Flag for whether the model is bayesian.
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.
        channel_group: dist.ProcessGroup
            Tensor-parallel process group.
        bias: torch.Tensor | None
            Bias if applicable.
        kernel_size: tuple[int]
            Kernel size of the convolution.
        stride: tuple[int]
            Size of the stride.
        device: torch.device
            Device upon which communication occurs.
        """
        self.conv1dtranspose_fn = self._conv_transpose1d_fn_selection(bayesian)
        self.conv_transpose1d_grad_fn = self._conv_transpose1d_grad_fn_selection(
            bayesian
        )
        self.weight_reshape = self._reshape_weights_selection(bayesian)
        self.grad_bias = self._grad_bias_selection(spatial_group, bias)
        # ``stride == kernel_size`` makes each rank's raw output tile the global
        # output with no overlap, so the spatial split needs no communication.
        # Otherwise neighbouring ranks' raw outputs overlap and the spatial ranks
        # exchange halos.
        if all(k == s for k, s in zip(kernel_size, stride)):
            self.dist_conv_transpose1d_fn = dist_conv.conv_transpose1d_nway
            self.dist_conv_transpose1d_backward_fn = (
                dist_conv.conv_transpose1d_nway_backward
            )
        else:
            self.dist_conv_transpose1d_fn = dist_conv.conv_transpose1d_halo_nway
            self.dist_conv_transpose1d_backward_fn = (
                dist_conv.conv_transpose1d_halo_nway_backward
            )

    def _conv_transpose1d_fn_selection(self, bayesian: bool) -> Callable:
        """Select the forward primitive, vmapped when the model is bayesian."""
        if bayesian:
            return torch.vmap(F.conv_transpose1d)
        return F.conv_transpose1d

    def _conv_transpose1d_grad_fn_selection(self, bayesian: bool) -> Callable:
        """Select the transpose-conv backward primitive, vmapped if bayesian."""
        if bayesian:
            return torch.vmap(
                torch.ops.aten.convolution_backward,
                in_dims=(0, 0, 0, None, None, None, None, None, None, None, None),
                out_dims=(0, 0, None),  # type: ignore[arg-type]
            )
        return torch.ops.aten.convolution_backward

    def _reshape_weights_selection(self, bayesian: bool) -> Callable:
        """Select the weight-reshape helper (a no-op unless bayesian)."""
        if bayesian:
            return self._reshape_weights
        return self._no_reshape_weights

    def _reshape_weights(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Broadcast the weights over the vmap (batch) dimension."""
        return w.expand(x.shape[0], *w.shape)

    def _no_reshape_weights(
        self, w: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """No-op."""
        return w

    def _grad_bias_selection(
        self, spatial_group: dist.ProcessGroup, bias: Optional[torch.Tensor]
    ) -> Callable:
        """Select the grad-bias reducer; the spatial reduction is done by a hook."""
        if bias is None:
            return self._no_bias
        return self._sum_grad_bias

    def _no_bias(self, *args: Any, **kwargs: Any) -> None:
        """No-op."""
        return None

    def _sum_grad_bias(
        self,
        grad_output: torch.Tensor,
        spatial_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """Sum ``grad_output`` over batch and length, leaving the channel axis."""
        return grad_output.sum(dim=(-3, -1))
