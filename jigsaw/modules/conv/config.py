"""Config setup for distributed convolutions."""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight
from torch.nn.parameter import Parameter

from . import distributed_convolutions as dist_conv


class ConvConfig:
    """Selection strategy for the convolution kernels used in the backward pass."""

    # type hints
    dist_conv2d_fn: Callable
    dist_conv2d_backward_fn: Callable

    def __init__(
        self,
        bayesian: bool,
        spatial_groups: tuple[dist.ProcessGroup | None, ...],
        channel_group: dist.ProcessGroup,
        bias: Parameter | None,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        device: torch.device,
    ):
        """
        Initialize selection strategy for conv.

        bayesian: bool
            Flag for whether the model is bayesian.
        spatial_groups: tuple[dist.ProcessGroup | None, ...]
            Per-axis domain-parallel process groups, aligned with ``(H, W)``.
            ``None`` (or size 1) marks an axis that is not split.
        channel_group: dist.ProcessGroup
            Tensor-parallel process group.
        bias: torch.Tensor | None
            Bias if applicable
        kernel_size: Tuple[int, int]
            Kernel size of the convolution
        stride: Tuple[int, int]
            Size of the stride
        device: torch.device
            Device upon which communication occurs
        """
        self.conv2d_fn = self._conv2d_fn_selection(bayesian)
        self.conv2d_input_fn = self._conv2d_input_fn_selection(bayesian)
        self.conv2d_weight_fn = self._conv2d_weight_fn_selection(bayesian)
        self.weight_reshape = self._reshape_weights_selection(bayesian)
        self.grad_bias = self._grad_bias_selection(bias)
        # ``stride == kernel_size`` (on every axis) tiles the domain with no
        # overlap, so the spatial split needs no communication. Otherwise
        # neighbouring output columns share input columns and the spatial ranks
        # exchange halos. One kernel covers every group-size combination in each
        # regime.
        if all(k == s for k, s in zip(kernel_size, stride)):
            self.dist_conv2d_fn = dist_conv.conv2d_nway
            self.dist_conv2d_backward_fn = dist_conv.conv2d_nway_backward
        else:
            self.dist_conv2d_fn = dist_conv.conv2d_halo_nway
            self.dist_conv2d_backward_fn = dist_conv.conv2d_halo_nway_backward

    def _conv2d_fn_selection(self, bayesian: bool) -> Callable:
        """
        Selector function for forward pass.

        Wraps the forward call in a vmap if necessary.

        Parameters
        ----------
        bayesian: bool
            Flag for whether model should be bayesian.

        Returns
        -------
        Callable
        """
        if bayesian:
            return torch.vmap(F.conv2d)
        else:
            return F.conv2d

    def _conv2d_input_fn_selection(self, bayesian: bool) -> Callable:
        """
        Selector function for backward pass.

        Wraps the backward grad_input calculation in a vmap if necessary.

        Parameters
        ----------
        bayesian: bool
            Flag for whether model should be bayesian.

        Returns
        -------
        Callable
        """
        if bayesian:
            return torch.vmap(conv2d_input, in_dims=(None, 0, 0))
        else:
            return conv2d_input

    def _conv2d_weight_fn_selection(self, bayesian: bool) -> Callable:
        """
        Selector function for backward pass.

        Wraps the backward grad_weight calculation in a vmap if necessary.

        Parameters
        ----------
        bayesian: bool
            Flag for whether model should be bayesian.

        Returns
        -------
        Callable
        """
        if bayesian:
            return torch.vmap(conv2d_weight, in_dims=(0, None, 0))
        else:
            return conv2d_weight

    def _reshape_weights_selection(self, bayesian: bool) -> Callable:
        """
        Selector function for reshaping the weights.

        Parameters
        ----------
        bayesian: bool
            Flag for whether the model is Bayesian.

        Returns
        -------
        reshape_weights: Callable
        """
        if bayesian:
            return self._reshape_weights
        else:
            return self._no_reshape_weights

    def _reshape_weights(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape weights as necessary for vmapped model.

        Parameters
        ----------
        w: torch.Tensor
            Convolution model weights.
        x: torch.Tensor
            data.

        Returns
        -------
        w: torch.Tensor
            Reshaped weights.
        """
        w = w.expand(x.shape[0], *w.shape)
        return w

    def _no_reshape_weights(self, w: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """No-op."""
        return w

    def _grad_bias_selection(self, bias: Optional[torch.Tensor]) -> Callable:
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
        """Sum ``grad_output`` over batch and spatial axes, leaving the channels.

        The spatial-grid reduction of this local sum is done by the parameter
        gradient hook registered on the bias.
        """
        return grad_output.sum(dim=(-4, -2, -1))
