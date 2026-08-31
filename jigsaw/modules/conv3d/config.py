"""Config setup for distributed 3-D convolutions.

Mirrors :class:`jigsaw.modules.conv2d.config.Conv2dConfig` for ``F.conv3d``.
"""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.grad import conv3d_input, conv3d_weight
from torch.nn.parameter import Parameter

from jigsaw.modules.conv_channel_comm import resolve_channel_comm

from . import distributed_convolutions_3d as dist_conv


class Conv3dConfig:
    """Selection strategy for the 3-D convolution kernels used in fwd/bwd."""

    # type hints
    dist_conv3d_fn: Callable
    dist_conv3d_backward_fn: Callable

    def __init__(
        self,
        bayesian: bool,
        channel_group: dist.ProcessGroup,
        bias: Parameter | None,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        device: torch.device,
        channel_comm: str = "auto",
    ):
        """
        Initialize selection strategy for conv3d.

        bayesian: bool
            Flag for whether the model is bayesian.
        channel_group: dist.ProcessGroup
            Tensor-parallel process group.
        bias: torch.Tensor | None
            Bias if applicable.
        kernel_size: tuple[int, int, int]
            Kernel size of the convolution.
        stride: tuple[int, int, int]
            Size of the stride.
        device: torch.device
            Device upon which communication occurs.
        channel_comm: str
            How the channel-parallel group exchanges input slices in the forward
            pass: ``"all_gather"`` (one-shot gather), ``"p2p"`` (double-buffered
            batched p2p that overlaps comm with compute and lowers the memory
            peak) or ``"all_to_all"`` (same ring over masked all-to-all hops,
            NCCL only). ``"auto"`` (the default) picks ``"all_to_all"`` on a
            CUDA ``device`` and ``"p2p"`` otherwise.
        """
        self.channel_comm = resolve_channel_comm(channel_comm, device)
        self.conv3d_fn = self._conv3d_fn_selection(bayesian)
        self.conv3d_input_fn = self._conv3d_input_fn_selection(bayesian)
        self.conv3d_weight_fn = self._conv3d_weight_fn_selection(bayesian)
        self.weight_reshape = self._reshape_weights_selection(bayesian)
        self.grad_bias = self._grad_bias_selection(bias)
        # ``stride == kernel_size`` tiles the domain with no overlap, so the
        # spatial split needs no communication. Otherwise neighbouring output
        # positions share input positions and the spatial ranks exchange halos.
        if all(k == s for k, s in zip(kernel_size, stride)):
            self.dist_conv3d_fn = dist_conv.conv3d_nway
            self.dist_conv3d_backward_fn = dist_conv.conv3d_nway_backward
        else:
            self.dist_conv3d_fn = dist_conv.conv3d_halo_nway
            self.dist_conv3d_backward_fn = dist_conv.conv3d_halo_nway_backward

    def _conv3d_fn_selection(self, bayesian: bool) -> Callable:
        """Select the forward primitive, vmapped when the model is bayesian."""
        if bayesian:
            return torch.vmap(F.conv3d)
        return F.conv3d

    def _conv3d_input_fn_selection(self, bayesian: bool) -> Callable:
        """Select the grad-input primitive, vmapped when the model is bayesian."""
        if bayesian:
            return torch.vmap(conv3d_input, in_dims=(None, 0, 0))
        return conv3d_input

    def _conv3d_weight_fn_selection(self, bayesian: bool) -> Callable:
        """Select the grad-weight primitive, vmapped when the model is bayesian."""
        if bayesian:
            return torch.vmap(conv3d_weight, in_dims=(0, None, 0))
        return conv3d_weight

    def _reshape_weights_selection(self, bayesian: bool) -> Callable:
        """Select the weight-reshape helper (a no-op unless bayesian)."""
        if bayesian:
            return self._reshape_weights
        return self._no_reshape_weights

    def _reshape_weights(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Broadcast the weights over the vmap (batch) dimension."""
        return w.expand(x.shape[0], *w.shape)

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

    def _sum_grad_bias(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Sum ``grad_output`` over batch and spatial axes, leaving the channels."""
        return grad_output.sum(dim=(-5, -3, -2, -1))
