"""Config setup for distributed 1-D transpose convolutions.

Mirrors :class:`jigsaw.modules.conv2dtranspose.config.Conv2dTransposeConfig` for
``F.conv_transpose1d``.
"""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from jigsaw.modules.conv_channel_comm import resolve_channel_comm

from . import distributed_conv1dtranspose as dist_conv


class Conv1dTransposeConfig:
    """Selection strategy for the 1-D transpose-conv kernels used in fwd/bwd."""

    # type hints
    dist_conv_transpose1d_fn: Callable
    dist_conv_transpose1d_backward_fn: Callable

    def __init__(
        self,
        bayesian: bool,
        channel_group: dist.ProcessGroup,
        bias: Optional[Parameter],
        kernel_size: tuple[int],
        stride: tuple[int],
        device: torch.device,
        channel_comm: str = "auto",
    ):
        """
        Initialize selection strategy for conv transpose 1d.

        bayesian: bool
            Flag for whether the model is bayesian.
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
        channel_comm: str
            How the channel-parallel group exchanges input slices in the forward
            pass: ``"all_gather"`` (one-shot gather), ``"p2p"`` (double-buffered
            batched p2p that overlaps comm with compute and lowers the memory
            peak) or ``"all_to_all"`` (same ring over masked all-to-all hops,
            NCCL only). ``"auto"`` (the default) picks ``"all_to_all"`` on a
            CUDA ``device`` and ``"p2p"`` otherwise.
        """
        self.channel_comm = resolve_channel_comm(channel_comm, device)
        self.conv1dtranspose_fn = self._conv_transpose1d_fn_selection(bayesian)
        self.conv_transpose1d_grad_fn = self._conv_transpose1d_grad_fn_selection(
            bayesian
        )
        self.weight_reshape = self._reshape_weights_selection(bayesian)
        self.grad_bias = self._grad_bias_selection(bias)
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
        """Sum ``grad_output`` over batch and length, leaving the channel axis."""
        return grad_output.sum(dim=(-3, -1))
