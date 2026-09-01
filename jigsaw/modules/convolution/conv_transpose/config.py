"""Config setup for the distributed N-D transpose convolution.

One :class:`ConvTransposeNdConfig` serves ``F.conv_transpose1d`` /
``F.conv_transpose2d`` / ``F.conv_transpose3d``; the spatial rank is inferred from
``len(kernel_size)`` and selects the native forward primitive. The grad-input /
grad-weight primitive (``torch.ops.aten.convolution_backward``) and every other
selection rule (the no-halo vs halo regime split, the grad-bias reduction, the
bayesian vmapping) are dimension-independent.

The transpose analogue of :class:`jigsaw.modules.convolution.conv.config.ConvNdConfig`.
"""

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
    """Selection strategy for the N-D transpose-conv kernels used in fwd/bwd."""

    # type hints
    n_spatial: int
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
    ):
        """
        Initialize selection strategy for the N-D transpose convolution.

        bayesian: bool
            Flag for whether the model is bayesian.
        channel_group: dist.ProcessGroup
            Tensor-parallel process group.
        bias: torch.Tensor | None
            Bias if applicable.
        kernel_size: tuple[int, ...]
            Kernel size of the convolution; its length is the spatial rank.
        stride: tuple[int, ...]
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
        self.n_spatial = len(kernel_size)
        self.channel_comm = resolve_channel_comm(channel_comm, device)
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
        """Select the forward primitive, vmapped when the model is bayesian."""
        conv_transpose = _CONV_TRANSPOSE_FN[self.n_spatial - 1]
        if bayesian:
            return torch.vmap(conv_transpose)
        return conv_transpose

    def _conv_transpose_grad_fn_selection(self, bayesian: bool) -> Callable:
        """Select the transpose-conv backward primitive, vmapped if bayesian.

        ``torch.ops.aten.convolution_backward`` is dimension-independent -- the
        spatial rank rides along in the ``stride`` / ``padding`` / ``dilation``
        lists -- so the same op serves every spatial rank.
        """
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
        """Sum ``grad_output`` over batch and spatial axes, leaving the channels.

        Negative axes so the vmap (bayesian) leading axis is left untouched:
        ``(N, C, *spatial)`` sums the batch axis at ``-(n_spatial + 2)`` and the
        trailing ``n_spatial`` spatial axes, leaving the channel axis.
        """
        n = self.n_spatial
        return grad_output.sum(dim=(-(n + 2), *range(-n, 0)))
