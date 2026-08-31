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

    def __init__(
        self,
        bayesian: bool,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bias: Parameter | None,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        device: torch.device,
        channel_comm: str = "auto",
    ):
        """
        Initialize selection strategy for conv.

        bayesian: bool
            Flag for whether the model is bayesian.
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.
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
        channel_comm: str
            How the channel-parallel group exchanges input slices in the forward
            pass: ``"all_gather"`` (one-shot gather), ``"ring"`` (double-buffered
            batched p2p that overlaps comm with compute and lowers the memory
            peak) or ``"all_to_all"`` (same ring over masked all-to-all hops,
            NCCL only). ``"auto"`` (the default) picks ``"all_to_all"`` on a
            CUDA ``device`` and ``"ring"`` otherwise. See
            :mod:`jigsaw.modules.conv.distributed_convolutions`.
        """
        self.channel_comm = self._resolve_channel_comm(channel_comm, device)
        self.conv2d_fn = self._conv2d_fn_selection(bayesian)
        self.conv2d_input_fn = self._conv2d_input_fn_selection(bayesian)
        self.conv2d_weight_fn = self._conv2d_weight_fn_selection(bayesian)
        self.weight_reshape = self._reshape_weights_selection(bayesian)
        self.grad_bias = self._grad_bias_selection(spatial_group, bias)
        # One kernel covers every (spatial_group.size(), channel_group.size())
        # combination in the no-halo regime.
        self.dist_conv2d_fn = dist_conv.conv2d_nway

    @staticmethod
    def _resolve_channel_comm(channel_comm: str, device: torch.device) -> str:
        """Resolve ``"auto"`` from ``device`` and validate the result.

        ``"auto"`` -> ``"all_to_all"`` on CUDA (NCCL has a real all-to-all),
        ``"ring"`` (batched ``isend`` / ``irecv``) everywhere else.
        """
        if channel_comm == "auto":
            channel_comm = "all_to_all" if device.type == "cuda" else "ring"
        if channel_comm not in dist_conv._CHANNEL_COMM_FNS:
            raise ValueError(
                f"channel_comm must be 'auto' or one of "
                f"{sorted(dist_conv._CHANNEL_COMM_FNS)}, got {channel_comm!r}"
            )
        return channel_comm

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

    def _grad_bias_selection(
        self, spatial_group: dist.ProcessGroup, bias: Optional[torch.Tensor]
    ) -> Callable:
        """
        Selector function for grad bias reduction.

        Grad of bias needs to be all-reduced to synchronize across spatial process.

        Parameters
        ----------
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.
        bias: Optional[torch.Tensor]
            Bias tensor.

        Returns
        -------
        Callable
        """
        if bias is None:
            return self._no_bias
        elif spatial_group.size() == 2:
            return self._no_reduce_grad_bias
        else:
            return self._reduce_bias_over_spatial_group

    def _no_bias(self, *args: Any, **kwargs: Any) -> None:
        """No-op."""
        return None

    def _no_reduce_grad_bias(
        self,
        grad_output: torch.Tensor,
        spatial_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """
        Sum grad_output to get grad_bias.

        Parameters
        ----------
        grad_output: torch.Tensor
            Gradient wrt output.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.

        Returns
        -------
        grad_bias: torch.Tensor
            Gradient wrt bias.
        """
        return grad_output.sum(dim=(-4, -2, -1))

    def _reduce_bias_over_spatial_group(
        self,
        grad_output: torch.Tensor,
        spatial_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """
        Sum grad_output to get grad_bias.

        Parameters
        ----------
        grad_output: torch.Tensor
            Gradient wrt output.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.

        Returns
        -------
        grad_bias: torch.Tensor
            Gradient wrt bias.
        """
        grad_b = grad_output.sum(dim=(-4, -2, -1))
        return grad_b
