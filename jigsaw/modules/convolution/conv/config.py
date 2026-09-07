"""Config setup for the distributed N-D convolution.

One :class:`ConvNdConfig` serves ``F.conv1d`` / ``F.conv2d`` / ``F.conv3d``; the
spatial rank is inferred from ``len(kernel_size)`` and selects the native
primitives (``F.conv{n}d``, ``torch.nn.grad.conv{n}d_input`` /
``conv{n}d_weight``). The regime split (no-halo vs halo) and every other
selection rule are dimension-independent.

There is no ``bayesian`` construction flag: the ensemble / members axis is owned
entirely by :meth:`~jigsaw.modules.convolution.conv.functional.ConvNd.vmap`,
which hands the kernels an operand with an extra leading batch dim. The
``conv_fn`` / ``conv_input_fn`` / ``conv_weight_fn`` dispatchers notice that dim
at call time (rank ``> n_spatial + 2``) and route through a ``torch.vmap`` of the
*local* primitive so each member gets its own weights; without it they call the
native primitive directly.
"""

from typing import Any, Callable, Optional, cast

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

from ..common.channel_comm import resolve_channel_comm
from . import distributed as dist_conv

# native primitives indexed by ``n_spatial - 1``.
_CONV_FN = (F.conv1d, F.conv2d, F.conv3d)
_CONV_INPUT_FN = (conv1d_input, conv2d_input, conv3d_input)
_CONV_WEIGHT_FN = (conv1d_weight, conv2d_weight, conv3d_weight)


class ConvNdConfig:
    """Selection strategy for the N-D convolution kernels used in fwd/bwd."""

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
        """
        Initialize selection strategy for the N-D convolution.

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
        groups: int
            Convolution groups (PyTorch semantics). When ``groups`` is a multiple
            of the channel-parallel group size ``G`` every rank owns whole
            conv-groups: the kernels run a plain grouped conv on this rank's
            channel shard with ``groups_local = groups // G`` groups and no
            channel-group communication. ``groups == 1`` keeps the
            full-channel-gather path.
        """
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
        # Native primitive + its ``torch.vmap`` (over a leading members axis); the
        # dispatchers below pick between them from the operand rank at call time.
        conv = _CONV_FN[self.n_spatial - 1]
        conv_input = _CONV_INPUT_FN[self.n_spatial - 1]
        conv_weight = _CONV_WEIGHT_FN[self.n_spatial - 1]
        self._conv_fn = conv
        self._conv_fn_members = torch.vmap(conv)
        self._conv_input_fn = conv_input
        self._conv_input_fn_members = torch.vmap(conv_input, in_dims=(None, 0, 0))
        self._conv_weight_fn = conv_weight
        self._conv_weight_fn_members = torch.vmap(conv_weight, in_dims=(0, None, 0))
        self.grad_bias = self._grad_bias_selection(bias)
        # ``stride == kernel_size`` tiles the domain with no overlap, so the
        # spatial split needs no communication. Otherwise neighbouring output
        # positions share input positions and the spatial ranks exchange halos.
        if all(k == s for k, s in zip(kernel_size, stride)):
            self.dist_conv_fn = dist_conv.conv_nway
            self.dist_conv_backward_fn = dist_conv.conv_nway_backward
        else:
            self.dist_conv_fn = dist_conv.conv_halo_nway
            self.dist_conv_backward_fn = dist_conv.conv_halo_nway_backward

    def _has_members(self, ndim: int) -> bool:
        """Whether ``ndim`` carries a leading members axis on top of ``(N, C, *spatial)``."""
        return ndim > self.n_spatial + 2

    def local_shape(self, shape: torch.Size) -> torch.Size:
        """Return the trailing ``(N, C, *spatial)`` shape, dropping any members axis.

        ``conv{n}d_input`` / ``conv{n}d_weight`` take an *un-batched* size; under
        the members ``torch.vmap`` the extra leading axis must be stripped so the
        stated size matches one member's operand.
        """
        return shape[-(self.n_spatial + 2) :]

    def conv_fn(
        self, x: torch.Tensor, w: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Native forward conv; ``torch.vmap``-ed when ``w`` carries a members axis."""
        fn = self._conv_fn_members if self._has_members(w.ndim) else self._conv_fn
        return cast(torch.Tensor, fn(x, w, *args, **kwargs))

    def conv_input_fn(
        self,
        input_size: torch.Size,
        weight: torch.Tensor,
        grad_output: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Native grad-input conv; ``torch.vmap``-ed when ``weight`` carries a members axis."""
        fn = (
            self._conv_input_fn_members
            if self._has_members(weight.ndim)
            else self._conv_input_fn
        )
        return cast(torch.Tensor, fn(input_size, weight, grad_output, **kwargs))

    def conv_weight_fn(
        self,
        input: torch.Tensor,
        weight_size: torch.Size,
        grad_output: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Native grad-weight conv; ``torch.vmap``-ed when ``input`` carries a members axis."""
        fn = (
            self._conv_weight_fn_members
            if self._has_members(input.ndim)
            else self._conv_weight_fn
        )
        return cast(torch.Tensor, fn(input, weight_size, grad_output, **kwargs))

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
