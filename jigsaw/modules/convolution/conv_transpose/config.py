"""Config setup for the distributed N-D transpose convolution.

One :class:`ConvTransposeNdConfig` serves ``F.conv_transpose1d`` /
``F.conv_transpose2d`` / ``F.conv_transpose3d``; the spatial rank is inferred from
``len(kernel_size)`` and selects the native forward primitive. The grad-input /
grad-weight primitive (``torch.ops.aten.convolution_backward``) and every other
selection rule (the no-halo vs halo regime split, the grad-bias reduction) are
dimension-independent.

There is no ``bayesian`` construction flag: the ensemble / members axis is owned
entirely by
:meth:`~jigsaw.modules.convolution.conv_transpose.functional.ConvTransposeNd.vmap`,
which hands the kernels an operand with an extra leading batch dim. The
``conv_transpose_fn`` / ``conv_transpose_grad_fn`` dispatchers notice that dim at
call time (rank ``> n_spatial + 2``) and route through a ``torch.vmap`` of the
*local* primitive so each member gets its own weights.

The transpose analogue of :class:`jigsaw.modules.convolution.conv.config.ConvNdConfig`.
"""

from typing import Any, Callable, Optional, cast

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
        channel_group: dist.ProcessGroup,
        bias: Parameter | None,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        device: torch.device,
        channel_comm: str = "auto",
        groups: int = 1,
    ):
        """
        Initialize selection strategy for the N-D transpose convolution.

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
            Transpose-convolution groups (PyTorch semantics). When ``groups`` is
            a multiple of the channel-parallel group size ``G`` every rank owns
            whole conv-groups: the kernels run a plain grouped transpose conv on
            this rank's channel shard with ``groups_local = groups // G`` groups
            and no channel-group communication. ``groups == 1`` keeps the
            full-channel-gather path.
        """
        self.n_spatial = len(kernel_size)
        self.channel_comm = resolve_channel_comm(channel_comm, device)
        self.groups = groups
        # ``channel_local``: this rank owns whole conv-groups -> no channel-group
        # exchange, the native transpose conv is called with ``groups_local``
        # groups. (base validated the divisibility.)
        group_size = channel_group.size()
        self.channel_local = groups % group_size == 0
        self.groups_local = groups // group_size if self.channel_local else 1
        # Native primitives + their ``torch.vmap`` (over a leading members axis);
        # the dispatchers below pick from the operand rank at call time.
        conv_transpose = _CONV_TRANSPOSE_FN[self.n_spatial - 1]
        self._conv_transpose_fn = conv_transpose
        self._conv_transpose_fn_members = torch.vmap(conv_transpose)
        self._conv_transpose_grad_fn = torch.ops.aten.convolution_backward
        # ``convolution_backward`` is dimension-independent (the spatial rank
        # rides in the stride/padding/dilation lists); grad_output / input /
        # weight carry the members axis, the trailing scalars / lists do not.
        self._conv_transpose_grad_fn_members = torch.vmap(
            torch.ops.aten.convolution_backward,
            in_dims=(0, 0, 0, None, None, None, None, None, None, None, None),
            out_dims=(0, 0, None),  # type: ignore[arg-type]
        )
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

    def _has_members(self, ndim: int) -> bool:
        """Whether ``ndim`` carries a leading members axis on top of ``(N, C, *spatial)``."""
        return ndim > self.n_spatial + 2

    def conv_transpose_fn(
        self, x: torch.Tensor, w: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Native forward transpose conv; ``torch.vmap``-ed when ``w`` carries a members axis."""
        fn = (
            self._conv_transpose_fn_members
            if self._has_members(w.ndim)
            else self._conv_transpose_fn
        )
        return cast(torch.Tensor, fn(x, w, *args, **kwargs))

    def conv_transpose_grad_fn(
        self,
        grad_output: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        *args: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """``convolution_backward``; ``torch.vmap``-ed when ``weight`` carries a members axis."""
        fn = (
            self._conv_transpose_grad_fn_members
            if self._has_members(weight.ndim)
            else self._conv_transpose_grad_fn
        )
        return cast(
            "tuple[torch.Tensor, torch.Tensor, None]",
            fn(grad_output, input, weight, *args),
        )

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
