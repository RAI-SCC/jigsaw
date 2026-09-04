"""Autograd definition for the distributed N-D convolution.

One :class:`ConvNd` autograd Function serves ``F.conv1d`` / ``F.conv2d`` /
``F.conv3d``; the spatial rank is carried by ``config.n_spatial`` (the length of
the normalised ``kernel_size``). The 1-/2-/3-D modules pick their dimension by
constructing a :class:`~jigsaw.modules.convolution.conv.config.ConvNdConfig` of the matching
arity.
"""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from ..common.autocast import cast_forward_inputs
from .config import ConvNdConfig


class ConvNdCtx(Protocol):
    """Protocol for the values stashed on ``ctx`` between forward and backward."""

    local_in_spatial: tuple[int, ...]
    stride: int | tuple[int, ...]
    padding: int | tuple[int, ...]
    dilation: int | tuple[int, ...]
    spatial_groups: tuple[dist.ProcessGroup | None, ...]
    channel_group: dist.ProcessGroup
    bayesian: bool
    config: ConvNdConfig
    saved_tensors: tuple[torch.Tensor, torch.Tensor]


class ConvNd(torch.autograd.Function):
    """Autograd function for the distributed N-D convolution.

    Handles both the no-halo regime (``stride == kernel_size``) and the halo
    regime (``stride != kernel_size``); the kernels are picked by
    :class:`~jigsaw.modules.convolution.conv.config.ConvNdConfig`.
    """

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        spatial_groups: tuple[dist.ProcessGroup | None, ...],
        channel_group: dist.ProcessGroup,
        bayesian: bool,
        config: ConvNdConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the distributed ConvNd function.

        Returns ``(y, x_aug)`` where ``y`` is this rank's output-channel (and,
        in the halo regime, output-position) slice and ``x_aug`` is the input
        kept for backward: the gathered full-channel input in the no-halo
        regime, or additionally halo-padded along every split spatial axis in
        the halo regime.
        """
        x, weight, bias = cast_forward_inputs(x, weight, bias)
        y, x_aug = config.dist_conv_fn(
            x,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            spatial_groups=spatial_groups,
            channel_group=channel_group,
            config=config,
        )

        return y, x_aug

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: Any,
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Stash tensors and metadata needed for the backward pass."""
        (
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            spatial_groups,
            channel_group,
            bayesian,
            config,
        ) = inputs

        _, x_aug = output

        # x_aug carries no gradient of its own; g_x flows through input 0.
        ctx.mark_non_differentiable(x_aug)

        ctx.save_for_backward(x_aug, weight)
        c = cast(ConvNdCtx, ctx)
        c.spatial_groups = spatial_groups
        c.channel_group = channel_group
        # spatial sizes of this rank's owned input slice (pre channel-gather,
        # pre halo); the halo backward needs them to fold borrowed grads to
        # owners.
        c.local_in_spatial = tuple(x.shape[-config.n_spatial :])
        c.stride = stride
        c.padding = padding
        c.dilation = dilation
        c.bayesian = bayesian
        c.config = config

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
        grad_x_aug: None,  # No gradient for x_aug
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        """Backward pass of the distributed ConvNd.

        ``g_b`` is the local ``grad_output`` sum (spatial ranks are reduced by
        the parameter hook). ``g_x`` / ``g_w`` are delegated to the regime-
        specific kernel picked in :class:`ConvNdConfig`: in the no-halo regime
        each rank holds the full-channel grad wrt input for its own
        output-channel slice and only the channel group is reduced; in the halo
        regime the grad is additionally folded back onto the owning spatial
        ranks.
        """
        c = cast(ConvNdCtx, ctx)
        x_aug, weight = c.saved_tensors
        config = c.config

        # Under autocast the forward cast the inputs, so ``x_aug`` comes back in
        # the autocast dtype; run backward in ``weight.dtype`` (the master
        # precision ``grad_output`` is already reconciled to downstream).
        x_aug = x_aug.to(weight.dtype)

        g_b = config.grad_bias(grad_output)

        g_x, g_w = config.dist_conv_backward_fn(
            grad_output,
            x_aug,
            weight,
            c.stride,
            c.padding,
            c.dilation,
            c.local_in_spatial,
            c.spatial_groups,
            c.channel_group,
            c.bayesian,
            config,
        )

        return (
            g_x,
            g_w,
            g_b,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: int,
        padding: int,
        dilation: int,
        spatial_groups: tuple[dist.ProcessGroup | None, ...],
        channel_group: dist.ProcessGroup,
        bayesian: bool,
        config: ConvNdConfig,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Handle vmap for the distributed ConvNd."""
        out = ConvNd.apply(
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            spatial_groups,
            channel_group,
            bayesian,
            config,
        )

        return out, 0
