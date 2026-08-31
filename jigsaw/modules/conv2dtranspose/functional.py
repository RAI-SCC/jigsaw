"""Autograd definition for the distributed 2-D transpose convolution.

The 1-D and 3-D transpose autograd functions
(:mod:`jigsaw.modules.conv1dtranspose.functional`,
:mod:`jigsaw.modules.conv3dtranspose.functional`) mirror this one.
"""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from .config import Conv2dTransposeConfig


class Conv2dTransposeCtx(Protocol):
    """Protocol for the values stashed on ``ctx`` between forward and backward."""

    local_in_spatial: tuple[int, ...]
    stride: int | tuple[int, ...]
    padding: int | tuple[int, ...]
    dilation: int | tuple[int, ...]
    spatial_groups: tuple[dist.ProcessGroup | None, ...]
    channel_group: dist.ProcessGroup
    in_channels_global: int
    bayesian: bool
    config: Conv2dTransposeConfig
    saved_tensors: tuple[torch.Tensor, torch.Tensor]


class Conv2dTranspose(torch.autograd.Function):
    """Autograd function for the distributed 2-D transpose convolution.

    Handles both the no-halo regime (``stride == kernel_size``) and the halo
    regime (``stride != kernel_size``); the kernels are picked by
    :class:`~jigsaw.modules.conv2dtranspose.config.Conv2dTransposeConfig`.
    """

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        in_channels: int,
        spatial_groups: tuple[dist.ProcessGroup | None, ...],
        channel_group: dist.ProcessGroup,
        bayesian: bool,
        config: Conv2dTransposeConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the distributed Conv2dTranspose function.

        Returns ``(y, x_aug)`` where ``y`` is this rank's output-channel (and, in
        the halo regime, output-position) slice and ``x_aug`` is the gathered
        full-channel input kept for the backward pass.
        """
        y, x_aug = config.dist_conv_transpose2d_fn(
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
            in_channels,
            spatial_groups,
            channel_group,
            bayesian,
            config,
        ) = inputs

        _, x_aug = output

        # x_aug carries no gradient of its own; g_x flows through input 0.
        ctx.mark_non_differentiable(x_aug)

        ctx.save_for_backward(x_aug, weight)
        c = cast(Conv2dTransposeCtx, ctx)
        c.spatial_groups = spatial_groups
        c.channel_group = channel_group
        c.in_channels_global = in_channels
        # (h, w) sizes of this rank's owned input slice (pre channel-gather); the
        # halo backward needs them to size the raw-output span it folds grads
        # back onto.
        c.local_in_spatial = tuple(x.shape[-2:])
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
        None,
    ]:
        """Backward pass of the distributed Conv2dTranspose.

        ``g_b`` is the local ``grad_output`` sum (spatial ranks are reduced by
        the parameter hook). ``g_x`` / ``g_w`` are delegated to the regime-
        specific kernel picked in :class:`Conv2dTransposeConfig`: each rank holds
        the full-channel grad wrt input for its own output-channel slice, which
        is reduced across the channel group; the halo regime additionally folds
        the grad back onto the owning spatial ranks.
        """
        c = cast(Conv2dTransposeCtx, ctx)
        x_aug, weight = c.saved_tensors
        config = c.config

        g_b = config.grad_bias(grad_output)

        g_x, g_w = config.dist_conv_transpose2d_backward_fn(
            grad_output,
            x_aug,
            weight,
            c.stride,
            c.padding,
            c.dilation,
            c.in_channels_global,
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
            None,
        )

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: int,
        padding: int,
        dilation: int,
        in_channels: int,
        spatial_groups: tuple[dist.ProcessGroup | None, ...],
        channel_group: dist.ProcessGroup,
        bayesian: bool,
        config: Conv2dTransposeConfig,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Handle vmap for the distributed Conv2dTranspose."""
        out = Conv2dTranspose.apply(
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            in_channels,
            spatial_groups,
            channel_group,
            bayesian,
            config,
        )

        return out, 0
