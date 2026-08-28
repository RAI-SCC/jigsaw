"""Autograd definition for the distributed 3-D transpose convolution.

Mirrors :mod:`jigsaw.modules.conv_transpose.functional` for ``F.conv_transpose3d``.
"""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from .config import ConvTranspose3dConfig


class ConvTranspose3dCtx(Protocol):
    """Protocol for the values stashed on ``ctx`` between forward and backward."""

    local_in_width: int
    stride: int | tuple[int, ...]
    padding: int | tuple[int, ...]
    dilation: int | tuple[int, ...]
    spatial_group: dist.ProcessGroup
    channel_group: dist.ProcessGroup
    in_channels_global: int
    bayesian: bool
    config: ConvTranspose3dConfig
    saved_tensors: tuple[torch.Tensor, torch.Tensor]


class ConvTranspose3d(torch.autograd.Function):
    """Autograd function for the distributed 3-D transpose convolution.

    Handles both the no-halo regime (``stride == kernel_size``) and the halo
    regime (``stride != kernel_size``); the kernels are picked by
    :class:`~jigsaw.modules.conv_transpose_3d.config.ConvTranspose3dConfig`.
    """

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        in_channels: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        config: ConvTranspose3dConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the distributed ConvTranspose3d function.

        Returns ``(y, x_aug)`` where ``y`` is this rank's output-channel (and, in
        the halo regime, output-column) slice and ``x_aug`` is the gathered
        full-channel input kept for the backward pass.
        """
        y, x_aug = config.dist_conv_transpose3d_fn(
            x,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            spatial_group=spatial_group,
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
            kernel_size,
            stride,
            padding,
            dilation,
            in_channels,
            spatial_group,
            channel_group,
            device,
            parallelism,
            empty,
            bayesian,
            config,
        ) = inputs

        _, x_aug = output

        # x_aug carries no gradient of its own; g_x flows through input 0.
        ctx.mark_non_differentiable(x_aug)

        ctx.save_for_backward(x_aug, weight)
        c = cast(ConvTranspose3dCtx, ctx)
        c.spatial_group = spatial_group
        c.channel_group = channel_group
        c.in_channels_global = in_channels
        # width of this rank's owned input slice (pre channel-gather); the halo
        # backward needs it to size the raw-output span it folds grads back onto.
        c.local_in_width = x.shape[-1]
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
        None,
        None,
        None,
        None,
    ]:
        """Backward pass of the distributed ConvTranspose3d.

        ``g_b`` is the local ``grad_output`` sum (spatial ranks are reduced by
        the parameter hook). ``g_x`` / ``g_w`` are delegated to the regime-
        specific kernel picked in :class:`ConvTranspose3dConfig`.
        """
        c = cast(ConvTranspose3dCtx, ctx)
        x_aug, weight = c.saved_tensors
        config = c.config

        g_b = config.grad_bias(grad_output, c.spatial_group)

        g_x, g_w = config.dist_conv_transpose3d_backward_fn(
            grad_output,
            x_aug,
            weight,
            c.stride,
            c.padding,
            c.dilation,
            c.in_channels_global,
            c.local_in_width,
            c.spatial_group,
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
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        in_channels: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        config: ConvTranspose3dConfig,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Handle vmap for the distributed ConvTranspose3d."""
        out = ConvTranspose3d.apply(
            x,
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            dilation,
            in_channels,
            spatial_group,
            channel_group,
            device,
            parallelism,
            empty,
            bayesian,
            config,
        )

        return out, 0
