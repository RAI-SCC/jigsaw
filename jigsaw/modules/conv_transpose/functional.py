"""Autograd definition for ConvTranspose."""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from .config import ConvTransposeConfig


class ConvTransposeCtx(Protocol):
    """Protocol for types stored within ctx."""

    halo_sizes: int | tuple[int, ...]
    x_shape_padded: torch.Size
    x_shape: torch.Size
    kernel_size: int | tuple[int, ...]
    stride: int | tuple[int, ...]
    padding: int | tuple[int, ...]
    dilation: int | tuple[int, ...]
    spatial_group: dist.ProcessGroup
    channel_group: dist.ProcessGroup
    device: torch.device
    parallelism: int
    in_channels_global: int
    b: torch.Tensor | None
    bayesian: bool
    config: ConvTransposeConfig
    empty: torch.Tensor
    saved_tensors: tuple[torch.Tensor, torch.Tensor]


class ConvTranspose2d(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

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
        config: ConvTransposeConfig,
    ) -> tuple[Any, Any, torch.Tensor, Any]:
        """Forward pass for distributed conv transpose 2d."""
        y, x_aug, x, halo_sizes = config.dist_conv_transpose2d_fn(
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
            empty,
            config,
        )

        # Return output and any intermediate values needed for backward
        return y, x_aug, x, halo_sizes

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: Any,
        output: tuple[Any, Any, torch.Tensor, Any],
    ) -> None:
        """Set up context."""
        # Unpack inputs
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

        _, x_aug, x_orig, halo_sizes = output
        ctx.mark_non_differentiable(x_aug)
        ctx.mark_non_differentiable(x_orig)
        ctx.mark_non_differentiable(halo_sizes)
        ctx.save_for_backward(x_aug, weight)

        c = cast(ConvTransposeCtx, ctx)

        c.halo_sizes = halo_sizes
        c.x_shape_padded = x_aug.shape
        c.x_shape = x_orig.shape

        c.stride, c.padding, c.dilation = (stride, padding, dilation)
        c.spatial_group = spatial_group
        c.channel_group = channel_group
        c.device = device
        c.parallelism = parallelism
        c.in_channels_global = in_channels
        c.b = bias
        c.empty = empty  # pre-initialized empty tensor
        c.bayesian = bayesian
        c.config = config

    @staticmethod
    def backward(
        ctx: ConvTransposeCtx,
        grad_output: torch.Tensor,
        grad_x_aug: torch.Tensor,
        grad_x_orig: torch.Tensor,
        halo_sizes: torch.Tensor,
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
        None,
    ]:
        """
        Backward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function.ctx
            Context object to save information for backward computation.
        grad_output : tensor
            The gradient of the loss with respect to the output of the layer.
        grad_x_aug: None
            Empty tensor for torch compatability reasons.
        grad_x_orig: None
            Empty tensor for torch compatability reasons.
        halo_sizes: None
            Empty tensor for torch compatability reasons.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        ctx = cast(ConvTransposeCtx, ctx)
        x_aug, weight = ctx.saved_tensors
        empty = ctx.empty

        stride, padding, dilation = (ctx.stride, ctx.padding, ctx.dilation)
        spatial_group, channel_group = (
            ctx.spatial_group,
            ctx.channel_group,
        )
        config = ctx.config
        in_channels_global = ctx.in_channels_global

        g_x, g_w = config.dist_conv_transpose2d_backward_fn(
            grad_output.to(weight.dtype),
            x_aug,
            weight,
            halo_sizes,
            in_channels_global,
            stride,
            padding,
            dilation,
            empty,
            spatial_group,
            channel_group,
            config,
        )

        g_b = config.grad_bias(grad_output, spatial_group)

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
            None,
        )

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        kernel_size: tuple[int | int],
        stride: tuple[int | int],
        padding: tuple[int | int],
        dilation: tuple[int | int],
        in_channels: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        config: ConvTransposeConfig,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Handle vmap for ConvTranspose2d function."""
        out = ConvTranspose2d.apply(
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
