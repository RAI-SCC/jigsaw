"""Autograd definition for convolutional layers."""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from jigsaw.modules.buffers import get_preallocated_buffer

from . import distributed_convolutions as dist_conv
from .config import ConvConfig


class ConvCtx(Protocol):
    """Protocol for types stored within ctx."""

    halo_sizes: torch.Tensor
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
    config: ConvConfig
    empty: torch.Tensor
    saved_tensors: tuple[torch.Tensor, torch.Tensor]


class Conv2d(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        kernel_size: tuple[int, int],
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
        config: ConvConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of distributed Conv2d function."""
        # Compute the forward pass

        y, x0, x1, right_to_left, left_to_right = config.dist_conv2d_fn(
            x,
            weight,
            bias,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            in_channels=in_channels,
            spatial_group=spatial_group,
            channel_group=channel_group,
            empty=empty,
            config=config,
        )

        x_aug = torch.cat([x0, x1], dim=-3)  # concatenate along the channel dimension
        halo_sizes = torch.tensor(
            [right_to_left, left_to_right]
        )  # save halo sizes in tensor

        # y is result, rest of the outputs are required for the backward pass
        return y, x_aug, x, halo_sizes

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: Any,
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """
        Set up context for backward pass.

        Parameters
        ----------
        ctx : context object
            Context to save information for backward.
        inputs : tuple
            All inputs passed to forward (x, weight, bias, kernel_size, ...).
        output : tuple
            All outputs returned from forward (y, x_aug, x, right_to_left, left_to_right).
        """
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

        # Unpack outputs
        _, x_aug, x_orig, halo_sizes = output

        # Mark non-differentiable outputs (indices 1-4)
        ctx.mark_non_differentiable(x_aug)
        ctx.mark_non_differentiable(x_orig)
        ctx.mark_non_differentiable(halo_sizes)

        # Save everything needed for backward
        ctx.save_for_backward(x_aug, weight)
        c = cast(ConvCtx, ctx)
        c.halo_sizes = halo_sizes
        # Save non-tensor attributes
        c.device = device
        c.parallelism = parallelism
        c.spatial_group = spatial_group
        c.channel_group = channel_group
        c.in_channels_global = in_channels
        c.b = bias
        c.empty = empty
        c.x_shape_padded = x_aug.shape
        c.x_shape = x_orig.shape
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
        grad_x_orig: None,  # No gradient for x_orig
        halo_sizes: None,
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
        """Backward pass of Conv2d distributed across 4 or 8 processes."""
        c = cast(ConvCtx, ctx)
        x_aug, weight = c.saved_tensors
        empty = c.empty
        stride, padding, dilation = (c.stride, c.padding, c.dilation)
        spatial_group, channel_group = (
            c.spatial_group,
            c.channel_group,
        )
        config = c.config

        right_to_left, left_to_right = c.halo_sizes[0], c.halo_sizes[1]
        x_shape_padded, x_shape = c.x_shape_padded, c.x_shape

        g_b = config.grad_bias(grad_output, c.spatial_group)

        g_x_aug = config.conv2d_input_fn(
            x_shape_padded[int(c.bayesian) :],
            weight,
            grad_output.to(weight.dtype),  # handles autocast
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        left_to_right, right_to_left = (right_to_left, left_to_right)

        # g_x calculation
        # only communicates halos if they exist
        g_x = config.communicate_halos_backward(
            g_x_aug,
            x_shape,
            right_to_left,
            left_to_right,
            c.parallelism,
            empty,
            spatial_group,
            async_op=True,
        )

        g_x_buffer = get_preallocated_buffer(
            torch.Size((*g_x.shape[:-3], g_x.shape[-3] // 2, *g_x.shape[-2:])),
            g_x.dtype,
            channel_group.rank(),
            g_x.device,
        )

        g_x_recv, g_x_handles = dist_conv.send_g_x_channels_2d(
            g_x,
            g_x_buffer,
            empty,
            c.in_channels_global,
            channel_group,
            async_op=True,
            comm_fn=config.comm_fn,
        )

        g_w = config.conv2d_weight_fn(
            x_aug,
            weight.shape[int(c.bayesian) :],
            grad_output.to(weight.dtype),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        for g_x_handle in g_x_handles:
            if g_x_handle is not None:
                g_x_handle.wait()

        g_x = (
            g_x[..., : c.in_channels_global // 2, :, :] + g_x_recv
            if channel_group.rank() % 2 == 0
            else g_x[..., c.in_channels_global // 2 :, :, :] + g_x_recv
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
        config: ConvConfig,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Handle vmap for Distributed Conv2d."""
        out = Conv2d.apply(
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
