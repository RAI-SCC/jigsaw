"""Autograd definition for convolutional layers."""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from .config import ConvConfig


class ConvCtx(Protocol):
    """Protocol for types stored within ctx."""

    x_shape_padded: torch.Size
    stride: int | tuple[int, ...]
    padding: int | tuple[int, ...]
    dilation: int | tuple[int, ...]
    spatial_group: dist.ProcessGroup
    channel_group: dist.ProcessGroup
    in_channels_global: int
    bayesian: bool
    config: ConvConfig
    saved_tensors: tuple[torch.Tensor, torch.Tensor]


class Conv2d(torch.autograd.Function):
    """Autograd function for distributed 2D convolution (stride == kernel_size)."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        in_channels: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        config: ConvConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of distributed Conv2d function.

        Returns ``(y, x_aug)`` where ``y`` is this rank's output-channel slice
        and ``x_aug`` is the gathered full-channel input kept for backward.
        """
        if any(k != s for k, s in zip(kernel_size, stride)):
            raise NotImplementedError(
                "Distributed Conv2d currently supports only stride == kernel_size "
                "(no halo exchange)."
            )

        y, x_aug = config.dist_conv2d_fn(
            x,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
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
        """
        Set up context for backward pass.

        Parameters
        ----------
        ctx : context object
            Context to save information for backward.
        inputs : tuple
            All inputs passed to forward (x, weight, bias, kernel_size, ...).
        output : tuple
            All outputs returned from forward (y, x_aug).
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
        _, x_aug = output

        # x_aug carries no gradient of its own; g_x flows through input 0.
        ctx.mark_non_differentiable(x_aug)

        # Save everything needed for backward
        ctx.save_for_backward(x_aug, weight)
        c = cast(ConvCtx, ctx)
        c.spatial_group = spatial_group
        c.channel_group = channel_group
        c.in_channels_global = in_channels
        c.x_shape_padded = x_aug.shape
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
        """Backward pass of the distributed Conv2d.

        Each rank holds the full-channel grad wrt input for its own
        output-channel slice; summing those across the channel group and keeping
        this rank's channel slice yields ``g_x``. ``g_w`` / ``g_b`` need no
        channel reduction (spatial ranks are reduced by the parameter hook).
        """
        c = cast(ConvCtx, ctx)
        x_aug, weight = c.saved_tensors
        stride, padding, dilation = (c.stride, c.padding, c.dilation)
        channel_group = c.channel_group
        config = c.config

        x_shape_padded = c.x_shape_padded

        g_b = config.grad_bias(grad_output, c.spatial_group)

        g_x = config.conv2d_input_fn(
            x_shape_padded[int(c.bayesian) :],
            weight,
            grad_output.to(weight.dtype),  # handles autocast
            stride=stride,
            padding=padding,
            dilation=dilation,
        ).contiguous()

        g_x_handle = dist.all_reduce(
            g_x, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
        )

        g_w = config.conv2d_weight_fn(
            x_aug,
            weight.shape[int(c.bayesian) :],
            grad_output.to(weight.dtype),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        g_x_handle.wait()

        shard = c.in_channels_global // channel_group.size()
        start = channel_group.rank() * shard
        g_x = g_x[..., start : start + shard, :, :]

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
