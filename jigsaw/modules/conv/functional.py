"""Autograd definition for convolutional layers."""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from .config import ConvConfig


class ConvCtx(Protocol):
    """Protocol for types stored within ctx."""

    local_in_spatial: tuple[int, ...]
    stride: int | tuple[int, ...]
    padding: int | tuple[int, ...]
    dilation: int | tuple[int, ...]
    spatial_groups: tuple[dist.ProcessGroup | None, ...]
    channel_group: dist.ProcessGroup
    in_channels_global: int
    bayesian: bool
    config: ConvConfig
    saved_tensors: tuple[torch.Tensor, torch.Tensor]


class Conv2d(torch.autograd.Function):
    """Autograd function for distributed 2D convolution.

    Handles both the no-halo regime (``stride == kernel_size``) and the halo
    regime (``stride != kernel_size``); the kernels are picked by
    :class:`~jigsaw.modules.conv.config.ConvConfig`.
    """

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
        spatial_groups: tuple[dist.ProcessGroup | None, ...],
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        config: ConvConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of distributed Conv2d function.

        Returns ``(y, x_aug)`` where ``y`` is this rank's output-channel (and,
        in the halo regime, output-position) slice and ``x_aug`` is the input
        kept for backward: the gathered full-channel input in the no-halo
        regime, or additionally halo-padded along every split spatial axis in
        the halo regime.
        """
        y, x_aug = config.dist_conv2d_fn(
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
            spatial_groups,
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
        c.spatial_groups = spatial_groups
        c.channel_group = channel_group
        c.in_channels_global = in_channels
        # (h, w) sizes of this rank's owned input slice (pre channel-gather, pre
        # halo); the halo backward needs them to fold borrowed grads to owners.
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
        None,
        None,
        None,
        None,
    ]:
        """Backward pass of the distributed Conv2d.

        ``g_b`` is the local ``grad_output`` sum (spatial ranks are reduced by
        the parameter hook). ``g_x`` / ``g_w`` are delegated to the regime-
        specific kernel picked in :class:`ConvConfig`: in the no-halo regime each
        rank holds the full-channel grad wrt input for its own output-channel
        slice and only the channel group is reduced; in the halo regime the grad
        is additionally folded back onto the owning spatial ranks.
        """
        c = cast(ConvCtx, ctx)
        x_aug, weight = c.saved_tensors
        config = c.config

        g_b = config.grad_bias(grad_output)

        g_x, g_w = config.dist_conv2d_backward_fn(
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
        spatial_groups: tuple[dist.ProcessGroup | None, ...],
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
            spatial_groups,
            channel_group,
            device,
            parallelism,
            empty,
            bayesian,
            config,
        )

        return out, 0
