"""Autograd function of the distributed convolution."""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from jigsaw._core.autocast import cast_to_autocast_dtype

from .._common.vmap import conv_vmap
from .config import ConvNdConfig


class ConvNdCtx(Protocol):
    """Attributes stored on the autograd context."""

    local_in_spatial: tuple[int, ...]
    stride: int | tuple[int, ...]
    padding: int | tuple[int, ...]
    dilation: int | tuple[int, ...]
    spatial_groups: tuple[dist.ProcessGroup | None, ...]
    channel_group: dist.ProcessGroup
    config: ConvNdConfig
    saved_tensors: tuple[torch.Tensor, torch.Tensor]


class _ConvNd(torch.autograd.Function):
    """Autograd function of the distributed convolution."""

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
        config: ConvNdConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the local output slice.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Output and the input kept for the backward pass.
        """
        # Opt into autocast: this Function hides the inner ``F.conv*`` from
        # autocast's op-level policy, and the kernels size their channel-gather /
        # halo buffers from these tensors before the conv runs.
        x, weight, bias = cast_to_autocast_dtype(x, weight, bias)
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
        """Save tensors and metadata for the backward pass."""
        (
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            spatial_groups,
            channel_group,
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
    ]:
        """Compute the gradients of the input, weight and bias."""
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
        )

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        *args: Any,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
        """Batching rule for ``torch.vmap``."""
        return conv_vmap(_ConvNd.apply, info, in_dims, *args)
