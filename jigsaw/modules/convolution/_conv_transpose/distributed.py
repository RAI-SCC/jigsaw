"""Kernels of the distributed transpose convolution.

Without halo (``stride == kernel_size``) each spatial rank computes its own
slice. Otherwise the overlapping raw outputs of neighboring ranks are summed
along each split axis.
"""

from typing import Any

import torch
import torch.distributed as dist

from .._common.spatial_split import (
    _local_padding,
    _needs_halo,
    _reduce_scatter_channels,
    _split_axes,
)
from .channel_comm import over_channels_transpose
from .halo import transpose_halo_expand, transpose_halo_reduce


def _grad_input_weight(
    grad_output: torch.Tensor,
    x_full: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    config: Any,
    output_mask: tuple[bool, bool, bool] = (True, True, False),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the local input and weight gradients.

    ``output_mask`` selects the gradients to compute. The others are ``None``.
    """
    g_x, g_w, _ = config.conv_transpose_grad_fn(
        grad_output.to(weight.dtype),  # handles autocast
        x_full,
        weight,
        None,  # bias_sizes: grad_bias is handled separately
        list(stride),
        list(padding),
        list(dilation),
        True,  # transposed
        [0] * config.n_spatial,  # output_padding
        config.groups_local,  # groups (1 in the full-channel-gather regime)
        list(output_mask),  # (grad_input, grad_weight, grad_bias)
    )
    return g_x, g_w


def _channel_reduce_grad_input(
    g_x: torch.Tensor, channel_group: dist.ProcessGroup, config: Any
) -> tuple[torch.Tensor, dist.Work | None]:
    """Sum the input gradient over the channel group and keep the local shard.

    No reduction is needed if each rank owns whole conv-groups.
    """
    if config.channel_local:
        return g_x.contiguous(), None
    return _reduce_scatter_channels(g_x, channel_group, config.n_spatial)


def _grad_input_weight_reduced(
    grad_output: torch.Tensor,
    x_full: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    channel_group: dist.ProcessGroup,
    config: Any,
    need: tuple[bool, bool] = (True, True),
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute the input and weight gradients, reducing the input gradient over channels.

    The weight gradient is computed after the asynchronous reduction of the
    input gradient is started, unless the model is vmapped. ``need`` selects
    the input and weight gradients; one that is not needed is ``None``.
    """
    need_x, need_w = need
    if not need_x:
        _, g_w = _grad_input_weight(
            grad_output,
            x_full,
            weight,
            stride,
            padding,
            dilation,
            config,
            output_mask=(False, True, False),
        )
        return None, g_w
    if not need_w:
        g_x, _ = _grad_input_weight(
            grad_output,
            x_full,
            weight,
            stride,
            padding,
            dilation,
            config,
            output_mask=(True, False, False),
        )
        g_x, handle = _channel_reduce_grad_input(g_x, channel_group, config)
        if handle is not None:
            handle.wait()
        return g_x, None
    reduce_pending = channel_group.size() > 1 and not config.channel_local
    if reduce_pending and not config.vmapped:
        g_x, _ = _grad_input_weight(
            grad_output,
            x_full,
            weight,
            stride,
            padding,
            dilation,
            config,
            output_mask=(True, False, False),
        )
        g_x, handle = _channel_reduce_grad_input(g_x, channel_group, config)
        _, g_w = _grad_input_weight(
            grad_output,
            x_full,
            weight,
            stride,
            padding,
            dilation,
            config,
            output_mask=(False, True, False),
        )
        if handle is not None:
            handle.wait()
        return g_x, g_w

    g_x, g_w = _grad_input_weight(
        grad_output, x_full, weight, stride, padding, dilation, config
    )
    g_x, handle = _channel_reduce_grad_input(g_x, channel_group, config)
    if handle is not None:
        handle.wait()
    return g_x, g_w


# ---------------------------------------------------------------------------
# no-halo regime (stride == kernel_size)
# ---------------------------------------------------------------------------
def conv_transpose_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    spatial_groups: tuple[dist.ProcessGroup | None, ...],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the transpose convolution without halo exchange (``stride == kernel_size``).

    Delegates to :func:`conv_transpose_halo_nway` if a split axis has padding or
    dilation.

    Parameters
    ----------
    x : torch.Tensor
        Local input.
    w : torch.Tensor
        Local weight.
    b : torch.Tensor | None
        Local bias.
    stride : tuple[int, ...]
        Stride of each spatial axis.
    padding : tuple[int, ...]
        Padding of each spatial axis.
    dilation : tuple[int, ...]
        Dilation of each spatial axis.
    spatial_groups : tuple[dist.ProcessGroup | None, ...]
        Process group of each spatial axis, ``None`` if the axis is not split.
    channel_group : dist.ProcessGroup
        Process group splitting the channels.
    config : ConvTransposeNdConfig
        Kernel selection.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Output and the full-channel input kept for the backward pass.
    """
    if _needs_halo(spatial_groups, padding, dilation):
        return conv_transpose_halo_nway(
            x, w, b, stride, padding, dilation, spatial_groups, channel_group, config
        )

    return over_channels_transpose(
        x,
        w,
        b,
        {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": config.groups_local,
        },
        channel_group=channel_group,
        conv_func=config.conv_transpose_fn,
        n_spatial_dims=config.n_spatial,
        channel_comm=config.channel_comm,
        local=config.channel_local,
    )


def conv_transpose_nway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    local_in_spatial: tuple[int, ...],
    spatial_groups: tuple[dist.ProcessGroup | None, ...],
    channel_group: dist.ProcessGroup,
    config: Any,
    need: tuple[bool, bool] = (True, True),
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute the gradients of the transpose convolution without halo exchange.

    Delegates to :func:`conv_transpose_halo_nway_backward` if a split axis has
    padding or dilation.

    Parameters
    ----------
    grad_output : torch.Tensor
        Gradient of the output.
    x_aug : torch.Tensor
        Full-channel input kept from the forward pass.
    weight : torch.Tensor
        Local weight.
    stride : tuple[int, ...]
        Stride of each spatial axis.
    padding : tuple[int, ...]
        Padding of each spatial axis.
    dilation : tuple[int, ...]
        Dilation of each spatial axis.
    local_in_spatial : tuple[int, ...]
        Spatial sizes of the local input shard.
    spatial_groups : tuple[dist.ProcessGroup | None, ...]
        Process group of each spatial axis, ``None`` if the axis is not split.
    channel_group : dist.ProcessGroup
        Process group splitting the channels.
    config : ConvTransposeNdConfig
        Kernel selection.
    need : tuple[bool, bool]
        Whether the input and the weight gradient are needed; a gradient
        that is not needed is ``None`` and its communication is skipped.

    Returns
    -------
    tuple[torch.Tensor | None, torch.Tensor | None]
        Gradients of the input and the weight.
    """
    if _needs_halo(spatial_groups, padding, dilation):
        return conv_transpose_halo_nway_backward(
            grad_output,
            x_aug,
            weight,
            stride,
            padding,
            dilation,
            local_in_spatial,
            spatial_groups,
            channel_group,
            config,
            need,
        )

    return _grad_input_weight_reduced(
        grad_output,
        x_aug,
        weight,
        stride,
        padding,
        dilation,
        channel_group,
        config,
        need,
    )


# ---------------------------------------------------------------------------
# halo regime (stride != kernel_size)
#
# The transpose halo exchange is dimension-agnostic and lives in
# :mod:`jigsaw.modules.convolution._conv_transpose.halo`. One exchange is run per split
# axis; the split axis's conv parameters are indexed out of the per-axis tuples
# by ``spatial_idx``.
# ---------------------------------------------------------------------------
def conv_transpose_halo_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    spatial_groups: tuple[dist.ProcessGroup | None, ...],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the transpose convolution with halo reduction (``stride != kernel_size``).

    The overlapping raw outputs are summed along each split axis before the bias
    is added.

    Parameters
    ----------
    x : torch.Tensor
        Local input.
    w : torch.Tensor
        Local weight.
    b : torch.Tensor | None
        Local bias.
    stride : tuple[int, ...]
        Stride of each spatial axis.
    padding : tuple[int, ...]
        Padding of each spatial axis.
    dilation : tuple[int, ...]
        Dilation of each spatial axis.
    spatial_groups : tuple[dist.ProcessGroup | None, ...]
        Process group of each spatial axis, ``None`` if the axis is not split.
    channel_group : dist.ProcessGroup
        Process group splitting the channels.
    config : ConvTransposeNdConfig
        Kernel selection.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Output and the full-channel input kept for the backward pass.
    """
    local_in_spatial = tuple(x.shape[-config.n_spatial :])

    def _post_fn(y: torch.Tensor) -> torch.Tensor:
        for idx, axis, group in _split_axes(spatial_groups):
            y = transpose_halo_reduce(
                y,
                group,
                local_in_spatial[idx],
                w.shape[axis],
                stride[idx],
                padding[idx],
                dilation[idx],
                axis=axis,
            )
        return y

    return over_channels_transpose(
        x,
        w,
        b,
        {
            "stride": stride,
            "padding": _local_padding(padding, spatial_groups),
            "dilation": dilation,
            "groups": config.groups_local,
        },
        channel_group=channel_group,
        conv_func=config.conv_transpose_fn,
        n_spatial_dims=config.n_spatial,
        channel_comm=config.channel_comm,
        post_fn=_post_fn,
        local=config.channel_local,
    )


def conv_transpose_halo_nway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    local_in_spatial: tuple[int, ...],
    spatial_groups: tuple[dist.ProcessGroup | None, ...],
    channel_group: dist.ProcessGroup,
    config: Any,
    need: tuple[bool, bool] = (True, True),
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute the gradients of the transpose convolution with halo reduction.

    Parameters
    ----------
    grad_output : torch.Tensor
        Gradient of the output.
    x_aug : torch.Tensor
        Full-channel input kept from the forward pass.
    weight : torch.Tensor
        Local weight.
    stride : tuple[int, ...]
        Stride of each spatial axis.
    padding : tuple[int, ...]
        Padding of each spatial axis.
    dilation : tuple[int, ...]
        Dilation of each spatial axis.
    local_in_spatial : tuple[int, ...]
        Spatial sizes of the local input shard.
    spatial_groups : tuple[dist.ProcessGroup | None, ...]
        Process group of each spatial axis, ``None`` if the axis is not split.
    channel_group : dist.ProcessGroup
        Process group splitting the channels.
    config : ConvTransposeNdConfig
        Kernel selection.
    need : tuple[bool, bool]
        Whether the input and the weight gradient are needed; a gradient
        that is not needed is ``None`` and its communication is skipped.

    Returns
    -------
    tuple[torch.Tensor | None, torch.Tensor | None]
        Gradients of the input and the weight.
    """
    grad_raw = grad_output
    for idx, axis, group in reversed(list(_split_axes(spatial_groups))):
        grad_raw = transpose_halo_expand(
            grad_raw,
            group,
            local_in_spatial[idx],
            weight.shape[axis],
            stride[idx],
            padding[idx],
            dilation[idx],
            axis=axis,
        )

    return _grad_input_weight_reduced(
        grad_raw,
        x_aug,
        weight,
        stride,
        _local_padding(padding, spatial_groups),
        dilation,
        channel_group,
        config,
        need,
    )
