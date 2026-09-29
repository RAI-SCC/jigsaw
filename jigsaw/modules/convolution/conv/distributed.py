"""Kernels of the distributed convolution.

Without halo (``stride == kernel_size``) each spatial rank convolves its own
slice. Otherwise a halo exchange along each split axis precedes the
convolution.
"""

from typing import Any, Optional

import torch
import torch.distributed as dist

from ..common.spatial_split import (
    _local_padding,
    _needs_halo,
    _reduce_scatter_channels,
    _split_axes,
)
from .channel_comm import over_channels
from .halo import halo_gather, halo_scatter_add


def _channel_reduce_grad_input(
    g_x: torch.Tensor, channel_group: dist.ProcessGroup, config: Any
) -> tuple[torch.Tensor, Optional[dist.Work]]:
    """Sum the input gradient over the channel group and keep the local shard.

    No reduction is needed if each rank owns whole conv-groups.
    """
    if config.channel_local:
        return g_x.contiguous(), None
    return _reduce_scatter_channels(g_x, channel_group, config.n_spatial)


# ---------------------------------------------------------------------------
# no-halo regime (stride == kernel_size)
# ---------------------------------------------------------------------------
def conv_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, ...],
    padding: str | tuple[int, ...],
    dilation: tuple[int, ...],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convolve without halo exchange (``stride == kernel_size``).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Output and the full-channel input kept for the backward pass.
    """
    if _needs_halo(spatial_groups, padding, dilation):
        return conv_halo_nway(
            x, w, b, stride, padding, dilation, spatial_groups, channel_group, config
        )

    return over_channels(
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
        conv_func=config.conv_fn,
        n_spatial_dims=config.n_spatial,
        channel_comm=config.channel_comm,
        local=config.channel_local,
    )


def conv_nway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    local_in_spatial: tuple[int, ...],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    channel_group: dist.ProcessGroup,
    bayesian: bool,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the gradients of the convolution without halo exchange."""
    if _needs_halo(spatial_groups, padding, dilation):
        return conv_halo_nway_backward(
            grad_output,
            x_aug,
            weight,
            stride,
            padding,
            dilation,
            local_in_spatial,
            spatial_groups,
            channel_group,
            bayesian,
            config,
        )

    g_x = config.conv_input_fn(
        x_aug.shape[int(bayesian) :],
        weight,
        grad_output.to(weight.dtype),  # handles autocast
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=config.groups_local,
    )

    g_x, g_x_handle = _channel_reduce_grad_input(g_x, channel_group, config)

    g_w = config.conv_weight_fn(
        x_aug,
        weight.shape[int(bayesian) :],
        grad_output.to(weight.dtype),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=config.groups_local,
    )

    if g_x_handle is not None:
        g_x_handle.wait()

    return g_x, g_w


# ---------------------------------------------------------------------------
# halo regime (stride != kernel_size)
#
# The halo exchange itself is dimension-agnostic and lives in
# :mod:`jigsaw.modules.convolution.conv.halo`. One halo exchange is run per split axis;
# the split axis's conv parameters are indexed out of the per-axis tuples by
# ``spatial_idx``.
# ---------------------------------------------------------------------------
def conv_halo_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, ...],
    padding: str | tuple[int, ...],
    dilation: tuple[int, ...],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convolve with a halo exchange along each split axis (``stride != kernel_size``).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Output and the halo-padded full-channel input kept for the backward pass.
    """
    if not isinstance(padding, tuple):
        raise NotImplementedError(
            "Distributed ConvNd with stride != kernel_size does not support "
            f"string padding ({padding!r}); pass an explicit per-axis padding."
        )

    def halo_fn(t: torch.Tensor) -> torch.Tensor:
        for idx, axis, group in _split_axes(spatial_groups):
            t, _ = halo_gather(
                t,
                group,
                w.shape[axis],
                stride[idx],
                padding[idx],
                dilation[idx],
                axis=axis,
            )
        return t

    return over_channels(
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
        conv_func=config.conv_fn,
        n_spatial_dims=config.n_spatial,
        channel_comm=config.channel_comm,
        halo_fn=halo_fn,
        local=config.channel_local,
    )


def conv_halo_nway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    local_in_spatial: tuple[int, ...],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    channel_group: dist.ProcessGroup,
    bayesian: bool,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the gradients of the convolution with halo exchange."""
    local_padding = _local_padding(padding, spatial_groups)

    g_x_need = config.conv_input_fn(
        x_aug.shape[int(bayesian) :],
        weight,
        grad_output.to(weight.dtype),  # handles autocast
        stride=stride,
        padding=local_padding,
        dilation=dilation,
        groups=config.groups_local,
    ).contiguous()

    # Enqueue grad-weight before the (spatial-group) halo exchange below so the
    # two overlap instead of running back to back.
    g_w = config.conv_weight_fn(
        x_aug,
        weight.shape[int(bayesian) :],
        grad_output.to(weight.dtype),
        stride=stride,
        padding=local_padding,
        dilation=dilation,
        groups=config.groups_local,
    )

    # Fold the halo-borrowed positions back onto their owners *before* the
    # channel-group reduce. halo_scatter_add (spatial groups) and the reduce
    # (channel group) act on independent group axes and are both linear, so they
    # commute; folding first shrinks the reduce payload from the halo-padded
    # need-span to this rank's owned span.
    g_x = g_x_need
    for idx, axis, group in reversed(list(_split_axes(spatial_groups))):
        g_x = halo_scatter_add(
            g_x,
            group,
            weight.shape[axis],
            stride[idx],
            padding[idx],
            dilation[idx],
            local_in_spatial[idx],
            axis=axis,
        )

    g_x, g_x_handle = _channel_reduce_grad_input(g_x, channel_group, config)

    if g_x_handle is not None:
        g_x_handle.wait()

    return g_x, g_w
