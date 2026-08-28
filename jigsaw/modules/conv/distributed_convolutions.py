"""Distributed convolution implementation.

Two regimes are supported, selected by :class:`~jigsaw.modules.conv.config.ConvConfig`:

* ``stride == kernel_size`` (and no dilation/padding on the split axis): the
  spatial split needs no communication -- every spatial rank convolves its own
  domain slice independently (:func:`conv2d_nway`).
* ``stride != kernel_size``: neighbouring output columns share input columns, so
  each spatial rank first pulls the missing input columns from the other ranks
  with a halo exchange, then convolves (:func:`conv2d_halo_nway`).

Both regimes cooperate with the channel-parallel group through
:func:`_gather_full_channels`, which works for any group size.
The spatial split is 1-D, along the last axis (width); height is never split.
"""

from typing import Any, Callable

import torch
import torch.distributed as dist

from jigsaw.modules.spatial_halo import halo_gather, halo_scatter_add


# ---------------------------------------------------------------------------
# channel-parallel helpers
# ---------------------------------------------------------------------------
def _gather_full_channels(
    x: torch.Tensor, channel_group: dist.ProcessGroup
) -> torch.Tensor:
    """All-gather the channel-parallel input slices into a full-channel tensor.

    Each rank owns a contiguous ``C_in / G`` slice of the input channels; the
    group all-gathers them and rebuilds ``(..., C_in, H, W)``. This rank's own
    slice is spliced back in so it stays wired into the autograd graph. Group
    size one is the identity.
    """
    group_size = channel_group.size()
    if group_size == 1:
        return x

    local_channel_rank = channel_group.rank()
    slices = [torch.empty_like(x) for _ in range(group_size)]
    dist.all_gather(slices, x, group=channel_group)
    slices[local_channel_rank] = x
    return torch.cat(slices, dim=-3)


def distributed_conv2d_over_channels_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    channel_group: dist.ProcessGroup,
    conv_func: Callable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-parallel Conv2d for an arbitrary channel-group size.

    Each rank owns a contiguous ``in_channels // channel_group.size()`` slice of
    the input channels and the matching slice of the weight's *output* channels
    (``w`` still carries every input channel). The ranks all-gather their input
    slices, rebuild the full-channel input and run the local convolution, so the
    result already holds this rank's output-channel slice.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
    """
    x_full = _gather_full_channels(x.contiguous(), channel_group)

    if b is not None:
        y = conv_func(x_full, w, b, stride=stride, padding=padding, dilation=dilation)
    else:
        y = conv_func(x_full, w, stride=stride, padding=padding, dilation=dilation)

    return y, x_full


def conv2d_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed 2d convolution for the no-halo regime (``stride == kernel``).

    Every spatial rank convolves its own domain slice independently while the
    channel group cooperates through
    :func:`distributed_conv2d_over_channels_nway`.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
    """
    return distributed_conv2d_over_channels_nway(
        x,
        w,
        b,
        stride,
        padding,
        dilation,
        channel_group,
        config.conv2d_fn,
    )


def conv2d_nway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    in_channels_global: int,
    local_in_width: int,
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    bayesian: bool,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward for the no-halo regime.

    Each rank holds the full-channel grad wrt input for its own output-channel
    slice; summing those across the channel group and keeping this rank's channel
    slice yields ``g_x``. ``g_w`` needs no channel reduction (spatial ranks are
    reduced by the parameter hook).
    """
    g_x = config.conv2d_input_fn(
        x_aug.shape[int(bayesian) :],
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
        weight.shape[int(bayesian) :],
        grad_output.to(weight.dtype),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    g_x_handle.wait()

    shard = in_channels_global // channel_group.size()
    start = channel_group.rank() * shard
    g_x = g_x[..., start : start + shard, :, :]

    return g_x, g_w


# ---------------------------------------------------------------------------
# halo regime (stride != kernel_size)
#
# The halo exchange itself is dimension-agnostic and lives in
# :mod:`jigsaw.modules.spatial_halo` (shared with the 1-D convolution); only the
# split axis's conv parameters are threaded through here (``*[1]`` -- the width).
# ---------------------------------------------------------------------------
def _width_padding(padding: tuple[int, int]) -> tuple[int, int]:
    """Return the local-conv padding: height as given, width already materialised."""
    return (padding[0], 0)


def conv2d_halo_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed 2d convolution for the halo regime (``stride != kernel``).

    Gathers the full-channel input, pulls the missing width columns from the
    other spatial ranks, then runs the local convolution so the result already
    holds this rank's output-channel *and* output-column slice.

    Returns ``(y, x_need)`` where ``x_need`` is the halo-padded full-channel
    input kept for the backward pass.
    """
    if not isinstance(padding, tuple):
        raise NotImplementedError(
            "Distributed Conv2d with stride != kernel_size does not support "
            f"string padding ({padding!r}); pass an explicit (h, w) padding."
        )

    x_full = _gather_full_channels(x.contiguous(), channel_group)
    x_need, _ = halo_gather(
        x_full, spatial_group, w.shape[-1], stride[1], padding[1], dilation[1]
    )

    local_padding = _width_padding(padding)
    if b is not None:
        y = config.conv2d_fn(
            x_need, w, b, stride=stride, padding=local_padding, dilation=dilation
        )
    else:
        y = config.conv2d_fn(
            x_need, w, stride=stride, padding=local_padding, dilation=dilation
        )

    return y, x_need


def conv2d_halo_nway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    in_channels_global: int,
    local_in_width: int,
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    bayesian: bool,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward for the halo regime.

    ``g_w`` comes straight from the halo-padded input and this rank's
    ``grad_output`` (spatial ranks are summed by the parameter hook). ``g_x`` is
    computed on the halo-padded span, reduced across the channel group, folded
    back onto the owning spatial ranks (:func:`_halo_scatter_add`) and finally
    sliced to this rank's channel shard.
    """
    local_padding = _width_padding(padding)

    g_x_need = config.conv2d_input_fn(
        x_aug.shape[int(bayesian) :],
        weight,
        grad_output.to(weight.dtype),  # handles autocast
        stride=stride,
        padding=local_padding,
        dilation=dilation,
    ).contiguous()

    g_x_handle = dist.all_reduce(
        g_x_need, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
    )

    g_w = config.conv2d_weight_fn(
        x_aug,
        weight.shape[int(bayesian) :],
        grad_output.to(weight.dtype),
        stride=stride,
        padding=local_padding,
        dilation=dilation,
    )

    g_x_handle.wait()

    g_x = halo_scatter_add(
        g_x_need,
        spatial_group,
        weight.shape[-1],
        stride[1],
        padding[1],
        dilation[1],
        local_in_width,
    )

    shard = in_channels_global // channel_group.size()
    start = channel_group.rank() * shard
    g_x = g_x[..., start : start + shard, :, :]

    return g_x, g_w
