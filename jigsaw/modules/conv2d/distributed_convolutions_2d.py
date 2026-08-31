"""Distributed 2-D convolution implementation.

The 1-D and 3-D implementations
(:mod:`jigsaw.modules.conv1d.distributed_convolutions_1d`,
:mod:`jigsaw.modules.conv3d.distributed_convolutions_3d`) mirror this one.
Input is ``(N, C, H, W)``; the domain is split along the width axis (the last
axis) across the spatial group, and the channels are split across the channel
group.

Two regimes are selected by :class:`~jigsaw.modules.conv2d.config.Conv2dConfig`:

* ``stride == kernel_size`` (and no dilation/padding on the split axis): the
  width split tiles with no overlap, so every spatial rank convolves its own
  slice independently (:func:`conv2d_nway`).
* otherwise: neighbouring output columns share input columns, so each spatial
  rank first pulls the missing width columns from the other ranks with a halo
  exchange, then convolves (:func:`conv2d_halo_nway`). The halo exchange itself
  is dimension-agnostic and shared with the 1-/3-D convolutions via
  :mod:`jigsaw.modules.spatial_halo`.

Either regime cooperates with the channel-parallel group through the shared
``config.channel_comm`` strategies
(:func:`jigsaw.modules.conv_channel_comm.over_channels`).
"""

from typing import Any

import torch
import torch.distributed as dist

from jigsaw.modules.conv_channel_comm import over_channels
from jigsaw.modules.spatial_halo import halo_gather, halo_scatter_add


def _needs_halo(
    spatial_group: dist.ProcessGroup,
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
) -> bool:
    """Whether a nominally no-halo call (``stride == kernel``) still needs a halo.

    ``stride == kernel_size`` only tiles the split (width) axis with no overlap
    when it also has zero padding and unit dilation. With a real spatial split
    and either of those non-trivial, the local slices' interior edges would be
    wrongly zero-padded / their windows would overlap, so the halo path is
    required.
    """
    return (
        spatial_group.size() > 1
        and isinstance(padding, tuple)
        and (padding[1] != 0 or dilation[1] != 1)
    )


def _width_padding(padding: tuple[int, int]) -> tuple[int, int]:
    """Local-conv padding: height as given, width already materialised."""
    return (padding[0], 0)


# ---------------------------------------------------------------------------
# no-halo regime (stride == kernel_size)
# ---------------------------------------------------------------------------
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

    Every spatial rank convolves its own width slice independently while the
    channel group cooperates through ``config.channel_comm``.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
    """
    if _needs_halo(spatial_group, padding, dilation):
        return conv2d_halo_nway(
            x, w, b, stride, padding, dilation, spatial_group, channel_group, config
        )

    return over_channels(
        x,
        w,
        b,
        {"stride": stride, "padding": padding, "dilation": dilation},
        channel_group=channel_group,
        conv_func=config.conv2d_fn,
        n_spatial_dims=2,
        channel_comm=config.channel_comm,
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
    if _needs_halo(spatial_group, padding, dilation):
        return conv2d_halo_nway_backward(
            grad_output,
            x_aug,
            weight,
            stride,
            padding,
            dilation,
            in_channels_global,
            local_in_width,
            spatial_group,
            channel_group,
            bayesian,
            config,
        )

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
# :mod:`jigsaw.modules.spatial_halo` (shared with the 1-/3-D convolutions); only
# the split axis's conv parameters are threaded through here (``*[1]`` -- the
# width).
# ---------------------------------------------------------------------------
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

    Each spatial rank pulls the missing width columns from the other spatial
    ranks with a halo exchange, then convolves, so the result already holds this
    rank's output-channel *and* output-column slice. The channel group
    cooperates through ``config.channel_comm`` exactly as :func:`conv2d_nway`
    does.

    Returns ``(y, x_need)`` where ``x_need`` is the halo-padded full-channel
    input kept for the backward pass.
    """
    if not isinstance(padding, tuple):
        raise NotImplementedError(
            "Distributed Conv2d with stride != kernel_size does not support "
            f"string padding ({padding!r}); pass an explicit (h, w) padding."
        )

    kernel_w = w.shape[-1]

    def halo_fn(t: torch.Tensor) -> torch.Tensor:
        span, _ = halo_gather(
            t, spatial_group, kernel_w, stride[1], padding[1], dilation[1]
        )
        return span

    return over_channels(
        x,
        w,
        b,
        {"stride": stride, "padding": _width_padding(padding), "dilation": dilation},
        channel_group=channel_group,
        conv_func=config.conv2d_fn,
        n_spatial_dims=2,
        channel_comm=config.channel_comm,
        halo_fn=halo_fn,
    )


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
    back onto the owning spatial ranks (:func:`halo_scatter_add`) and finally
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
