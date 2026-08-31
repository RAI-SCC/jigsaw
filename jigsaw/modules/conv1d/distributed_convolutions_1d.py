"""Distributed 1-D convolution implementation.

Mirrors :mod:`jigsaw.modules.conv2d.distributed_convolutions_2d` for ``F.conv1d``.
Input is ``(N, C, L)``; the domain is split along the length axis (the last
axis) across the spatial group, and the channels are split across the channel
group.

Two regimes are selected by :class:`~jigsaw.modules.conv1d.config.Conv1dConfig`:

* ``stride == kernel_size`` (and no dilation/padding on the split axis): the
  length split tiles with no overlap, so every spatial rank convolves its own
  slice independently (:func:`conv1d_nway`).
* otherwise: neighbouring output positions share input positions, so each
  spatial rank pulls the missing length positions from the other ranks with a
  halo exchange and convolves (:func:`conv1d_halo_nway`). The halo exchange
  itself is dimension-agnostic and shared with the 2-/3-D convolutions via
  :mod:`jigsaw.modules.spatial_halo`.

Either regime cooperates with the channel-parallel group through the shared
``config.channel_comm`` strategies
(:func:`jigsaw.modules.conv_channel_comm.over_channels`). In the halo regime the
halo exchange is handed to ``over_channels`` as a ``halo_fn``: the
``"all_gather"`` strategy applies it once to the gathered full-channel tensor,
while the streaming (``"p2p"`` / ``"all_to_all"``) strategies apply it to each
channel slab as the slabs travel around the ring.
"""

from typing import Any

import torch
import torch.distributed as dist

from jigsaw.modules.conv_channel_comm import over_channels
from jigsaw.modules.spatial_halo import halo_gather, halo_scatter_add


def _needs_halo(
    spatial_group: dist.ProcessGroup,
    padding: str | tuple[int],
    dilation: tuple[int],
) -> bool:
    """Whether a nominally no-halo call (``stride == kernel``) still needs a halo.

    ``stride == kernel_size`` only tiles the length axis with no overlap when it
    also has zero padding and unit dilation. With a real spatial split and either
    of those non-trivial, the local slices' interior edges would be wrongly
    zero-padded / their windows would overlap, so the halo path is required.
    """
    return (
        spatial_group.size() > 1
        and isinstance(padding, tuple)
        and (padding[0] != 0 or dilation[0] != 1)
    )


# ---------------------------------------------------------------------------
# no-halo regime (stride == kernel_size)
# ---------------------------------------------------------------------------
def conv1d_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int],
    padding: str | tuple[int],
    dilation: tuple[int],
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed 1d convolution for the no-halo regime (``stride == kernel``).

    Every spatial rank convolves its own length slice independently while the
    channel group cooperates through ``config.channel_comm``.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
    """
    if _needs_halo(spatial_group, padding, dilation):
        return conv1d_halo_nway(
            x, w, b, stride, padding, dilation, spatial_group, channel_group, config
        )

    return over_channels(
        x,
        w,
        b,
        {"stride": stride, "padding": padding, "dilation": dilation},
        channel_group=channel_group,
        conv_func=config.conv1d_fn,
        n_spatial_dims=1,
        channel_comm=config.channel_comm,
    )


def conv1d_nway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
    in_channels_global: int,
    local_in_length: int,
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
        return conv1d_halo_nway_backward(
            grad_output,
            x_aug,
            weight,
            stride,
            padding,
            dilation,
            in_channels_global,
            local_in_length,
            spatial_group,
            channel_group,
            bayesian,
            config,
        )

    g_x = config.conv1d_input_fn(
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

    g_w = config.conv1d_weight_fn(
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
    g_x = g_x[..., start : start + shard, :]

    return g_x, g_w


# ---------------------------------------------------------------------------
# halo regime (stride != kernel_size)
# ---------------------------------------------------------------------------
def conv1d_halo_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int],
    padding: str | tuple[int],
    dilation: tuple[int],
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed 1d convolution for the halo regime (``stride != kernel``).

    Builds a ``halo_fn`` that pulls the missing length positions from the other
    spatial ranks, and hands it to
    :func:`~jigsaw.modules.conv_channel_comm.over_channels` together with the
    channel-parallel work. ``over_channels`` invokes ``halo_fn`` once on the
    gathered full-channel tensor (``channel_comm="all_gather"``) or once per
    channel slab as the slabs stream around the ring (``"p2p"`` /
    ``"all_to_all"``); either way this rank ends with its output-channel *and*
    output-position slice.

    Returns ``(y, x_need)`` where ``x_need`` is the halo-padded full-channel
    input kept for the backward pass.
    """
    if not isinstance(padding, tuple):
        raise NotImplementedError(
            "Distributed Conv1d with stride != kernel_size does not support "
            f"string padding ({padding!r}); pass an explicit padding."
        )

    kernel_w = w.shape[-1]

    def halo_fn(t: torch.Tensor) -> torch.Tensor:
        span, _ = halo_gather(
            t, spatial_group, kernel_w, stride[0], padding[0], dilation[0]
        )
        return span

    # padding along the (only) split axis is materialised into the halo output.
    return over_channels(
        x,
        w,
        b,
        {"stride": stride, "padding": 0, "dilation": dilation},
        channel_group=channel_group,
        conv_func=config.conv1d_fn,
        n_spatial_dims=1,
        channel_comm=config.channel_comm,
        halo_fn=halo_fn,
    )


def conv1d_halo_nway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int],
    padding: tuple[int],
    dilation: tuple[int],
    in_channels_global: int,
    local_in_length: int,
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    bayesian: bool,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward for the halo regime.

    ``g_w`` comes straight from the halo-padded input and this rank's
    ``grad_output`` (spatial ranks are summed by the parameter hook). ``g_x`` is
    computed on the halo-padded span, folded back onto the owning spatial ranks
    (:func:`halo_scatter_add`), reduced across the channel group (overlapped with
    the ``g_w`` compute) and finally sliced to this rank's channel shard.
    """
    g_x_need = config.conv1d_input_fn(
        x_aug.shape[int(bayesian) :],
        weight,
        grad_output.to(weight.dtype),  # handles autocast
        stride=stride,
        padding=0,
        dilation=dilation,
    ).contiguous()

    # Fold the halo-borrowed positions back onto their owners *before* the
    # channel-group reduce. halo_scatter_add (spatial group) and the all-reduce
    # (channel group) act on independent group axes and are both linear, so they
    # commute; folding first shrinks the all-reduce payload from the halo-padded
    # need-span to this rank's owned span.
    g_x = halo_scatter_add(
        g_x_need,
        spatial_group,
        weight.shape[-1],
        stride[0],
        padding[0],
        dilation[0],
        local_in_length,
    ).contiguous()

    g_x_handle = dist.all_reduce(
        g_x, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
    )

    g_w = config.conv1d_weight_fn(
        x_aug,
        weight.shape[int(bayesian) :],
        grad_output.to(weight.dtype),
        stride=stride,
        padding=0,
        dilation=dilation,
    )

    g_x_handle.wait()

    shard = in_channels_global // channel_group.size()
    start = channel_group.rank() * shard
    g_x = g_x[..., start : start + shard, :]

    return g_x, g_w
