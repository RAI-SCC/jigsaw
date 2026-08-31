"""Distributed 1-D convolution implementation.

Mirrors :mod:`jigsaw.modules.conv2d.distributed_convolutions_2d` for ``F.conv1d``.
Input is ``(N, C, L)``; the domain is split along the length axis (the last
axis) across the spatial group, and the channels are split across the channel
group.

Two regimes are selected by :class:`~jigsaw.modules.conv1d.config.Conv1dConfig`:

* ``stride == kernel_size``: the length split tiles with no overlap, so every
  spatial rank convolves its own slice independently (:func:`conv1d_nway`).
* ``stride != kernel_size``: neighbouring output positions share input
  positions, so each spatial rank first pulls the missing length positions from
  the other ranks with a halo exchange, then convolves
  (:func:`conv1d_halo_nway`). The halo exchange itself is dimension-agnostic and
  shared with the 2-D convolution via :mod:`jigsaw.modules.spatial_halo`.
"""

from typing import Any

import torch
import torch.distributed as dist

from jigsaw.modules.spatial_halo import halo_gather, halo_scatter_add


# ---------------------------------------------------------------------------
# channel-parallel helper
# ---------------------------------------------------------------------------
def _gather_full_channels(
    x: torch.Tensor, channel_group: dist.ProcessGroup
) -> torch.Tensor:
    """All-gather the channel-parallel input slices into a full-channel tensor.

    Each rank owns a contiguous ``C_in / G`` slice of the input channels; the
    group all-gathers them and rebuilds ``(..., C_in, L)``. This rank's own
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
    return torch.cat(slices, dim=-2)


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
    channel group cooperates through the input all-gather.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
    """
    if _needs_halo(spatial_group, padding, dilation):
        return conv1d_halo_nway(
            x, w, b, stride, padding, dilation, spatial_group, channel_group, config
        )

    x_full = _gather_full_channels(x.contiguous(), channel_group)

    if b is not None:
        y = config.conv1d_fn(
            x_full, w, b, stride=stride, padding=padding, dilation=dilation
        )
    else:
        y = config.conv1d_fn(
            x_full, w, stride=stride, padding=padding, dilation=dilation
        )

    return y, x_full


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

    Gathers the full-channel input, pulls the missing length positions from the
    other spatial ranks, then runs the local convolution so the result already
    holds this rank's output-channel *and* output-position slice.

    Returns ``(y, x_need)`` where ``x_need`` is the halo-padded full-channel
    input kept for the backward pass.
    """
    if not isinstance(padding, tuple):
        raise NotImplementedError(
            "Distributed Conv1d with stride != kernel_size does not support "
            f"string padding ({padding!r}); pass an explicit padding."
        )

    x_full = _gather_full_channels(x.contiguous(), channel_group)
    x_need, _ = halo_gather(
        x_full, spatial_group, w.shape[-1], stride[0], padding[0], dilation[0]
    )

    # padding along the split axis is already materialised into ``x_need``.
    if b is not None:
        y = config.conv1d_fn(x_need, w, b, stride=stride, padding=0, dilation=dilation)
    else:
        y = config.conv1d_fn(x_need, w, stride=stride, padding=0, dilation=dilation)

    return y, x_need


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
    computed on the halo-padded span, reduced across the channel group, folded
    back onto the owning spatial ranks (:func:`halo_scatter_add`) and finally
    sliced to this rank's channel shard.
    """
    g_x_need = config.conv1d_input_fn(
        x_aug.shape[int(bayesian) :],
        weight,
        grad_output.to(weight.dtype),  # handles autocast
        stride=stride,
        padding=0,
        dilation=dilation,
    ).contiguous()

    g_x_handle = dist.all_reduce(
        g_x_need, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
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

    g_x = halo_scatter_add(
        g_x_need,
        spatial_group,
        weight.shape[-1],
        stride[0],
        padding[0],
        dilation[0],
        local_in_length,
    )

    shard = in_channels_global // channel_group.size()
    start = channel_group.rank() * shard
    g_x = g_x[..., start : start + shard, :]

    return g_x, g_w
