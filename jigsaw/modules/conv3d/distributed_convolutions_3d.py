"""Distributed 3-D convolution implementation.

Mirrors :mod:`jigsaw.modules.conv2d.distributed_convolutions_2d` for ``F.conv3d``.
Input is ``(N, C, D, H, W)``; the domain is split across the spatial group(s)
and the channels are split across the channel group.

The spatial split is described by ``spatial_groups``: a 3-tuple
``(group_d, group_h, group_w)`` aligned with the ``(D, H, W)`` axes, where an
entry is ``None`` (or a group of size 1) when that axis is not split. The
default (a single group) splits the width axis only.

Two regimes are selected by :class:`~jigsaw.modules.conv3d.config.Conv3dConfig`:

* ``stride == kernel_size`` (and no dilation/padding on the split axes): the
  split tiles with no overlap, so every spatial rank convolves its own slice
  independently (:func:`conv3d_nway`).
* otherwise: neighbouring output positions share input positions, so each
  spatial rank pulls the missing rows/columns from the other ranks with a halo
  exchange -- one per split axis -- and convolves (:func:`conv3d_halo_nway`).
  The halo exchange itself is dimension-agnostic and shared with the 1-/2-D
  convolutions via :mod:`jigsaw.modules.spatial_halo`.

Either regime cooperates with the channel-parallel group through the shared
``config.channel_comm`` strategies
(:func:`jigsaw.modules.conv_channel_comm.over_channels`). In the halo regime the
halo exchange is handed to ``over_channels`` as a ``halo_fn``: the
``"all_gather"`` strategy applies it once to the gathered full-channel tensor,
while the streaming (``"p2p"`` / ``"all_to_all"``) strategies apply it to each
channel slab as the slabs travel around the ring.
"""

from typing import Any, Iterator, Optional

import torch
import torch.distributed as dist

from jigsaw.modules.conv_channel_comm import over_channels
from jigsaw.modules.spatial_halo import halo_gather, halo_scatter_add

# spatial axes owned by conv3d, as negative tensor indices: (D, H, W).
_SPATIAL_AXES = (-3, -2, -1)


def _split_axes(
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
) -> Iterator[tuple[int, int, dist.ProcessGroup]]:
    """Yield ``(spatial_idx, tensor_axis, group)`` for the axes actually split."""
    for idx, group in enumerate(spatial_groups):
        if group is not None and group.size() > 1:
            yield idx, _SPATIAL_AXES[idx], group


def _needs_halo(
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    padding: str | tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> bool:
    """Whether a nominally no-halo call (``stride == kernel``) still needs a halo.

    ``stride == kernel_size`` only tiles a split axis with no overlap when it
    also has zero padding and unit dilation. If *any* split axis has non-trivial
    padding or dilation, the local slices' interior edges would be wrongly
    zero-padded / their windows would overlap, so the halo path is required.
    """
    return isinstance(padding, tuple) and any(
        padding[idx] != 0 or dilation[idx] != 1
        for idx, _, _ in _split_axes(spatial_groups)
    )


# ---------------------------------------------------------------------------
# no-halo regime (stride == kernel_size)
# ---------------------------------------------------------------------------
def conv3d_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int, int],
    padding: str | tuple[int, int, int],
    dilation: tuple[int, int, int],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed 3d convolution for the no-halo regime (``stride == kernel``).

    Every spatial rank convolves its own domain slice independently -- this
    holds for a split along one axis or several -- while the channel group
    cooperates through ``config.channel_comm``.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
    """
    if _needs_halo(spatial_groups, padding, dilation):
        return conv3d_halo_nway(
            x, w, b, stride, padding, dilation, spatial_groups, channel_group, config
        )

    return over_channels(
        x,
        w,
        b,
        {"stride": stride, "padding": padding, "dilation": dilation},
        channel_group=channel_group,
        conv_func=config.conv3d_fn,
        n_spatial_dims=3,
        channel_comm=config.channel_comm,
    )


def conv3d_nway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    in_channels_global: int,
    local_in_spatial: tuple[int, ...],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
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
    if _needs_halo(spatial_groups, padding, dilation):
        return conv3d_halo_nway_backward(
            grad_output,
            x_aug,
            weight,
            stride,
            padding,
            dilation,
            in_channels_global,
            local_in_spatial,
            spatial_groups,
            channel_group,
            bayesian,
            config,
        )

    g_x = config.conv3d_input_fn(
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

    g_w = config.conv3d_weight_fn(
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
    g_x = g_x[..., start : start + shard, :, :, :]

    return g_x, g_w


# ---------------------------------------------------------------------------
# halo regime (stride != kernel_size)
#
# The halo exchange itself is dimension-agnostic and lives in
# :mod:`jigsaw.modules.spatial_halo` (shared with the 1-D/2-D convolutions). One
# halo exchange is run per split axis; the split axis's conv parameters are
# indexed out of the ``(d, h, w)`` tuples by ``spatial_idx``.
# ---------------------------------------------------------------------------
def _local_padding(
    padding: tuple[int, int, int],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
) -> tuple[int, int, int]:
    """Local-conv padding: split axes are 0 (materialised into the halo), else given."""
    split = {axis for _, axis, _ in _split_axes(spatial_groups)}
    local = [
        0 if _SPATIAL_AXES[idx] in split else pad for idx, pad in enumerate(padding)
    ]
    return (local[0], local[1], local[2])


def conv3d_halo_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int, int],
    padding: str | tuple[int, int, int],
    dilation: tuple[int, int, int],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed 3d convolution for the halo regime (``stride != kernel``).

    Builds a ``halo_fn`` that, for every split spatial axis, pulls the missing
    rows/columns from the other spatial ranks, and hands it to
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
            "Distributed Conv3d with stride != kernel_size does not support "
            f"string padding ({padding!r}); pass an explicit (d, h, w) padding."
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
        },
        channel_group=channel_group,
        conv_func=config.conv3d_fn,
        n_spatial_dims=3,
        channel_comm=config.channel_comm,
        halo_fn=halo_fn,
    )


def conv3d_halo_nway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    in_channels_global: int,
    local_in_spatial: tuple[int, ...],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    channel_group: dist.ProcessGroup,
    bayesian: bool,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward for the halo regime.

    ``g_w`` comes straight from the halo-padded input and this rank's
    ``grad_output`` (spatial ranks are summed by the parameter hook). ``g_x`` is
    computed on the halo-padded span, folded back onto the owning spatial ranks
    (:func:`halo_scatter_add`, once per split axis), reduced across the channel
    group (overlapped with the ``g_w`` compute) and finally sliced to this
    rank's channel shard.
    """
    local_padding = _local_padding(padding, spatial_groups)

    g_x_need = config.conv3d_input_fn(
        x_aug.shape[int(bayesian) :],
        weight,
        grad_output.to(weight.dtype),  # handles autocast
        stride=stride,
        padding=local_padding,
        dilation=dilation,
    ).contiguous()

    # Fold the halo-borrowed positions back onto their owners *before* the
    # channel-group reduce. halo_scatter_add (spatial groups) and the all-reduce
    # (channel group) act on independent group axes and are both linear, so they
    # commute; folding first shrinks the all-reduce payload from the halo-padded
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
    g_x = g_x.contiguous()

    g_x_handle = dist.all_reduce(
        g_x, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
    )

    g_w = config.conv3d_weight_fn(
        x_aug,
        weight.shape[int(bayesian) :],
        grad_output.to(weight.dtype),
        stride=stride,
        padding=local_padding,
        dilation=dilation,
    )

    g_x_handle.wait()

    shard = in_channels_global // channel_group.size()
    start = channel_group.rank() * shard
    g_x = g_x[..., start : start + shard, :, :, :]

    return g_x, g_w
