"""Distributed N-D convolution implementation.

One module serves ``F.conv1d`` / ``F.conv2d`` / ``F.conv3d``. Input is
``(N, C, *spatial)`` with ``len(spatial) == config.n_spatial``; the domain is
split across the spatial group(s) and the channels are split across the channel
group.

The spatial split is described by ``spatial_groups``: an ``n_spatial``-tuple
aligned with the trailing spatial axes, where an entry is ``None`` (or a group
of size 1) when that axis is not split. The default (a single group) splits the
last (width) axis only.

Two regimes are selected by :class:`~jigsaw.modules.convolution.conv.config.ConvNdConfig`:

* ``stride == kernel_size`` (and no dilation/padding on the split axes): the
  spatial split needs no communication -- every spatial rank convolves its own
  domain slice independently (:func:`conv_nway`).
* ``stride != kernel_size``: neighbouring output positions share input
  positions, so each spatial rank pulls the missing rows/columns from the other
  ranks with a halo exchange -- one per split axis -- and convolves
  (:func:`conv_halo_nway`). The halo exchange is dimension-agnostic and lives in
  :mod:`jigsaw.modules.convolution.conv.halo`.

Both regimes cooperate with the channel-parallel group through
:func:`jigsaw.modules.convolution.conv.channel_comm.over_channels`, which supports every
``channel_comm`` strategy and any group size. In the halo regime the halo
exchange is handed to ``over_channels`` as a ``halo_fn``: the ``"all_gather"``
strategy applies it once to the gathered full-channel tensor, while the
streaming (``"p2p"`` / ``"all_to_all"``) strategies apply it to each channel
slab as the slabs travel around the ring, overlapping it with the next hop.
"""

from itertools import groupby
from typing import Any, Optional

import torch
import torch.distributed as dist

from ..common.channel_comm import conv_group_blocks
from ..common.spatial_split import (
    _local_padding,
    _needs_halo,
    _reduce_scatter_channels,
    _split_axes,
)
from .channel_comm import over_channels
from .halo import halo_gather, halo_scatter_add


def _group_blocks(config: Any) -> Optional[list[tuple[int, int, int, int]]]:
    """Per-conv-group channel blocks for the ``channel_grouped`` regime, else ``None``."""
    if not config.channel_grouped:
        return None
    return conv_group_blocks(
        config.out_channels,
        config.in_channels,
        config.groups,
        config.channel_rank,
        config.channel_size,
    )


def _grouped_conv_grads(
    grad_output: torch.Tensor,
    x_full: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``channel_grouped`` backward: full-channel ``g_x`` and local ``g_w``, run by run.

    Mirrors :func:`_over_channels_grouped_gather`: the conv-groups the local
    output shard touches are differentiated against their full input-channel
    slice and this rank's local output-channel slice. Every such block spans the
    same ``C_in / groups`` input channels and -- except, at most, the two
    shard-boundary blocks whose conv-group is only partly owned here -- the same
    ``C_out / groups`` output channels, and consecutive blocks are contiguous on
    both axes. So each maximal run of equal-output-width blocks is one native
    grouped (``groups = len(run)``) grad-input / grad-weight call: at most three
    calls of each, versus one per block. ``g_x`` comes back shaped like
    ``x_full`` (the gathered, and in the halo regime halo-padded, full-channel
    input) with the untouched input-channel slices left at zero -- the caller
    reduce-scatters it over the channel group. ``g_w`` is this rank's full local
    weight.
    """
    channel_dim = -(config.n_spatial + 1)
    g_x = x_full.new_zeros(x_full.shape)
    g_w = weight.new_zeros(weight.shape)
    blocks = _group_blocks(config)
    assert blocks is not None  # only called in the channel_grouped regime
    for _, run_iter in groupby(blocks, key=lambda blk: blk[1] - blk[0]):
        run = list(run_iter)
        out_lo, in_lo = run[0][0], run[0][2]
        out_hi, in_hi = run[-1][1], run[-1][3]
        n_groups = len(run)
        xg = x_full.narrow(channel_dim, in_lo, in_hi - in_lo)
        go = grad_output.narrow(channel_dim, out_lo, out_hi - out_lo).to(weight.dtype)
        wg = weight[out_lo:out_hi]
        g_x.narrow(channel_dim, in_lo, in_hi - in_lo).copy_(
            config.conv_input_fn(
                xg.shape,
                wg,
                go,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=n_groups,
            )
        )
        g_w[out_lo:out_hi] = config.conv_weight_fn(
            xg,
            wg.shape,
            go,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=n_groups,
        )
    return g_x, g_w


def _channel_reduce_grad_input(
    g_x: torch.Tensor, channel_group: dist.ProcessGroup, config: Any
) -> tuple[torch.Tensor, Optional[dist.Work]]:
    """Sum the grad wrt input across the channel group, keeping this rank's shard.

    In the ``channel_local`` regime (every rank owns whole conv-groups) the
    forward did no channel exchange, so ``g_x`` is already this rank's complete
    channel shard and no reduction is needed. Otherwise this is the
    :func:`_reduce_scatter_channels` over the full-channel grad.
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
    """Distributed N-D convolution for the no-halo regime (``stride == kernel``).

    Every spatial rank convolves its own domain slice independently -- this
    holds for a split along one axis or several -- while the channel group
    cooperates through :func:`jigsaw.modules.convolution.conv.channel_comm.over_channels`.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
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
        group_blocks=_group_blocks(config),
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
    """Backward for the no-halo regime.

    Each rank holds the full-channel grad wrt input for its own output-channel
    slice; summing those across the channel group and keeping this rank's channel
    slice yields ``g_x``. ``g_w`` needs no channel reduction (spatial ranks are
    reduced by the parameter hook).
    """
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

    if config.channel_grouped:
        g_x, g_w = _grouped_conv_grads(
            grad_output, x_aug, weight, stride, padding, dilation, config
        )
        g_x, g_x_handle = _channel_reduce_grad_input(g_x, channel_group, config)
        if g_x_handle is not None:
            g_x_handle.wait()
        return g_x, g_w

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
    """Distributed N-D convolution for the halo regime (``stride != kernel``).

    Builds a ``halo_fn`` that, for every split spatial axis, pulls the missing
    rows/columns from the other spatial ranks, and hands it to
    :func:`~jigsaw.modules.convolution.conv.channel_comm.over_channels` together with the
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
        group_blocks=_group_blocks(config),
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
    """Backward for the halo regime.

    ``g_w`` comes straight from the halo-padded input and this rank's
    ``grad_output`` (spatial ranks are summed by the parameter hook); it depends
    on nothing the halo exchange produces, so it is enqueued first and runs
    alongside the halo exchange rather than after it. ``g_x`` is computed on the
    halo-padded span, folded back onto the owning spatial ranks
    (:func:`halo_scatter_add`, once per split axis), reduced+scattered across the
    channel group and returned as this rank's channel shard.
    """
    local_padding = _local_padding(padding, spatial_groups)

    if config.channel_grouped:
        g_x_need, g_w = _grouped_conv_grads(
            grad_output, x_aug, weight, stride, local_padding, dilation, config
        )
        g_x_need = g_x_need.contiguous()
    else:
        g_x_need = config.conv_input_fn(
            x_aug.shape[int(bayesian) :],
            weight,
            grad_output.to(weight.dtype),  # handles autocast
            stride=stride,
            padding=local_padding,
            dilation=dilation,
            groups=config.groups_local,
        ).contiguous()

        # Enqueue grad-weight before the (spatial-group) halo exchange below so
        # the two overlap instead of running back to back.
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
