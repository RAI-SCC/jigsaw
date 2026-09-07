"""Distributed N-D transpose-convolution implementation.

One module serves ``F.conv_transpose1d`` / ``F.conv_transpose2d`` /
``F.conv_transpose3d``. Input is ``(N, C, *spatial)`` with
``len(spatial) == config.n_spatial``; the channels are split across the channel
group and the domain is split across the spatial group(s).

The spatial split is described by ``spatial_groups``: an ``n_spatial``-tuple
aligned with the trailing spatial axes, where an entry is ``None`` (or a group
of size 1) when that axis is not split. The default (a single group) splits the
last (width) axis only.

Two regimes are selected by
:class:`~jigsaw.modules.convolution.conv_transpose.config.ConvTransposeNdConfig`:

* ``stride == kernel_size`` (and no dilation/padding on the split axes): every
  spatial rank's raw output tiles the global output with no overlap, so the
  spatial split needs no communication (:func:`conv_transpose_nway`).
* ``stride != kernel_size``: neighbouring ranks' raw outputs overlap and must be
  summed, then cropped and re-sharded -- a "transpose halo" exchange
  (:func:`conv_transpose_halo_nway`), run once per split axis. That exchange is
  dimension-agnostic and lives in :mod:`jigsaw.modules.convolution.conv_transpose.halo`.

Both regimes cooperate with the channel-parallel group through
:func:`jigsaw.modules.convolution.conv_transpose.channel_comm.over_channels_transpose`, which
supports every ``channel_comm`` strategy and any group size. In the halo regime
the per-axis transpose-halo reduce is handed to ``over_channels_transpose`` as a
``post_fn``: the ``"all_gather"`` strategy applies it once to the fused output,
while the streaming (``"p2p"`` / ``"all_to_all"``) strategies apply it once to
the output accumulated slab by slab around the ring.
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
    """Local ``grad_input`` / ``grad_weight`` for a transpose convolution.

    ``torch.ops.aten.convolution_backward`` returns both at once; ``grad_bias``
    is handled separately (the spatial reduction is a parameter hook).
    ``output_mask`` selects which grads to actually compute -- a masked-off entry
    comes back as ``None`` -- so the caller can split the two into separate
    passes and overlap the second with the channel-group reduce of the first.
    """
    g_x, g_w, _ = config.conv_transpose_grad_fn(
        grad_output.to(weight.dtype),  # handles autocast
        x_full,
        weight,
        None,  # bias_sizes -- grad_bias handled separately
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


def _grad_input_weight_reduced(
    grad_output: torch.Tensor,
    x_full: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local ``grad_input`` / ``grad_weight`` with ``grad_input`` reduced over channels.

    ``grad_input`` needs a channel-group sum in the gather regime (each rank holds
    the grad for its own output-channel slice); it needs none in the
    ``channel_local`` regime (the rank owns whole conv-groups, so its ``g_x`` is
    already complete) or for a channel group of one. ``grad_weight`` never needs a
    channel reduction (the spatial reduction is a parameter hook). When a real
    channel-group reduce is pending -- and the op is not vmapped, where a masked
    ``convolution_backward`` output cannot be stacked -- the two grads are
    computed in separate passes so the grad-weight pass overlaps the
    reduce-scatter of grad_input; otherwise a single fused pass feeds the reduce.
    """
    reduce_pending = channel_group.size() > 1 and not config.channel_local
    if reduce_pending and not config.bayesian:
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
    padding: str | tuple[int, ...],
    dilation: tuple[int, ...],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed transpose conv for the no-halo regime (``stride == kernel``).

    Every spatial rank transpose-convolves its own domain slice independently --
    this holds for a split along one axis or several -- while the channel group
    cooperates through :func:`over_channels_transpose`. The split axes carry no
    padding/dilation so the raw outputs tile the global output exactly.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
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
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward for the no-halo regime.

    Each rank holds the full-channel grad wrt input for its own output-channel
    slice; summing those across the channel group and keeping this rank's channel
    slice yields ``g_x``. ``g_w`` needs no channel reduction (spatial ranks are
    reduced by the parameter hook).
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
        )

    return _grad_input_weight_reduced(
        grad_output, x_aug, weight, stride, padding, dilation, channel_group, config
    )


# ---------------------------------------------------------------------------
# halo regime (stride != kernel_size)
#
# The transpose halo exchange is dimension-agnostic and lives in
# :mod:`jigsaw.modules.convolution.conv_transpose.halo`. One exchange is run per split
# axis; the split axis's conv parameters are indexed out of the per-axis tuples
# by ``spatial_idx``.
# ---------------------------------------------------------------------------
def conv_transpose_halo_nway(
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
    """Distributed transpose conv for the halo regime (``stride != kernel``).

    Transpose-convolves the local domain slice with no split-axis padding, then
    for every split spatial axis folds the overlapping raw outputs onto their
    owning ranks (:func:`transpose_halo_reduce`) and adds the bias afterwards so
    overlapped positions are not double-counted. The fold is handed to
    :func:`over_channels_transpose` as a ``post_fn`` so it runs once on the
    channel-accumulated output, whichever ``config.channel_comm`` is in use.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
    """
    if not isinstance(padding, tuple):
        raise NotImplementedError(
            "Distributed ConvTransposeNd with stride != kernel_size does not "
            f"support string padding ({padding!r}); pass an explicit per-axis "
            "padding."
        )

    local_in_spatial = tuple(x.shape[-config.n_spatial :])

    def post_fn(y: torch.Tensor) -> torch.Tensor:
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
        post_fn=post_fn,
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
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward for the halo regime.

    ``grad_output`` (this rank's output shard) is expanded back to the raw local
    output span (:func:`transpose_halo_expand`, once per split axis), then the
    local transpose-conv backward gives the full-channel ``g_x`` and the local
    ``g_w``. ``g_x`` is reduce-scattered across the channel group to this rank's
    channel shard; ``g_w`` is reduced across spatial ranks by the parameter hook.
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
    )
