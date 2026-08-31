"""Distributed 2-D transpose-convolution implementation.

The 1-D and 3-D transpose modules
(:mod:`jigsaw.modules.conv1dtranspose.distributed_conv1dtranspose`,
:mod:`jigsaw.modules.conv3dtranspose.distributed_conv3dtranspose`) mirror this
one for ``F.conv_transpose1d`` / ``F.conv_transpose3d``. Input is ``(N, C, H, W)``;
the channels are split across the channel group and the domain is split across
the spatial group(s).

The spatial split is described by ``spatial_groups``: a 2-tuple
``(group_h, group_w)`` aligned with the ``(H, W)`` axes, where an entry is
``None`` (or a group of size 1) when that axis is not split. The default (a
single group) splits the width axis only.

Two regimes are selected by
:class:`~jigsaw.modules.conv2dtranspose.config.Conv2dTransposeConfig`:

* ``stride == kernel_size`` (and no dilation/padding on the split axes): every
  spatial rank's raw output tiles the global output with no overlap, so no
  spatial communication is needed (:func:`conv_transpose2d_nway`).
* otherwise: neighbouring ranks' raw outputs overlap and must be summed, then
  cropped and re-sharded -- a "transpose halo" exchange
  (:func:`conv_transpose2d_halo_nway`), run once per split axis. That exchange is
  dimension-agnostic and lives in :mod:`jigsaw.modules.spatial_halo_transpose`.

Either regime cooperates with the channel-parallel group through the shared
``config.channel_comm`` strategies
(:func:`jigsaw.modules.conv_transpose_channel_comm.over_channels_transpose`). In
the halo regime the per-axis transpose-halo reduce is handed to
``over_channels_transpose`` as a ``post_fn``: the ``"all_gather"`` strategy
applies it once to the fused output, while the streaming (``"p2p"`` /
``"all_to_all"``) strategies apply it once to the output accumulated slab by slab
around the ring.
"""

from typing import Any, Iterator, Optional

import torch
import torch.distributed as dist

from jigsaw.modules.conv_transpose_channel_comm import over_channels_transpose
from jigsaw.modules.spatial_halo_transpose import (
    transpose_halo_expand,
    transpose_halo_reduce,
)

# spatial axes owned by conv_transpose2d, as negative tensor indices: (H, W).
_SPATIAL_AXES = (-2, -1)


def _split_axes(
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
) -> Iterator[tuple[int, int, dist.ProcessGroup]]:
    """Yield ``(spatial_idx, tensor_axis, group)`` for the axes actually split."""
    for idx, group in enumerate(spatial_groups):
        if group is not None and group.size() > 1:
            yield idx, _SPATIAL_AXES[idx], group


def _grad_input_weight(
    grad_output: torch.Tensor,
    x_full: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local ``grad_input`` / ``grad_weight`` for a transpose convolution."""
    g_x, g_w, _ = config.conv_transpose2d_grad_fn(
        grad_output.to(weight.dtype),  # handles autocast
        x_full,
        weight,
        None,  # bias_sizes -- grad_bias handled separately
        list(stride),
        list(padding),
        list(dilation),
        True,  # transposed
        [0, 0],  # output_padding
        1,  # groups
        [True, True, False],  # (grad_input, grad_weight, grad_bias)
    )
    return g_x, g_w


def _channel_shard(
    g_x: torch.Tensor, in_channels_global: int, channel_group: dist.ProcessGroup
) -> torch.Tensor:
    """Keep this rank's contiguous input-channel slice of a full-channel grad."""
    shard = in_channels_global // channel_group.size()
    start = channel_group.rank() * shard
    return g_x[..., start : start + shard, :, :]


def _needs_halo(
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
) -> bool:
    """Whether a nominally no-halo call (``stride == kernel``) still needs a halo.

    ``stride == kernel_size`` only tiles a split axis with no overlap when it
    also has zero padding and unit dilation. If *any* split axis has non-trivial
    padding or dilation, the raw local outputs overlap / are cropped and the halo
    path is required.
    """
    return isinstance(padding, tuple) and any(
        padding[idx] != 0 or dilation[idx] != 1
        for idx, _, _ in _split_axes(spatial_groups)
    )


# ---------------------------------------------------------------------------
# no-halo regime (stride == kernel_size)
# ---------------------------------------------------------------------------
def conv_transpose2d_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed transpose conv for the no-halo regime (``stride == kernel``).

    Every spatial rank transpose-convolves its own domain slice independently --
    this holds for a split along one axis or several -- while the channel group
    cooperates through ``config.channel_comm``. The split axes carry no
    padding/dilation so the raw outputs tile the global output exactly.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
    """
    if _needs_halo(spatial_groups, padding, dilation):
        return conv_transpose2d_halo_nway(
            x, w, b, stride, padding, dilation, spatial_groups, channel_group, config
        )

    return over_channels_transpose(
        x,
        w,
        b,
        {"stride": stride, "padding": padding, "dilation": dilation},
        channel_group=channel_group,
        conv_func=config.conv2dtranspose_fn,
        n_spatial_dims=2,
        channel_comm=config.channel_comm,
    )


def conv_transpose2d_nway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
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
        return conv_transpose2d_halo_nway_backward(
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

    g_x, g_w = _grad_input_weight(
        grad_output, x_aug, weight, stride, padding, dilation, config
    )
    g_x = g_x.contiguous()
    g_x_handle = dist.all_reduce(
        g_x, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
    )
    g_x_handle.wait()

    return _channel_shard(g_x, in_channels_global, channel_group), g_w


# ---------------------------------------------------------------------------
# halo regime (stride != kernel_size)
#
# The transpose halo exchange is dimension-agnostic and lives in
# :mod:`jigsaw.modules.spatial_halo_transpose`. One exchange is run per split
# axis; the split axis's conv parameters are indexed out of the ``(h, w)`` tuples
# by ``spatial_idx``.
# ---------------------------------------------------------------------------
def _local_padding(
    padding: tuple[int, int],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
) -> tuple[int, int]:
    """Local-conv padding: split axes are 0 (folded into the halo), else given."""
    split = {axis for _, axis, _ in _split_axes(spatial_groups)}
    local = [
        0 if _SPATIAL_AXES[idx] in split else pad for idx, pad in enumerate(padding)
    ]
    return (local[0], local[1])


def conv_transpose2d_halo_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed transpose conv for the halo regime (``stride != kernel``).

    Transpose-convolves the local domain slice with no split-axis padding, then
    for every split spatial axis folds the overlapping raw outputs onto their
    owning ranks (:func:`transpose_halo_reduce`) and adds the bias afterwards so
    overlapped positions are not double-counted. The fold is handed to
    ``over_channels_transpose`` as a ``post_fn`` so it runs once on the
    channel-accumulated output, whichever ``config.channel_comm`` is in use.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
    """
    if not isinstance(padding, tuple):
        raise NotImplementedError(
            "Distributed Conv2dTranspose with stride != kernel_size does not "
            f"support string padding ({padding!r}); pass an explicit (h, w) "
            "padding."
        )

    local_in_spatial = tuple(x.shape[-2:])

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
        },
        channel_group=channel_group,
        conv_func=config.conv2dtranspose_fn,
        n_spatial_dims=2,
        channel_comm=config.channel_comm,
        post_fn=post_fn,
    )


def conv_transpose2d_halo_nway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    in_channels_global: int,
    local_in_spatial: tuple[int, ...],
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    channel_group: dist.ProcessGroup,
    bayesian: bool,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward for the halo regime.

    ``grad_output`` (this rank's output shard) is expanded back to the raw local
    output span (:func:`transpose_halo_expand`, once per split axis), then the
    local transpose-conv backward gives the full-channel ``g_x`` and the local
    ``g_w``. ``g_x`` is reduced across the channel group and sliced to this
    rank's channel shard; ``g_w`` is reduced across spatial ranks by the
    parameter hook.
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

    g_x, g_w = _grad_input_weight(
        grad_raw,
        x_aug,
        weight,
        stride,
        _local_padding(padding, spatial_groups),
        dilation,
        config,
    )
    g_x = g_x.contiguous()
    g_x_handle = dist.all_reduce(
        g_x, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
    )
    g_x_handle.wait()

    return _channel_shard(g_x, in_channels_global, channel_group), g_w
