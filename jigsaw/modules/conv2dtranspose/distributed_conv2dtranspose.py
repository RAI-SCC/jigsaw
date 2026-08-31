"""Distributed 2-D transpose-convolution implementation.

The 1-D and 3-D transpose modules
(:mod:`jigsaw.modules.conv1dtranspose.distributed_conv1dtranspose`,
:mod:`jigsaw.modules.conv3dtranspose.distributed_conv3dtranspose`) mirror this
one for ``F.conv_transpose1d`` / ``F.conv_transpose3d``. Input is ``(N, C, H, W)``;
the channels are split across the channel group and the domain is split along the
width axis (the last axis) across the spatial group.

Two regimes are selected by
:class:`~jigsaw.modules.conv2dtranspose.config.Conv2dTransposeConfig`:

* ``stride == kernel_size`` (and no dilation/padding on the split axis): every
  spatial rank's raw output tiles the global output with no overlap, so no
  spatial communication is needed (:func:`conv_transpose2d_nway`).
* otherwise: neighbouring ranks' raw outputs overlap and must be summed, then
  cropped and re-sharded -- a "transpose halo" exchange
  (:func:`conv_transpose2d_halo_nway`). That exchange is dimension-agnostic and
  lives in :mod:`jigsaw.modules.spatial_halo_transpose`.

Either regime cooperates with the channel-parallel group through the shared
``config.channel_comm`` strategies
(:func:`jigsaw.modules.conv_transpose_channel_comm.over_channels_transpose`). In
the halo regime the transpose-halo reduce is handed to ``over_channels_transpose``
as a ``post_fn``: the ``"all_gather"`` strategy applies it once to the fused
output, while the streaming (``"p2p"`` / ``"all_to_all"``) strategies apply it
once to the output accumulated slab by slab around the ring.
"""

from typing import Any

import torch
import torch.distributed as dist

from jigsaw.modules.conv_transpose_channel_comm import over_channels_transpose
from jigsaw.modules.spatial_halo_transpose import (
    transpose_halo_expand,
    transpose_halo_reduce,
)


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
    spatial_group: dist.ProcessGroup,
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
) -> bool:
    """Whether a nominally no-halo call (``stride == kernel``) still needs a halo.

    ``stride == kernel_size`` only tiles the split (width) axis with no overlap
    when it also has zero padding and unit dilation. With a real spatial split
    and either of those non-trivial, the raw local outputs overlap / are cropped
    and the halo path is required.
    """
    return (
        spatial_group.size() > 1
        and isinstance(padding, tuple)
        and (padding[1] != 0 or dilation[1] != 1)
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
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed transpose conv for the no-halo regime (``stride == kernel``).

    Every spatial rank transpose-convolves its own width slice independently
    while the channel group cooperates through ``config.channel_comm``. The split
    axis carries no padding/dilation so the raw outputs tile the global output
    exactly.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
    """
    if _needs_halo(spatial_group, padding, dilation):
        return conv_transpose2d_halo_nway(
            x, w, b, stride, padding, dilation, spatial_group, channel_group, config
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
        return conv_transpose2d_halo_nway_backward(
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
# :mod:`jigsaw.modules.spatial_halo_transpose`; only the split axis's conv
# parameters are threaded through here (``*[1]`` -- the width).
# ---------------------------------------------------------------------------
def _width_padding(padding: tuple[int, int]) -> tuple[int, int]:
    """Local-conv padding: height as given, width folded into the halo reduce."""
    return (padding[0], 0)


def conv_transpose2d_halo_nway(
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
    """Distributed transpose conv for the halo regime (``stride != kernel``).

    Transpose-convolves the local width slice with no split-axis padding, then
    folds the overlapping raw outputs onto their owning ranks
    (:func:`transpose_halo_reduce`) and adds the bias afterwards so overlapped
    columns are not double-counted. The fold is handed to
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

    local_in_width = x.shape[-1]

    def post_fn(y_raw: torch.Tensor) -> torch.Tensor:
        return transpose_halo_reduce(
            y_raw,
            spatial_group,
            local_in_width,
            w.shape[-1],
            stride[1],
            padding[1],
            dilation[1],
        )

    return over_channels_transpose(
        x,
        w,
        b,
        {
            "stride": stride,
            "padding": _width_padding(padding),
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
    local_in_width: int,
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    bayesian: bool,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward for the halo regime.

    ``grad_output`` (this rank's output shard) is expanded back to the raw local
    output span (:func:`transpose_halo_expand`), then the local transpose-conv
    backward gives the full-channel ``g_x`` and the local ``g_w``. ``g_x`` is
    reduced across the channel group and sliced to this rank's channel shard;
    ``g_w`` is reduced across spatial ranks by the parameter hook.
    """
    grad_raw = transpose_halo_expand(
        grad_output,
        spatial_group,
        x_aug.shape[-1],
        weight.shape[-1],
        stride[1],
        padding[1],
        dilation[1],
    )

    g_x, g_w = _grad_input_weight(
        grad_raw, x_aug, weight, stride, _width_padding(padding), dilation, config
    )
    g_x = g_x.contiguous()
    g_x_handle = dist.all_reduce(
        g_x, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
    )
    g_x_handle.wait()

    return _channel_shard(g_x, in_channels_global, channel_group), g_w
