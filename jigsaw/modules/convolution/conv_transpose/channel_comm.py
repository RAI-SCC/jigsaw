"""Channel-parallel input exchange for the distributed 1/2/3-D transpose convs.

The transpose analogue of :mod:`jigsaw.modules.convolution.conv.channel_comm`.
Each rank owns a contiguous ``C_in / G`` slice of the input channels and the
matching slice of the weight's *output* channels (the weight still carries every
input channel, on its leading axis). Three interchangeable strategies reassemble
the full-channel input and produce this rank's output-channel slice, selected by
``config.channel_comm``:

* ``"all_gather"`` -- gather every rank's input-channel slice in one shot, then
  run a single fused transpose convolution.
* ``"p2p"`` -- circulate the slices one hop at a time with batched ``isend`` /
  ``irecv`` (:func:`jigsaw.comm.comm_1d_p2p`), overlapping each partial transpose
  convolution with the next transfer.
* ``"all_to_all"`` -- the same ring schedule, each hop a masked all-to-all
  (:func:`jigsaw.comm.comm_1d_alltoall`). Needs a backend with all-to-all support
  (NCCL); gloo has none.

Two axes differ from the plain-convolution version. The input-channel axis of the
transpose weight is its *leading* axis, at ``-(n_spatial_dims + 2)`` (the plain
weight keeps it at ``-(n_spatial_dims + 1)``). And the split-axis spatial
exchange happens *after* the convolution -- the raw local outputs of neighbouring
ranks overlap and are folded back with :func:`transpose_halo_reduce` -- so it is
passed as a ``post_fn`` applied to the accumulated output rather than a
``halo_fn`` applied to the input. ``transpose_halo_reduce`` is linear, so folding
once after the channel sum matches folding each slab. The bias is likewise added
after ``post_fn`` so overlapped positions are not double-counted.

:func:`jigsaw.modules.convolution.common.channel_comm.resolve_channel_comm` turns
the user-facing ``"auto"`` into a concrete name.
"""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from ..common.channel_comm import (
    _HopFn,
    _in_channel_block,
    _resolve_channel_hop,
    gather_full_channels,
)

# A transpose-conv primitive:
# ``fn(x, w[, b], stride=, padding=, dilation=) -> y``.
ConvTransposeFn = Callable[..., torch.Tensor]

# A post-convolution op on the accumulated output, e.g. the transpose-halo
# reduce: ``fn(y_raw) -> y``.
PostFn = Callable[[torch.Tensor], torch.Tensor]

# ``_HopFn`` (per-hop transport type), ``_resolve_channel_hop`` (streaming
# ``channel_comm`` -> transport) and ``_in_channel_block`` (weight input-channel
# slice) are identical to the plain-conv versions and imported from there.


def _bias_view(b: torch.Tensor, n_spatial_dims: int) -> torch.Tensor:
    """Reshape the bias to broadcast against the conv output.

    ``b`` is the raw bias parameter -- ``(C_out,)`` normally, or ``(S, C_out)``
    under ``vmap``. Its channel axis has to land at ``-(n_spatial_dims + 1)`` of
    the output: a singleton batch axis ahead of it, ``n_spatial_dims`` singleton
    spatial axes behind, and any leading ``vmap`` axis kept untouched -- the
    mirror of :meth:`ConvTransposeNdConfig._sum_grad_bias`. A plain ``(-1, 1,
    ..., 1)`` view would fold the ``vmap`` axis into the channels instead.
    """
    return b.view(*b.shape[:-1], 1, b.shape[-1], *((1,) * n_spatial_dims))


def _over_channels_all_gather(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    conv_kwargs: dict[str, Any],
    channel_group: dist.ProcessGroup,
    conv_func: ConvTransposeFn,
    channel_dim: int,
    n_spatial_dims: int,
    post_fn: Optional[PostFn],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``channel_comm="all_gather"``: gather every slice, then one fused transpose conv.

    Returns ``(y, x_full)`` -- ``x_full`` is the gathered full-channel input kept
    for the backward pass.
    """
    x_full = gather_full_channels(x.contiguous(), channel_group, channel_dim)

    y = conv_func(x_full, w, **conv_kwargs)
    if post_fn is not None:
        y = post_fn(y)
    if b is not None:
        y = y + _bias_view(b, n_spatial_dims)
    return y, x_full


def _over_channels_streaming(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    conv_kwargs: dict[str, Any],
    channel_group: dist.ProcessGroup,
    conv_func: ConvTransposeFn,
    channel_dim: int,
    in_dim: int,
    n_spatial_dims: int,
    post_fn: Optional[PostFn],
    hop_fn: _HopFn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ring channel-parallel transpose conv; ``hop_fn`` is the per-step transport.

    The channel slices travel one hop per step around the ring while each partial
    transpose convolution overlaps the next transfer. ``post_fn`` (the split-axis
    halo reduce) and the bias are applied once to the accumulated output.

    Returns ``(y, x_full)`` where ``x_full`` is the full-channel input,
    reassembled slab by slab, kept for the backward pass.
    """
    group_size = channel_group.size()
    rank = channel_group.rank()

    x = x.contiguous()
    c_in_local = x.shape[channel_dim]
    c_in_global = w.shape[in_dim]

    def _conv(slab: torch.Tensor, block: int) -> torch.Tensor:
        return conv_func(
            slab, _in_channel_block(w, in_dim, block, c_in_local), **conv_kwargs
        )

    def _finish(y_raw: torch.Tensor) -> torch.Tensor:
        y = post_fn(y_raw) if post_fn is not None else y_raw
        if b is not None:
            y = y + _bias_view(b, n_spatial_dims)
        return y

    if group_size == 1:
        return _finish(_conv(x, rank)), x

    x_full = torch.empty(
        (*x.shape[:channel_dim], c_in_global, *x.shape[channel_dim + 1 :]),
        dtype=x.dtype,
        device=x.device,
    )
    x_full.narrow(channel_dim, rank * c_in_local, c_in_local).copy_(x)

    buf_a = torch.empty_like(x)
    buf_b = torch.empty_like(x)
    recv_buf = buf_a

    # Prime the ring: this rank's own slice sets off while we convolve it.
    recv_buf, handle = hop_fn(
        x, recv_buf, direction=1, group=channel_group, async_op=True
    )
    y_raw = _conv(x, rank)

    for step in range(1, group_size):
        if handle is not None:
            handle.wait()
        slab, recv_buf = recv_buf, (buf_b if recv_buf is buf_a else buf_a)

        block = (rank - step) % group_size

        # Relay before the partial conv so it overlaps the next hop.
        if step < group_size - 1:
            recv_buf, handle = hop_fn(
                slab, recv_buf, direction=1, group=channel_group, async_op=True
            )

        x_full.narrow(channel_dim, block * c_in_local, c_in_local).copy_(slab)
        y_raw = y_raw + _conv(slab, block)

    return _finish(y_raw), x_full


def _over_channels_local(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    conv_kwargs: dict[str, Any],
    conv_func: ConvTransposeFn,
    n_spatial_dims: int,
    post_fn: Optional[PostFn],
) -> tuple[torch.Tensor, torch.Tensor]:
    """No channel exchange: transpose-convolve this rank's channel shard directly.

    Used when the channel group is size one, or when it owns whole conv-groups
    (``conv_kwargs["groups"]`` carries ``groups_local``). ``w`` is this rank's
    shard and already carries only the input channels it needs. ``post_fn`` (the
    split-axis halo reduce) and the bias are applied once to the output, matching
    the streaming path. Returns ``(y, x)`` -- ``x`` is kept for the backward pass.
    """
    span = x.contiguous()
    y = conv_func(span, w, **conv_kwargs)
    if post_fn is not None:
        y = post_fn(y)
    if b is not None:
        y = y + _bias_view(b, n_spatial_dims)
    return y, span


def over_channels_transpose(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    conv_kwargs: dict[str, Any],
    *,
    channel_group: dist.ProcessGroup,
    conv_func: ConvTransposeFn,
    n_spatial_dims: int,
    channel_comm: str,
    post_fn: Optional[PostFn] = None,
    local: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-parallel transpose convolution for any ``channel_comm`` and dimensionality.

    ``conv_kwargs`` holds ``stride`` / ``padding`` / ``dilation`` (and ``groups``)
    for ``conv_func``; in the halo regime the split-axis padding is folded into
    ``post_fn`` so pass ``0`` for that axis. When ``local`` is set (this rank owns
    whole conv-groups) or the channel group is size one, no exchange happens --
    see :func:`_over_channels_local`. Otherwise all strategies return the same
    ``(y, x_full)`` -- see :func:`_over_channels_all_gather` /
    :func:`_over_channels_streaming`.
    """
    channel_dim = -(n_spatial_dims + 1)
    in_dim = -(n_spatial_dims + 2)
    if local or channel_group.size() == 1:
        return _over_channels_local(
            x, w, b, conv_kwargs, conv_func, n_spatial_dims, post_fn
        )
    if channel_comm == "all_gather":
        return _over_channels_all_gather(
            x,
            w,
            b,
            conv_kwargs,
            channel_group,
            conv_func,
            channel_dim,
            n_spatial_dims,
            post_fn,
        )
    hop_fn = _resolve_channel_hop(channel_comm, channel_group)
    return _over_channels_streaming(
        x,
        w,
        b,
        conv_kwargs,
        channel_group,
        conv_func,
        channel_dim,
        in_dim,
        n_spatial_dims,
        post_fn,
        hop_fn,
    )
