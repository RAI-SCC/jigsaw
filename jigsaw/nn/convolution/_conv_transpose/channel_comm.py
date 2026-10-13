"""Channel-parallel input exchange of the distributed transpose convolution.

See :mod:`jigsaw.nn.convolution._conv.channel_comm`. The spatial halo
reduction is applied to the accumulated output through ``post_fn``, and the bias
is added afterwards.
"""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from .._common.channel_comm import (
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
    """Reshape the bias to broadcast against the output."""
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
    """Gather all channel slices, then apply the transpose convolution once.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Output and the full-channel input kept for the backward pass.
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
    """Circulate the channel slices in a ring and apply the transpose convolution to each.

    ``hop_fn`` is the communication function of one hop. ``post_fn`` and the bias
    are applied to the accumulated output.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Output and the full-channel input kept for the backward pass.
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
    """Apply the transpose convolution to the local channel shard without communication.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Output and the input kept for the backward pass.
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
    """Apply the transpose convolution over the channel-parallel group.

    Parameters
    ----------
    x : torch.Tensor
        Local input.
    w : torch.Tensor
        Local weight.
    b : torch.Tensor | None
        Local bias.
    conv_kwargs : dict[str, Any]
        Keyword arguments of ``conv_func``. In the halo case the padding of the
        split axis is applied by ``post_fn`` and must be ``0`` here.
    channel_group : dist.ProcessGroup
        Process group splitting the channels.
    conv_func : ConvTransposeFn
        Transpose convolution primitive.
    n_spatial_dims : int
        Number of spatial dimensions.
    channel_comm : str
        Channel communication strategy: ``"all_gather"``, ``"p2p"`` or
        ``"all_to_all"``.
    post_fn : PostFn, optional
        Operation applied to the accumulated output, such as the halo reduction.
    local : bool
        Whether each rank owns whole convolution groups, so that no channel
        communication is needed.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Output and the input kept for the backward pass.
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
