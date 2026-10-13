"""Channel-parallel input exchange of the distributed convolution.

The input channels are split over the channel group. The strategy
``config.channel_comm`` selects how the full-channel input is assembled:

* ``"all_gather"`` gathers all slices at once.
* ``"p2p"`` circulates the slices in a ring with batched ``isend`` and
  ``irecv``, overlapping each partial convolution with the next transfer.
* ``"all_to_all"`` uses the same ring with masked all-to-all hops (NCCL only).
"""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from .._common.channel_comm import (
    ConvFn,
    _HopFn,
    _in_channel_block,
    _resolve_channel_hop,
    gather_full_channels,
)

# A split-axis halo exchange: ``fn(tensor) -> halo_padded_tensor``.
HaloFn = Callable[[torch.Tensor], torch.Tensor]


def _over_channels_all_gather(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    conv_kwargs: dict[str, Any],
    channel_group: dist.ProcessGroup,
    conv_func: ConvFn,
    channel_dim: int,
    halo_fn: Optional[HaloFn],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather all channel slices, then convolve once.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Output and the full-channel input kept for the backward pass.
    """
    x_full = gather_full_channels(x.contiguous(), channel_group, channel_dim)
    if halo_fn is not None:
        x_full = halo_fn(x_full)

    if b is not None:
        y = conv_func(x_full, w, b, **conv_kwargs)
    else:
        y = conv_func(x_full, w, **conv_kwargs)
    return y, x_full


def _over_channels_streaming(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    conv_kwargs: dict[str, Any],
    channel_group: dist.ProcessGroup,
    conv_func: ConvFn,
    channel_dim: int,
    halo_fn: Optional[HaloFn],
    hop_fn: _HopFn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Circulate the channel slices in a ring and convolve each one.

    ``hop_fn`` is the communication function of one hop.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Output and the full-channel input kept for the backward pass.
    """
    group_size = channel_group.size()
    rank = channel_group.rank()

    x = x.contiguous()
    c_in_local = x.shape[channel_dim]
    c_in_global = w.shape[channel_dim]

    def _prep(slab: torch.Tensor) -> torch.Tensor:
        return slab if halo_fn is None else halo_fn(slab)

    def _conv(span: torch.Tensor, block: int, with_bias: bool) -> torch.Tensor:
        w_block = _in_channel_block(w, channel_dim, block, c_in_local)
        if with_bias and b is not None:
            return conv_func(span, w_block, b, **conv_kwargs)
        return conv_func(span, w_block, **conv_kwargs)

    own = _prep(x)
    if group_size == 1:
        return _conv(own, rank, with_bias=True), own

    x_full = torch.empty(
        (*own.shape[:channel_dim], c_in_global, *own.shape[channel_dim + 1 :]),
        dtype=own.dtype,
        device=own.device,
    )
    x_full.narrow(channel_dim, rank * c_in_local, c_in_local).copy_(own)

    buf_a = torch.empty_like(x)
    buf_b = torch.empty_like(x)
    recv_buf = buf_a

    # Prime the ring: this rank's own slice sets off while we convolve it.
    recv_buf, handle = hop_fn(
        x, recv_buf, direction=1, group=channel_group, async_op=True
    )
    y = _conv(own, rank, with_bias=True)

    for step in range(1, group_size):
        if handle is not None:
            handle.wait()
        slab, recv_buf = recv_buf, (buf_b if recv_buf is buf_a else buf_a)

        block = (rank - step) % group_size

        # Relay before the (possibly collective) halo + partial conv, so they
        # overlap the next hop.
        if step < group_size - 1:
            recv_buf, handle = hop_fn(
                slab, recv_buf, direction=1, group=channel_group, async_op=True
            )

        span = _prep(slab)
        x_full.narrow(channel_dim, block * c_in_local, c_in_local).copy_(span)
        y = y + _conv(span, block, with_bias=False)

    return y, x_full


def _over_channels_local(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    conv_kwargs: dict[str, Any],
    conv_func: ConvFn,
    halo_fn: Optional[HaloFn],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convolve the local channel shard without channel communication.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Output and the input shard kept for the backward pass.
    """
    span = x.contiguous()
    if halo_fn is not None:
        span = halo_fn(span)
    if b is not None:
        y = conv_func(span, w, b, **conv_kwargs)
    else:
        y = conv_func(span, w, **conv_kwargs)
    return y, span


def over_channels(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    conv_kwargs: dict[str, Any],
    *,
    channel_group: dist.ProcessGroup,
    conv_func: ConvFn,
    n_spatial_dims: int,
    channel_comm: str,
    halo_fn: Optional[HaloFn] = None,
    local: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convolve over the channel-parallel group with the configured strategy.

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
        split axis is applied by ``halo_fn`` and must be ``0`` here.
    channel_group : dist.ProcessGroup
        Process group splitting the channels.
    conv_func : ConvFn
        Convolution primitive.
    n_spatial_dims : int
        Number of spatial dimensions.
    channel_comm : str
        Channel communication strategy: ``"all_gather"``, ``"p2p"`` or
        ``"all_to_all"``.
    halo_fn : HaloFn, optional
        Halo exchange applied to the input.
    local : bool
        Whether each rank owns whole convolution groups, so that no channel
        communication is needed.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Output and the input kept for the backward pass.
    """
    channel_dim = -(n_spatial_dims + 1)
    if local or channel_group.size() == 1:
        return _over_channels_local(x, w, b, conv_kwargs, conv_func, halo_fn)
    if channel_comm == "all_gather":
        return _over_channels_all_gather(
            x, w, b, conv_kwargs, channel_group, conv_func, channel_dim, halo_fn
        )
    hop_fn = _resolve_channel_hop(channel_comm, channel_group)
    return _over_channels_streaming(
        x, w, b, conv_kwargs, channel_group, conv_func, channel_dim, halo_fn, hop_fn
    )
