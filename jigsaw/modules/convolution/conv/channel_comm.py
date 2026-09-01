"""Channel-parallel input exchange for the distributed 1/2/3-D convolutions.

Each rank owns a contiguous ``C_in / G`` slice of the input channels and the
matching slice of the weight's *output* channels (the weight still carries every
input channel). Three interchangeable strategies reassemble the full-channel
input and produce this rank's output-channel slice, selected by
``config.channel_comm``:

* ``"all_gather"`` -- gather every rank's input-channel slice in one shot, then
  run a single fused convolution. Simple and fast, but its transient memory peak
  is ``~2 * C_in`` (the gather list plus the concatenated copy).
* ``"p2p"`` -- circulate the slices one hop at a time with batched ``isend`` /
  ``irecv`` (:func:`jigsaw.comm.comm_1d_p2p`), overlapping each partial
  convolution with the next transfer, keeping the peak to ``C_in`` plus two
  slice buffers.
* ``"all_to_all"`` -- the same ring schedule, each hop a masked all-to-all
  (:func:`jigsaw.comm.comm_1d_alltoall`): an empty tensor goes to every rank
  except the ring neighbour. Needs a backend with all-to-all support (NCCL);
  gloo has none.

Every routine is dimension-agnostic: for an ``n_spatial_dims``-D convolution the
input-channel axis sits at ``-(n_spatial_dims + 1)`` in both ``x`` and ``w``
(leading batch / vmap axes and trailing kernel axes stay opaque). The halo
regime passes a ``halo_fn`` that pulls the missing split-axis positions from the
spatial group before each partial convolution; the no-halo regime leaves it
``None``. Strategy resolution and the per-hop transport / gather primitives are
shared with the transpose driver via
:mod:`jigsaw.modules.convolution.common.channel_comm`.
"""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from ..common.channel_comm import (
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
    """``channel_comm="all_gather"``: gather every slice, then one fused conv.

    Returns ``(y, x_full)`` -- ``x_full`` is the full-channel input (additionally
    halo-padded along the split axis when ``halo_fn`` is given) kept for the
    backward pass.
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
    """Ring channel-parallel conv; ``hop_fn`` is the per-step transport.

    The channel slices travel one hop per step around the ring while each partial
    convolution overlaps the next transfer. When ``halo_fn`` is given every slab
    is first halo-gathered along the split axis, so the spatial exchange happens
    once per channel slab rather than once for the whole full-channel tensor.

    Returns ``(y, x_full)`` where ``x_full`` is the full-channel input (halo-
    padded when ``halo_fn`` is given), reassembled slab by slab, kept for the
    backward pass.
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-parallel convolution for any ``channel_comm`` and dimensionality.

    ``conv_kwargs`` holds ``stride`` / ``padding`` / ``dilation`` for
    ``conv_func``; in the halo regime the split-axis padding is already
    materialised into the ``halo_fn`` output, so pass ``0`` for that axis. All
    strategies return the same ``(y, x_full)`` -- see
    :func:`_over_channels_all_gather` / :func:`_over_channels_streaming`.
    """
    channel_dim = -(n_spatial_dims + 1)
    if channel_comm == "all_gather":
        return _over_channels_all_gather(
            x, w, b, conv_kwargs, channel_group, conv_func, channel_dim, halo_fn
        )
    hop_fn = _resolve_channel_hop(channel_comm, channel_group)
    return _over_channels_streaming(
        x, w, b, conv_kwargs, channel_group, conv_func, channel_dim, halo_fn, hop_fn
    )
