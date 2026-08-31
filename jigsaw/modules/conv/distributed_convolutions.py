"""Distributed convolution implementation.

Only the no-halo case (``stride == kernel_size``) is supported: every spatial
rank convolves its own domain slice independently, so the spatial split needs no
communication. The channel-parallel group cooperates through one of two
interchangeable kernels, selected by ``config.channel_comm``:

* ``"all_gather"`` -- :func:`distributed_conv2d_over_channels_nway` gathers every
  rank's input-channel slice in one shot. Simple and fast, but its transient
  memory peak is ``~2 * C_in`` (the gather list plus the concatenated copy).
* ``"ring"`` -- :func:`distributed_conv2d_over_channels_ring` circulates the
  slices one hop at a time with double-buffered ``isend`` / ``irecv``, overlaps
  each partial convolution with the next transfer and keeps the peak to
  ``C_in`` plus two slice buffers.

Both work for any channel-group size and return the same ``(y, x_full)``.
"""

from typing import Any, Callable

import torch
import torch.distributed as dist


def distributed_conv2d_over_channels_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    channel_group: dist.ProcessGroup,
    conv_func: Callable[..., torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-parallel Conv2d for an arbitrary channel-group size.

    Each rank owns a contiguous ``in_channels // channel_group.size()`` slice of
    the input channels and the matching slice of the weight's *output* channels
    (``w`` still carries every input channel). The ranks all-gather their input
    slices, rebuild the full-channel input and run the local convolution, so the
    result already holds this rank's output-channel slice. Group sizes of one
    (all-gather is the identity) and two are handled by the same code path.

    Parameters
    ----------
    x : torch.Tensor
        This rank's input-channel slice, shape ``(..., C_in / G, H, W)``.
    w : torch.Tensor
        This rank's weight slice, shape ``(C_out / G, C_in, kH, kW)``.
    b : torch.Tensor
        This rank's bias slice, or ``None``.
    stride : tuple[int, int]
        Convolution stride.
    padding : str | tuple[int, int]
        Convolution padding.
    dilation : tuple[int, int]
        Convolution dilation.
    channel_group : dist.ProcessGroup
        Channel-parallel process group.
    conv_func : Callable
        Convolution primitive (``F.conv2d`` or a vmapped variant).

    Returns
    -------
    y : torch.Tensor
        This rank's output-channel slice.
    x_full : torch.Tensor
        The gathered full-channel input, needed for the backward pass.
    """
    group_size = channel_group.size()
    local_channel_rank = channel_group.rank()

    x = x.contiguous()
    slices = [torch.empty_like(x) for _ in range(group_size)]
    dist.all_gather(slices, x, group=channel_group)
    # Keep this rank's own slice wired into the autograd graph.
    slices[local_channel_rank] = x
    x_full = torch.cat(slices, dim=-3)

    if b is not None:
        y = conv_func(x_full, w, b, stride=stride, padding=padding, dilation=dilation)
    else:
        y = conv_func(x_full, w, stride=stride, padding=padding, dilation=dilation)

    return y, x_full


def distributed_conv2d_over_channels_ring(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    channel_group: dist.ProcessGroup,
    conv_func: Callable[..., torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-parallel Conv2d that streams the input slices around a ring.

    Functionally identical to :func:`distributed_conv2d_over_channels_nway` but
    memory- and overlap-friendly: instead of ``all_gather``-ing every rank's
    input-channel slice at once, the slices travel one hop per step
    (``rank -> rank + 1``) through double-buffered ``isend`` / ``irecv``. Each
    rank convolves the slice it currently holds while the next slice is already
    in flight, so communication overlaps computation. Convolution is linear in
    the input channels, so ``y`` is just the running sum of the per-slice
    partial convolutions, with the bias folded into the first partial only.

    The transient memory peak drops from ``~2 * C_in`` (the gather list plus the
    concatenated copy) to ``C_in`` (the reassembled ``x_full``) plus two
    slice-sized send/recv buffers. ``x_full`` is still assembled and returned so
    the backward-pass contract is unchanged; a variant that also rings the
    backward and never materialises ``x_full`` would remove the last full-size
    allocation but is out of scope here.

    Parameters
    ----------
    x : torch.Tensor
        This rank's input-channel slice, shape ``(..., C_in / G, H, W)``.
    w : torch.Tensor
        This rank's weight slice, shape ``(..., C_out / G, C_in, kH, kW)``.
    b : torch.Tensor
        This rank's bias slice, or ``None``.
    stride : tuple[int, int]
        Convolution stride.
    padding : str | tuple[int, int]
        Convolution padding.
    dilation : tuple[int, int]
        Convolution dilation.
    channel_group : dist.ProcessGroup
        Channel-parallel process group.
    conv_func : Callable
        Convolution primitive (``F.conv2d`` or a vmapped variant).

    Returns
    -------
    y : torch.Tensor
        This rank's output-channel slice.
    x_full : torch.Tensor
        The full-channel input, reassembled slice by slice, kept for backward.
    """
    group_size = channel_group.size()
    rank = channel_group.rank()

    x = x.contiguous()
    c_in_local = x.shape[-3]
    c_in_global = w.shape[-3]

    def _conv(slab: torch.Tensor, j: int, with_bias: bool) -> torch.Tensor:
        """Partial convolution of input-channel block ``j`` with its weight block."""
        w_j = w[..., j * c_in_local : (j + 1) * c_in_local, :, :].contiguous()
        if with_bias and b is not None:
            return conv_func(
                slab, w_j, b, stride=stride, padding=padding, dilation=dilation
            )
        return conv_func(slab, w_j, stride=stride, padding=padding, dilation=dilation)

    if group_size == 1:
        return _conv(x, rank, with_bias=True), x

    x_full = torch.empty(
        (*x.shape[:-3], c_in_global, *x.shape[-2:]), dtype=x.dtype, device=x.device
    )
    x_full[..., rank * c_in_local : (rank + 1) * c_in_local, :, :] = x

    send_to = (rank + 1) % group_size
    recv_from = (rank - 1) % group_size

    def _exchange(
        send_slab: torch.Tensor, recv_slab: torch.Tensor
    ) -> tuple[dist.Work, dist.Work]:
        """Post one ring hop; even/odd ordering keeps the Gloo rendezvous safe."""
        if rank % 2 == 0:
            send = dist.isend(send_slab, group_dst=send_to, group=channel_group)
            recv = dist.irecv(recv_slab, group_src=recv_from, group=channel_group)
        else:
            recv = dist.irecv(recv_slab, group_src=recv_from, group=channel_group)
            send = dist.isend(send_slab, group_dst=send_to, group=channel_group)
        assert send is not None and recv is not None
        return send, recv

    buf_a = torch.empty_like(x)
    buf_b = torch.empty_like(x)
    recv_buf = buf_a

    # Prime the ring: this rank's own slice starts its trip while we convolve it.
    send, recv = _exchange(x, recv_buf)
    y = _conv(x, rank, with_bias=True)

    for step in range(1, group_size):
        send.wait()
        recv.wait()
        slab, recv_buf = recv_buf, (buf_b if recv_buf is buf_a else buf_a)

        j = (rank - step) % group_size
        x_full[..., j * c_in_local : (j + 1) * c_in_local, :, :] = slab

        # Launch the next hop before convolving, so this partial overlaps it.
        if step < group_size - 1:
            send, recv = _exchange(slab, recv_buf)

        y = y + _conv(slab, j, with_bias=False)

    return y, x_full


_CHANNEL_COMM_FNS: dict[str, Callable[..., tuple[torch.Tensor, torch.Tensor]]] = {
    "all_gather": distributed_conv2d_over_channels_nway,
    "ring": distributed_conv2d_over_channels_ring,
}


def conv2d_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed 2d convolution for any spatial- and channel-group size.

    Requires ``stride == kernel_size``, so the spatial split needs no halo
    exchange: every spatial rank convolves its own domain slice independently
    while the channel group cooperates through the kernel named by
    ``config.channel_comm`` (``"all_gather"`` by default, or ``"ring"`` for the
    memory-lean overlapped variant).

    Returns ``(y, x_full)`` where ``x_full`` is the full-channel input kept for
    the backward pass.
    """
    channel_comm = getattr(config, "channel_comm", "all_gather")
    over_channels_fn = _CHANNEL_COMM_FNS[channel_comm]
    return over_channels_fn(
        x,
        w,
        b,
        stride,
        padding,
        dilation,
        channel_group,
        config.conv2d_fn,
    )
