"""Distributed convolution implementation.

Only the no-halo case (``stride == kernel_size``) is supported: every spatial
rank convolves its own domain slice independently, so the spatial split needs no
communication. The channel-parallel group cooperates through one of three
interchangeable kernels, selected by ``config.channel_comm``:

* ``"all_gather"`` -- :func:`distributed_conv2d_over_channels_nway` gathers every
  rank's input-channel slice in one shot. Simple and fast, but its transient
  memory peak is ``~2 * C_in`` (the gather list plus the concatenated copy).
* ``"ring"`` -- :func:`distributed_conv2d_over_channels_p2p` circulates the
  slices one hop at a time with batched ``isend`` / ``irecv``
  (:func:`jigsaw.comm.comm_1d_p2p`), overlaps each partial convolution with the
  next transfer and keeps the peak to ``C_in`` plus two slice buffers.
* ``"all_to_all"`` -- :func:`distributed_conv2d_over_channels_alltoall` runs the
  same ring schedule but moves each hop with a masked all-to-all
  (:func:`jigsaw.comm.comm_1d_alltoall`): an empty tensor goes to every rank
  except the ring neighbour. Needs a backend with all-to-all support (NCCL);
  gloo has none.

All three work for any channel-group size and return the same ``(y, x_full)``.
``ConvConfig`` resolves the default: ``"all_to_all"`` on a CUDA device,
``"ring"`` otherwise.
"""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from jigsaw.comm import comm_1d_alltoall, comm_1d_p2p

# A per-hop transport: ``fn(data, buffer, direction=, group=, async_op=) ->
# (recv_buffer, wait_handle)``. Satisfied by ``comm_1d_p2p`` / ``comm_1d_alltoall``.
_HopFn = Callable[..., tuple[torch.Tensor, Optional[dist.Work]]]


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


def _conv2d_over_channels_p2p(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    channel_group: dist.ProcessGroup,
    conv_func: Callable[..., torch.Tensor],
    hop_fn: _HopFn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ring channel-parallel Conv2d; ``hop_fn`` is the per-step transport.

    Instead of ``all_gather``-ing every rank's input-channel slice at once, the
    slices travel one hop per step (``rank -> rank + 1``). ``hop_fn`` moves one
    slice one hop and returns ``(recv_buffer, wait_handle)``; it is
    :func:`jigsaw.comm.comm_1d_p2p` (batched ``isend`` / ``irecv``) or
    :func:`jigsaw.comm.comm_1d_alltoall` (masked all-to-all). Each rank
    convolves the slice it currently holds while the next slice is already in
    flight, so communication overlaps computation. Convolution is linear in the
    input channels, so ``y`` is the running sum of the per-slice partial
    convolutions, with the bias folded into the first partial only.

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
    hop_fn : Callable
        Per-hop transport, ``comm_1d_p2p`` or ``comm_1d_alltoall``.

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

    buf_a = torch.empty_like(x)
    buf_b = torch.empty_like(x)
    recv_buf = buf_a

    # Prime the ring: this rank's own slice sets off while we convolve it.
    recv_buf, handle = hop_fn(
        x, recv_buf, direction=1, group=channel_group, async_op=True
    )
    y = _conv(x, rank, with_bias=True)

    for step in range(1, group_size):
        if handle is not None:
            handle.wait()
        slab, recv_buf = recv_buf, (buf_b if recv_buf is buf_a else buf_a)

        j = (rank - step) % group_size
        x_full[..., j * c_in_local : (j + 1) * c_in_local, :, :] = slab

        # Relay before convolving, so this partial overlaps the next hop.
        if step < group_size - 1:
            recv_buf, handle = hop_fn(
                slab, recv_buf, direction=1, group=channel_group, async_op=True
            )

        y = y + _conv(slab, j, with_bias=False)

    return y, x_full


def distributed_conv2d_over_channels_p2p(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    channel_group: dist.ProcessGroup,
    conv_func: Callable[..., torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-parallel Conv2d that streams the input slices around a p2p ring.

    Thin wrapper over :func:`_conv2d_over_channels_p2p` with
    :func:`jigsaw.comm.comm_1d_p2p` (one ``dist.batch_isend_irecv`` per hop) as
    the transport. See :func:`_conv2d_over_channels_p2p` for the schedule and
    the memory characteristics.
    """
    return _conv2d_over_channels_p2p(
        x, w, b, stride, padding, dilation, channel_group, conv_func, comm_1d_p2p
    )


def distributed_conv2d_over_channels_alltoall(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    channel_group: dist.ProcessGroup,
    conv_func: Callable[..., torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-parallel Conv2d, ring schedule, each hop a masked all-to-all.

    Thin wrapper over :func:`_conv2d_over_channels_p2p` with
    :func:`jigsaw.comm.comm_1d_alltoall` as the transport: every hop is a
    ``dist.all_to_all`` in which this rank sends its slice to the ring neighbour
    and an empty tensor to everyone else. Requires a backend with all-to-all
    support (NCCL); ``gloo`` has none, so this raises there -- use ``"ring"``.
    """
    if dist.get_backend(channel_group) == "gloo":
        raise RuntimeError(
            "channel_comm='all_to_all' needs a backend with all-to-all support "
            "(e.g. NCCL); gloo has none. Use channel_comm='ring' instead."
        )
    return _conv2d_over_channels_p2p(
        x, w, b, stride, padding, dilation, channel_group, conv_func, comm_1d_alltoall
    )


_CHANNEL_COMM_FNS: dict[str, Callable[..., tuple[torch.Tensor, torch.Tensor]]] = {
    "all_gather": distributed_conv2d_over_channels_nway,
    "ring": distributed_conv2d_over_channels_p2p,
    "all_to_all": distributed_conv2d_over_channels_alltoall,
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
    ``config.channel_comm`` -- ``"all_gather"`` (one-shot gather), ``"ring"``
    (memory-lean overlapped p2p) or ``"all_to_all"`` (same ring over masked
    all-to-all hops). ``ConvConfig`` resolves the default from the device.

    Returns ``(y, x_full)`` where ``x_full`` is the full-channel input kept for
    the backward pass.
    """
    over_channels_fn = _CHANNEL_COMM_FNS[config.channel_comm]
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
