"""Distributed convolution implementation.

Two regimes are supported, selected by :class:`~jigsaw.modules.conv.config.ConvConfig`:

* ``stride == kernel_size`` (and no dilation/padding on the split axis): the
  spatial split needs no communication -- every spatial rank convolves its own
  domain slice independently (:func:`conv2d_nway`).
* ``stride != kernel_size``: neighbouring output columns share input columns, so
  each spatial rank first pulls the missing input columns from the other ranks
  with a halo exchange, then convolves (:func:`conv2d_halo_nway`).

Both regimes cooperate with the channel-parallel group through
:func:`_gather_full_channels`, which works for any group size.
The spatial split is 1-D, along the last axis (width); height is never split.
"""

from typing import Any, Callable

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# channel-parallel helpers
# ---------------------------------------------------------------------------
def _gather_full_channels(
    x: torch.Tensor, channel_group: dist.ProcessGroup
) -> torch.Tensor:
    """All-gather the channel-parallel input slices into a full-channel tensor.

    Each rank owns a contiguous ``C_in / G`` slice of the input channels; the
    group all-gathers them and rebuilds ``(..., C_in, H, W)``. This rank's own
    slice is spliced back in so it stays wired into the autograd graph. Group
    size one is the identity.
    """
    group_size = channel_group.size()
    if group_size == 1:
        return x

    local_channel_rank = channel_group.rank()
    slices = [torch.empty_like(x) for _ in range(group_size)]
    dist.all_gather(slices, x, group=channel_group)
    slices[local_channel_rank] = x
    return torch.cat(slices, dim=-3)


def distributed_conv2d_over_channels_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    channel_group: dist.ProcessGroup,
    conv_func: Callable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-parallel Conv2d for an arbitrary channel-group size.

    Each rank owns a contiguous ``in_channels // channel_group.size()`` slice of
    the input channels and the matching slice of the weight's *output* channels
    (``w`` still carries every input channel). The ranks all-gather their input
    slices, rebuild the full-channel input and run the local convolution, so the
    result already holds this rank's output-channel slice.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
    """
    x_full = _gather_full_channels(x.contiguous(), channel_group)

    if b is not None:
        y = conv_func(x_full, w, b, stride=stride, padding=padding, dilation=dilation)
    else:
        y = conv_func(x_full, w, stride=stride, padding=padding, dilation=dilation)

    return y, x_full


def conv2d_nway(
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
    """Distributed 2d convolution for the no-halo regime (``stride == kernel``).

    Every spatial rank convolves its own domain slice independently while the
    channel group cooperates through
    :func:`distributed_conv2d_over_channels_nway`.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
    """
    return distributed_conv2d_over_channels_nway(
        x,
        w,
        b,
        stride,
        padding,
        dilation,
        channel_group,
        config.conv2d_fn,
    )


def conv2d_nway_backward(
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
    g_x = config.conv2d_input_fn(
        x_aug.shape[int(bayesian) :],
        weight,
        grad_output.to(weight.dtype),  # handles autocast
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).contiguous()

    g_x_handle = dist.all_reduce(
        g_x, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
    )

    g_w = config.conv2d_weight_fn(
        x_aug,
        weight.shape[int(bayesian) :],
        grad_output.to(weight.dtype),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    g_x_handle.wait()

    shard = in_channels_global // channel_group.size()
    start = channel_group.rank() * shard
    g_x = g_x[..., start : start + shard, :, :]

    return g_x, g_w


# ---------------------------------------------------------------------------
# halo helpers (stride != kernel_size)
# ---------------------------------------------------------------------------
def _needed_input_span(
    rank: int,
    group_size: int,
    local_in_width: int,
    kernel_w: int,
    stride: int,
    padding: int,
    dilation: int,
) -> tuple[int, int, int, int]:
    """Input columns rank ``rank`` needs to produce its output-column block.

    The global output width is split evenly across the spatial group, so rank
    ``r`` produces global output columns ``[r * out_w_local, (r + 1) * out_w_local)``.
    Producing them needs global *input* columns (unpadded coordinates, so the
    left edge can be negative and the right edge can exceed the global width --
    those positions are the convolution's zero padding) ``[lo, hi)``.

    Returns ``(lo, hi, need_width, out_w_local)`` with ``hi - lo == need_width``.
    """
    extent = dilation * (kernel_w - 1) + 1  # effective kernel span
    global_in_width = group_size * local_in_width
    global_out_width = (global_in_width + 2 * padding - extent) // stride + 1

    if global_out_width % group_size != 0:
        raise NotImplementedError(
            "Distributed Conv2d with stride != kernel_size needs the global "
            f"output width ({global_out_width}) to be divisible by the spatial "
            f"group size ({group_size})."
        )
    out_w_local = global_out_width // group_size

    lo = rank * out_w_local * stride - padding
    need_width = (out_w_local - 1) * stride + extent
    return lo, lo + need_width, need_width, out_w_local


def _exchange_along_width(
    spatial_group: dist.ProcessGroup,
    template: torch.Tensor,
    send_parts: list[torch.Tensor],
    send_widths: list[int],
    recv_widths: list[int],
) -> list[torch.Tensor]:
    """Uneven all-to-all of width slices, in spatial-rank order.

    ``send_parts[p]`` (width ``send_widths[p]``, possibly zero) goes to rank
    ``p``; the returned list holds what every rank sent back, split to
    ``recv_widths``. Implemented with ``all_to_all_single`` (uneven split sizes,
    Gloo-compatible) by moving the width axis to the front.
    """
    lead = template.shape[:-1]
    parts = [p for p, w in zip(send_parts, send_widths) if w > 0]
    send_cat = (
        torch.cat(parts, dim=-1) if parts else template.new_empty((*lead, 0))
    )
    send_cat = send_cat.movedim(-1, 0).contiguous()
    recv_cat = template.new_empty((sum(recv_widths), *lead))

    dist.all_to_all_single(
        recv_cat,
        send_cat,
        output_split_sizes=list(recv_widths),
        input_split_sizes=list(send_widths),
        group=spatial_group,
    )

    recv_cat = recv_cat.movedim(0, -1)
    out, off = [], 0
    for width in recv_widths:
        out.append(recv_cat[..., off : off + width])
        off += width
    return out


def _halo_gather(
    x_full: torch.Tensor,
    spatial_group: dist.ProcessGroup,
    kernel_w: int,
    stride: int,
    padding: int,
    dilation: int,
) -> tuple[torch.Tensor, int]:
    """Assemble this rank's halo-padded input slice along the width axis.

    Every rank can compute every other rank's needed span locally, so the
    exchange is a single uneven all-to-all: send each peer the part of my owned
    columns that lies in *its* span, receive the parts of *my* span that live on
    the peers. Out-of-bounds columns (the convolution's zero padding) are left as
    zeros.

    Returns ``(x_need, out_w_local)``.
    """
    group_size = spatial_group.size()
    rank = spatial_group.rank()
    local_w = x_full.shape[-1]
    global_w = group_size * local_w

    lo, hi, need_width, out_w_local = _needed_input_span(
        rank, group_size, local_w, kernel_w, stride, padding, dilation
    )
    x_need = x_full.new_zeros((*x_full.shape[:-1], need_width))

    def _place(src: torch.Tensor, col_start: int) -> None:
        x_need[..., col_start - lo : col_start - lo + src.shape[-1]] = src

    if group_size == 1:
        lo_c, hi_c = max(lo, 0), min(hi, global_w)
        _place(x_full[..., lo_c:hi_c], lo_c)
        return x_need.contiguous(), out_w_local

    empty = x_full.new_empty((*x_full.shape[:-1], 0))
    send_parts: list[torch.Tensor] = []
    send_widths: list[int] = []
    recv_widths: list[int] = []
    recv_col: list[int] = []
    for peer in range(group_size):
        peer_lo, peer_hi, _, _ = _needed_input_span(
            peer, group_size, local_w, kernel_w, stride, padding, dilation
        )
        # columns I own that peer needs
        s0 = max(peer_lo, rank * local_w)
        sw = max(min(peer_hi, (rank + 1) * local_w) - s0, 0)
        send_widths.append(sw)
        col = s0 - rank * local_w
        send_parts.append(x_full[..., col : col + sw] if sw > 0 else empty)
        # columns of my span that peer owns (clipped to the real domain)
        r0 = max(lo, peer * local_w, 0)
        recv_widths.append(max(min(hi, (peer + 1) * local_w, global_w) - r0, 0))
        recv_col.append(r0)

    recv = _exchange_along_width(
        spatial_group, x_full, send_parts, send_widths, recv_widths
    )
    for peer in range(group_size):
        if recv_widths[peer] > 0:
            _place(recv[peer], recv_col[peer])

    return x_need.contiguous(), out_w_local


def _halo_scatter_add(
    g_x_need: torch.Tensor,
    spatial_group: dist.ProcessGroup,
    kernel_w: int,
    stride: int,
    padding: int,
    dilation: int,
    local_in_width: int,
) -> torch.Tensor:
    """Reverse of :func:`_halo_gather`: fold halo-column grads back to owners.

    ``g_x_need`` is the grad wrt this rank's halo-padded input span. Every input
    column that was borrowed by several ranks collects a contribution from each,
    so the reverse exchange *sums* overlapping pieces onto the owning rank.

    Returns the grad wrt this rank's owned input columns, width
    ``local_in_width``.
    """
    group_size = spatial_group.size()
    rank = spatial_group.rank()
    global_w = group_size * local_in_width

    lo, hi, _, _ = _needed_input_span(
        rank, group_size, local_in_width, kernel_w, stride, padding, dilation
    )
    g_x_local = g_x_need.new_zeros((*g_x_need.shape[:-1], local_in_width))

    if group_size == 1:
        return (g_x_need[..., -lo : -lo + local_in_width]).contiguous()

    empty = g_x_need.new_empty((*g_x_need.shape[:-1], 0))
    send_parts: list[torch.Tensor] = []
    send_widths: list[int] = []
    recv_widths: list[int] = []
    recv_off: list[int] = []
    for peer in range(group_size):
        # part of my span that peer owns -> send it back to peer
        s0 = max(lo, peer * local_in_width, 0)
        sw = max(min(hi, (peer + 1) * local_in_width, global_w) - s0, 0)
        send_widths.append(sw)
        send_parts.append(g_x_need[..., s0 - lo : s0 - lo + sw] if sw > 0 else empty)
        # part of peer's span that I own -> receive and accumulate
        peer_lo, peer_hi, _, _ = _needed_input_span(
            peer, group_size, local_in_width, kernel_w, stride, padding, dilation
        )
        r0 = max(peer_lo, rank * local_in_width, 0)
        recv_widths.append(
            max(min(peer_hi, (rank + 1) * local_in_width, global_w) - r0, 0)
        )
        recv_off.append(r0 - rank * local_in_width)

    recv = _exchange_along_width(
        spatial_group, g_x_need, send_parts, send_widths, recv_widths
    )
    for peer in range(group_size):
        if recv_widths[peer] > 0:
            off = recv_off[peer]
            g_x_local[..., off : off + recv_widths[peer]] += recv[peer]

    return g_x_local


def _width_padding(padding: tuple[int, int]) -> tuple[int, int]:
    """Return the local-conv padding: height as given, width already materialised."""
    return (padding[0], 0)


def conv2d_halo_nway(
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
    """Distributed 2d convolution for the halo regime (``stride != kernel``).

    Gathers the full-channel input, pulls the missing width columns from the
    other spatial ranks, then runs the local convolution so the result already
    holds this rank's output-channel *and* output-column slice.

    Returns ``(y, x_need)`` where ``x_need`` is the halo-padded full-channel
    input kept for the backward pass.
    """
    if not isinstance(padding, tuple):
        raise NotImplementedError(
            "Distributed Conv2d with stride != kernel_size does not support "
            f"string padding ({padding!r}); pass an explicit (h, w) padding."
        )

    x_full = _gather_full_channels(x.contiguous(), channel_group)
    x_need, _ = _halo_gather(
        x_full, spatial_group, w.shape[-1], stride[1], padding[1], dilation[1]
    )

    local_padding = _width_padding(padding)
    if b is not None:
        y = config.conv2d_fn(
            x_need, w, b, stride=stride, padding=local_padding, dilation=dilation
        )
    else:
        y = config.conv2d_fn(
            x_need, w, stride=stride, padding=local_padding, dilation=dilation
        )

    return y, x_need


def conv2d_halo_nway_backward(
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

    ``g_w`` comes straight from the halo-padded input and this rank's
    ``grad_output`` (spatial ranks are summed by the parameter hook). ``g_x`` is
    computed on the halo-padded span, reduced across the channel group, folded
    back onto the owning spatial ranks (:func:`_halo_scatter_add`) and finally
    sliced to this rank's channel shard.
    """
    local_padding = _width_padding(padding)

    g_x_need = config.conv2d_input_fn(
        x_aug.shape[int(bayesian) :],
        weight,
        grad_output.to(weight.dtype),  # handles autocast
        stride=stride,
        padding=local_padding,
        dilation=dilation,
    ).contiguous()

    g_x_handle = dist.all_reduce(
        g_x_need, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
    )

    g_w = config.conv2d_weight_fn(
        x_aug,
        weight.shape[int(bayesian) :],
        grad_output.to(weight.dtype),
        stride=stride,
        padding=local_padding,
        dilation=dilation,
    )

    g_x_handle.wait()

    g_x = _halo_scatter_add(
        g_x_need,
        spatial_group,
        weight.shape[-1],
        stride[1],
        padding[1],
        dilation[1],
        local_in_width,
    )

    shard = in_channels_global // channel_group.size()
    start = channel_group.rank() * shard
    g_x = g_x[..., start : start + shard, :, :]

    return g_x, g_w
