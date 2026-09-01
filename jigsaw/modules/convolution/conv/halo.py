"""Input-side spatial halo exchange for the distributed N-D convolution.

The domain is split along one or more of a tensor's spatial axes across the
matching ``spatial_group``. When ``stride != kernel_size`` neighbouring output
positions share input positions, so before convolving every rank must pull the
input positions it is missing from the other ranks (:func:`halo_gather`); the
backward pass folds the borrowed-position gradients back onto their owners
(:func:`halo_scatter_add`). The uneven, Gloo-safe all-to-all underneath
(:func:`exchange_along_axis`) is shared with the transpose halo via
:mod:`jigsaw.modules.convolution.common.halo`.

Both routines take an ``axis`` argument and operate on that axis directly,
treating every other axis opaquely, so the same code serves Conv1d
(``(N, C, L)``) and Conv2d (``(N, C, H, W)``) -- the caller passes the single set
of split-axis conv parameters (kernel, stride, padding, dilation along that
axis). A domain split along *several* spatial axes (height *and* width, or
depth/height/width) is handled by calling the exchange once per split axis.
"""

import torch
import torch.distributed as dist

from ..common.halo import exchange_along_axis


def needed_input_span(
    rank: int,
    group_size: int,
    local_in_size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
) -> tuple[int, int, int, int]:
    """Input positions rank ``rank`` needs to produce its output block.

    The global output size along the split axis is divided evenly across the
    spatial group, so rank ``r`` produces global output positions
    ``[r * out_local, (r + 1) * out_local)``. Producing them needs global *input*
    positions (unpadded coordinates, so the left edge can be negative and the
    right edge can exceed the global size -- those positions are the
    convolution's zero padding) ``[lo, hi)``.

    Returns ``(lo, hi, need_size, out_local)`` with ``hi - lo == need_size``.
    """
    extent = dilation * (kernel - 1) + 1  # effective kernel span
    global_in = group_size * local_in_size
    global_out = (global_in + 2 * padding - extent) // stride + 1

    if global_out % group_size != 0:
        raise NotImplementedError(
            "Distributed convolution with stride != kernel_size needs the global "
            f"output size along the split axis ({global_out}) to be divisible by "
            f"the spatial group size ({group_size})."
        )
    out_local = global_out // group_size

    lo = rank * out_local * stride - padding
    need_size = (out_local - 1) * stride + extent
    return lo, lo + need_size, need_size, out_local


def halo_gather(
    x_full: torch.Tensor,
    spatial_group: dist.ProcessGroup,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
    axis: int = -1,
) -> tuple[torch.Tensor, int]:
    """Assemble this rank's halo-padded input slice along ``axis``.

    Every rank can compute every other rank's needed span locally, so the
    exchange is a single uneven all-to-all: send each peer the part of my owned
    positions that lies in *its* span, receive the parts of *my* span that live
    on the peers. Out-of-bounds positions (the convolution's zero padding) are
    left as zeros. Operates on ``axis`` directly -- no whole-tensor transpose.

    Returns ``(x_need, out_local)``.
    """
    axis %= x_full.dim()
    group_size = spatial_group.size()
    rank = spatial_group.rank()
    local_n = x_full.shape[axis]
    global_n = group_size * local_n

    lo, hi, need_size, out_local = needed_input_span(
        rank, group_size, local_n, kernel, stride, padding, dilation
    )
    need_shape = list(x_full.shape)
    need_shape[axis] = need_size
    x_need = x_full.new_zeros(need_shape)

    def _place(src: torch.Tensor, start: int) -> None:
        n = src.shape[axis]
        if n:
            x_need.narrow(axis, start - lo, n).copy_(src)

    if group_size == 1:
        lo_c, hi_c = max(lo, 0), min(hi, global_n)
        if hi_c > lo_c:
            _place(x_full.narrow(axis, lo_c, hi_c - lo_c), lo_c)
        return x_need.contiguous(), out_local

    empty_shape = list(x_full.shape)
    empty_shape[axis] = 0
    empty = x_full.new_empty(empty_shape)
    send_parts: list[torch.Tensor] = []
    send_sizes: list[int] = []
    recv_sizes: list[int] = []
    recv_start: list[int] = []
    for peer in range(group_size):
        peer_lo, peer_hi, _, _ = needed_input_span(
            peer, group_size, local_n, kernel, stride, padding, dilation
        )
        # positions I own that peer needs
        s0 = max(peer_lo, rank * local_n)
        sn = max(min(peer_hi, (rank + 1) * local_n) - s0, 0)
        send_sizes.append(sn)
        col = s0 - rank * local_n
        send_parts.append(x_full.narrow(axis, col, sn) if sn > 0 else empty)
        # positions of my span that peer owns (clipped to the real domain)
        r0 = max(lo, peer * local_n, 0)
        recv_sizes.append(max(min(hi, (peer + 1) * local_n, global_n) - r0, 0))
        recv_start.append(r0)

    recv = exchange_along_axis(
        spatial_group, x_full, send_parts, send_sizes, recv_sizes, axis
    )
    for peer in range(group_size):
        if recv_sizes[peer] > 0:
            _place(recv[peer], recv_start[peer])

    return x_need.contiguous(), out_local


def halo_scatter_add(
    g_x_need: torch.Tensor,
    spatial_group: dist.ProcessGroup,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
    local_in_size: int,
    axis: int = -1,
) -> torch.Tensor:
    """Reverse of :func:`halo_gather`: fold halo-position grads back to owners.

    ``g_x_need`` is the grad wrt this rank's halo-padded input span along
    ``axis``. Every input position that was borrowed by several ranks collects a
    contribution from each, so the reverse exchange *sums* overlapping pieces
    onto the owning rank. Operates on ``axis`` directly -- no whole-tensor
    transpose.

    Returns the grad wrt this rank's owned input positions, size
    ``local_in_size`` along ``axis``.
    """
    axis %= g_x_need.dim()
    group_size = spatial_group.size()
    rank = spatial_group.rank()
    global_n = group_size * local_in_size

    lo, hi, _, _ = needed_input_span(
        rank, group_size, local_in_size, kernel, stride, padding, dilation
    )
    local_shape = list(g_x_need.shape)
    local_shape[axis] = local_in_size
    g_x_local = g_x_need.new_zeros(local_shape)

    if group_size == 1:
        # ``g_x_need`` covers global input positions ``[lo, hi)``; the owned
        # positions are ``[0, local_in_size)``. Positions of the owned range that
        # the span does not reach (a ``stride > 1`` tail, or the ``lo < 0``
        # padding head) carry no gradient and stay zero.
        lo_c, hi_c = max(lo, 0), min(hi, global_n)
        if hi_c > lo_c:
            g_x_local.narrow(axis, lo_c, hi_c - lo_c).copy_(
                g_x_need.narrow(axis, lo_c - lo, hi_c - lo_c)
            )
        return g_x_local

    empty_shape = list(g_x_need.shape)
    empty_shape[axis] = 0
    empty = g_x_need.new_empty(empty_shape)
    send_parts: list[torch.Tensor] = []
    send_sizes: list[int] = []
    recv_sizes: list[int] = []
    recv_off: list[int] = []
    for peer in range(group_size):
        # part of my span that peer owns -> send it back to peer
        s0 = max(lo, peer * local_in_size, 0)
        sn = max(min(hi, (peer + 1) * local_in_size, global_n) - s0, 0)
        send_sizes.append(sn)
        send_parts.append(g_x_need.narrow(axis, s0 - lo, sn) if sn > 0 else empty)
        # part of peer's span that I own -> receive and accumulate
        peer_lo, peer_hi, _, _ = needed_input_span(
            peer, group_size, local_in_size, kernel, stride, padding, dilation
        )
        r0 = max(peer_lo, rank * local_in_size, 0)
        recv_sizes.append(
            max(min(peer_hi, (rank + 1) * local_in_size, global_n) - r0, 0)
        )
        recv_off.append(r0 - rank * local_in_size)

    recv = exchange_along_axis(
        spatial_group, g_x_need, send_parts, send_sizes, recv_sizes, axis
    )
    for peer in range(group_size):
        if recv_sizes[peer] > 0:
            g_x_local.narrow(axis, recv_off[peer], recv_sizes[peer]).add_(recv[peer])

    return g_x_local
