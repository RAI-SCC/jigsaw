"""Spatial halo exchange for the distributed *transpose* convolution.

The transpose convolution is a scatter: every input position spreads its
contribution over ``dilation * (kernel - 1) + 1`` output positions, so when the
domain is split along a tensor's spatial axis each spatial rank produces a *raw*
local output that is wider than its input slice and whose edges **overlap** the
neighbouring ranks' raw outputs. The globally correct output is the sum of every
rank's placed raw output, cropped to the global output size and re-sharded evenly
across the group.

* :func:`transpose_halo_reduce` (forward): sum the overlapping raw-output pieces
  onto their owning rank and keep that rank's even output shard.
* :func:`transpose_halo_expand` (backward, the reverse): hand every raw-output
  column its gradient, pulled from whichever rank owns that output column.

Both routines take an ``axis`` argument and operate on that axis directly,
matching :mod:`jigsaw.modules.convolution.conv.halo`; the uneven, Gloo-safe
all-to-all (:func:`exchange_along_axis`) is shared by both via
:mod:`jigsaw.modules.convolution.common.halo`. A domain split along *several*
spatial axes (height *and* width, or depth/height/width) is handled by calling
the exchange once per split axis.
"""

import torch
import torch.distributed as dist

from ..common.halo import exchange_along_axis


def produced_output_span(
    rank: int,
    group_size: int,
    local_in_size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
) -> tuple[int, int, int, int]:
    """Where rank ``rank``'s raw local transpose-conv output sits globally.

    Running ``F.conv_transpose1d`` (``padding = 0``) on this rank's owned input
    slice yields a raw output of width ``wloc`` whose column ``i`` maps to global
    output column ``base + i`` (unpadded coordinates -- ``base`` can be negative
    and ``base + wloc`` can exceed the global output size; those columns fall in
    the transpose conv's cropped-away border).

    The global output size is divided evenly across the group, so rank ``r`` keeps
    global output columns ``[r * out_local, (r + 1) * out_local)``.

    Returns ``(base, wloc, out_local, global_out)``.
    """
    extent = dilation * (kernel - 1) + 1  # effective kernel span
    global_in = group_size * local_in_size
    global_out = (global_in - 1) * stride - 2 * padding + extent

    if global_out % group_size != 0:
        raise NotImplementedError(
            "Distributed ConvTranspose with stride != kernel_size needs the "
            f"global output size along the split axis ({global_out}) to be "
            f"divisible by the spatial group size ({group_size})."
        )
    out_local = global_out // group_size

    base = rank * local_in_size * stride - padding
    wloc = (local_in_size - 1) * stride + extent
    return base, wloc, out_local, global_out


def transpose_halo_reduce(
    y_raw: torch.Tensor,
    spatial_group: dist.ProcessGroup,
    local_in_size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
    axis: int = -1,
) -> torch.Tensor:
    """Fold overlapping raw transpose-conv outputs onto their owning rank.

    ``y_raw`` is this rank's raw local output (width ``wloc`` along ``axis``).
    Every rank sends the parts of its raw span that fall in each peer's output
    shard and receives + **sums** the parts of its own shard that the peers
    produced. Columns outside ``[0, global_out)`` are dropped. Operates on
    ``axis`` directly -- no whole-tensor transpose.

    Returns this rank's output shard, ``out_local`` wide along ``axis``.
    """
    axis %= y_raw.dim()
    group_size = spatial_group.size()
    rank = spatial_group.rank()

    base, wloc, out_local, global_out = produced_output_span(
        rank, group_size, local_in_size, kernel, stride, padding, dilation
    )
    keep_lo = rank * out_local
    local_shape = list(y_raw.shape)
    local_shape[axis] = out_local
    y_local = y_raw.new_zeros(local_shape)

    if group_size == 1:
        lo_c, hi_c = max(base, 0), min(base + wloc, global_out)
        if hi_c > lo_c:
            y_local.narrow(axis, lo_c, hi_c - lo_c).add_(
                y_raw.narrow(axis, lo_c - base, hi_c - lo_c)
            )
        return y_local.contiguous()

    empty_shape = list(y_raw.shape)
    empty_shape[axis] = 0
    empty = y_raw.new_empty(empty_shape)
    send_parts: list[torch.Tensor] = []
    send_sizes: list[int] = []
    recv_sizes: list[int] = []
    recv_start: list[int] = []
    for peer in range(group_size):
        # part of my raw span that lands in peer's output shard -> send to peer
        peer_keep_lo = peer * out_local
        s0 = max(base, peer_keep_lo, 0)
        sn = max(min(base + wloc, peer_keep_lo + out_local, global_out) - s0, 0)
        send_sizes.append(sn)
        send_parts.append(y_raw.narrow(axis, s0 - base, sn) if sn > 0 else empty)
        # part of my output shard that peer's raw span produced -> receive and sum
        peer_base, peer_wloc, _, _ = produced_output_span(
            peer, group_size, local_in_size, kernel, stride, padding, dilation
        )
        r0 = max(keep_lo, peer_base, 0)
        recv_sizes.append(
            max(min(keep_lo + out_local, peer_base + peer_wloc, global_out) - r0, 0)
        )
        recv_start.append(r0)

    recv = exchange_along_axis(
        spatial_group, y_raw, send_parts, send_sizes, recv_sizes, axis
    )
    for peer in range(group_size):
        if recv_sizes[peer] > 0:
            off = recv_start[peer] - keep_lo
            y_local.narrow(axis, off, recv_sizes[peer]).add_(recv[peer])

    return y_local.contiguous()


def transpose_halo_expand(
    grad_local: torch.Tensor,
    spatial_group: dist.ProcessGroup,
    local_in_size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
    axis: int = -1,
) -> torch.Tensor:
    """Reverse of :func:`transpose_halo_reduce`: grad wrt the raw local output.

    ``grad_local`` is the grad wrt this rank's output shard. Every raw-output
    column belongs to exactly one rank's shard, so this is a plain uneven
    all-to-all (no summation): each rank pulls the gradient for every column of
    its raw span from whichever rank owns that output column. Columns outside
    ``[0, global_out)`` stay zero. Operates on ``axis`` directly -- no
    whole-tensor transpose.

    Returns the grad wrt this rank's raw local output, ``wloc`` wide along
    ``axis``.
    """
    axis %= grad_local.dim()
    group_size = spatial_group.size()
    rank = spatial_group.rank()

    base, wloc, out_local, global_out = produced_output_span(
        rank, group_size, local_in_size, kernel, stride, padding, dilation
    )
    keep_lo = rank * out_local
    raw_shape = list(grad_local.shape)
    raw_shape[axis] = wloc
    grad_raw = grad_local.new_zeros(raw_shape)

    if group_size == 1:
        lo_c, hi_c = max(base, 0), min(base + wloc, global_out)
        if hi_c > lo_c:
            grad_raw.narrow(axis, lo_c - base, hi_c - lo_c).copy_(
                grad_local.narrow(axis, lo_c, hi_c - lo_c)
            )
        return grad_raw.contiguous()

    empty_shape = list(grad_local.shape)
    empty_shape[axis] = 0
    empty = grad_local.new_empty(empty_shape)
    send_parts: list[torch.Tensor] = []
    send_sizes: list[int] = []
    recv_sizes: list[int] = []
    recv_start: list[int] = []
    for peer in range(group_size):
        peer_base, peer_wloc, _, _ = produced_output_span(
            peer, group_size, local_in_size, kernel, stride, padding, dilation
        )
        # part of my output shard that peer's raw span needs -> send to peer
        s0 = max(keep_lo, peer_base, 0)
        sn = max(min(keep_lo + out_local, peer_base + peer_wloc, global_out) - s0, 0)
        send_sizes.append(sn)
        send_parts.append(
            grad_local.narrow(axis, s0 - keep_lo, sn) if sn > 0 else empty
        )
        # part of my raw span that peer owns -> receive
        peer_keep_lo = peer * out_local
        r0 = max(base, peer_keep_lo, 0)
        recv_sizes.append(
            max(min(base + wloc, peer_keep_lo + out_local, global_out) - r0, 0)
        )
        recv_start.append(r0)

    recv = exchange_along_axis(
        spatial_group, grad_local, send_parts, send_sizes, recv_sizes, axis
    )
    for peer in range(group_size):
        if recv_sizes[peer] > 0:
            off = recv_start[peer] - base
            grad_raw.narrow(axis, off, recv_sizes[peer]).copy_(recv[peer])

    return grad_raw.contiguous()
