"""Halo exchange of the distributed transpose convolution.

The raw local outputs of neighboring ranks overlap. :func:`transpose_halo_reduce`
sums the overlaps onto the owning ranks in the forward pass and
:func:`transpose_halo_expand` is its reverse in the backward pass.
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
    """Return the global position of the raw local output of a rank.

    Parameters
    ----------
    rank : int
        Rank within the spatial group.
    group_size : int
        Size of the spatial group.
    local_in_size : int
        Size of the local input shard along the split axis.
    kernel : int
        Kernel size along the split axis.
    stride : int
        Stride along the split axis.
    padding : int
        Padding along the split axis.
    dilation : int
        Dilation along the split axis.

    Returns
    -------
    tuple[int, int, int, int]
        ``(base, wloc, out_local, global_out)``: local column ``i`` of the raw
        output maps to global column ``base + i`` in unpadded coordinates, and
        each rank keeps ``out_local`` of the ``global_out`` columns.

    Raises
    ------
    NotImplementedError
        If the global output size along the split axis is not divisible by
        ``group_size``.
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
    """Sum the overlapping raw outputs onto their owning ranks along ``axis``.

    Columns outside the global output are dropped.

    Parameters
    ----------
    y_raw : torch.Tensor
        Raw local output of the transpose convolution.
    spatial_group : dist.ProcessGroup
        Process group of the split axis.
    local_in_size : int
        Size of the local input shard along the split axis.
    kernel : int
        Kernel size along the split axis.
    stride : int
        Stride along the split axis.
    padding : int
        Padding along the split axis.
    dilation : int
        Dilation along the split axis.
    axis : int
        Split axis of the tensor.

    Returns
    -------
    torch.Tensor
        Output shard of this rank.
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
    """Gather the gradient of the raw local output from the owning ranks along ``axis``.

    This is the reverse of :func:`transpose_halo_reduce`. Columns outside the
    global output are zero.

    Parameters
    ----------
    grad_local : torch.Tensor
        Gradient of the output shard of this rank.
    spatial_group : dist.ProcessGroup
        Process group of the split axis.
    local_in_size : int
        Size of the local input shard along the split axis.
    kernel : int
        Kernel size along the split axis.
    stride : int
        Stride along the split axis.
    padding : int
        Padding along the split axis.
    dilation : int
        Dilation along the split axis.
    axis : int
        Split axis of the tensor.

    Returns
    -------
    torch.Tensor
        Gradient of the raw local output.
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
