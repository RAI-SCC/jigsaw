"""Halo exchange primitive shared by the convolution kernels."""

import torch
import torch.distributed as dist


def exchange_along_axis(
    spatial_group: dist.ProcessGroup,
    template: torch.Tensor,
    send_parts: list[torch.Tensor],
    send_sizes: list[int],
    recv_sizes: list[int],
    axis: int = -1,
) -> list[torch.Tensor]:
    """Exchange slices along ``axis`` with all spatial ranks in an uneven all-to-all.

    ``send_parts[p]`` is sent to rank ``p``. The result holds the slices received
    from every rank, with sizes ``recv_sizes`` along ``axis``.

    Parameters
    ----------
    spatial_group : dist.ProcessGroup
        Process group of the split axis.
    template : torch.Tensor
        Tensor whose device and dtype the exchanged slices follow.
    send_parts : list[torch.Tensor]
        Slice sent to each rank.
    send_sizes : list[int]
        Size along ``axis`` of each slice in ``send_parts``.
    recv_sizes : list[int]
        Size along ``axis`` of the slice received from each rank.
    axis : int
        Axis along which the slices are exchanged.

    Returns
    -------
    list[torch.Tensor]
        Slice received from each rank.
    """
    axis %= template.dim()
    other = (*template.shape[:axis], *template.shape[axis + 1 :])

    parts = [p for p, n in zip(send_parts, send_sizes) if n > 0]
    if parts:
        send_cat = torch.cat(parts, dim=axis).movedim(axis, 0).contiguous()
    else:
        send_cat = template.new_empty((0, *other))
    recv_cat = template.new_empty((sum(recv_sizes), *other))

    dist.all_to_all_single(
        recv_cat,
        send_cat,
        output_split_sizes=list(recv_sizes),
        input_split_sizes=list(send_sizes),
        group=spatial_group,
    )

    recv_cat = recv_cat.movedim(0, axis)
    out, off = [], 0
    for size in recv_sizes:
        out.append(recv_cat.narrow(axis, off, size))
        off += size
    return out
