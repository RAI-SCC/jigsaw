"""Single-axis exchange primitive shared by the conv / conv-transpose halo kernels.

Both :mod:`jigsaw.modules.convolution.conv.halo` (input-side halo gather /
scatter-add) and :mod:`jigsaw.modules.convolution.conv_transpose.halo`
(output-side overlap reduce / expand) move data between spatial ranks with the
same uneven, Gloo-compatible all-to-all along one tensor axis.
"""

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
    """Uneven all-to-all of ``axis`` slices, in spatial-rank order.

    ``send_parts[p]`` (size ``send_sizes[p]`` along ``axis``, possibly zero) goes
    to rank ``p``; the returned list holds what every rank sent back, split to
    ``recv_sizes`` (each entry a view along ``axis`` of one receive buffer).

    Implemented with ``all_to_all_single`` (uneven split sizes, Gloo-compatible),
    which splits dim 0 -- so ``axis`` is moved to the front for the transfer
    only. The caller keeps its tensors in their natural layout, so a split along
    a non-innermost spatial axis costs one staging copy of the halo slabs rather
    than a transpose of the whole tensor.
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
