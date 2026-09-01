"""Last-axis exchange primitive shared by the conv / conv-transpose halo kernels.

Both :mod:`jigsaw.modules.convolution.conv.halo` (input-side halo gather /
scatter-add) and :mod:`jigsaw.modules.convolution.conv_transpose.halo`
(output-side overlap reduce / expand) move data between spatial ranks with the
same uneven, Gloo-compatible all-to-all along a tensor's last axis.
"""

import torch
import torch.distributed as dist


def exchange_along_last_axis(
    spatial_group: dist.ProcessGroup,
    template: torch.Tensor,
    send_parts: list[torch.Tensor],
    send_sizes: list[int],
    recv_sizes: list[int],
) -> list[torch.Tensor]:
    """Uneven all-to-all of last-axis slices, in spatial-rank order.

    ``send_parts[p]`` (size ``send_sizes[p]`` along the last axis, possibly zero)
    goes to rank ``p``; the returned list holds what every rank sent back, split
    to ``recv_sizes``. Implemented with ``all_to_all_single`` (uneven split
    sizes, Gloo-compatible) by moving the last axis to the front.
    """
    lead = template.shape[:-1]
    parts = [p for p, n in zip(send_parts, send_sizes) if n > 0]
    send_cat = torch.cat(parts, dim=-1) if parts else template.new_empty((*lead, 0))
    send_cat = send_cat.movedim(-1, 0).contiguous()
    recv_cat = template.new_empty((sum(recv_sizes), *lead))

    dist.all_to_all_single(
        recv_cat,
        send_cat,
        output_split_sizes=list(recv_sizes),
        input_split_sizes=list(send_sizes),
        group=spatial_group,
    )

    recv_cat = recv_cat.movedim(0, -1)
    out, off = [], 0
    for size in recv_sizes:
        out.append(recv_cat[..., off : off + size])
        off += size
    return out
