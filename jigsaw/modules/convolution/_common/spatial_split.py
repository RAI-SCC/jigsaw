"""Helpers for domain splits along several spatial axes."""

from collections.abc import Iterator

import torch
import torch.distributed as dist


def _split_axes(
    spatial_groups: tuple[dist.ProcessGroup | None, ...],
) -> Iterator[tuple[int, int, dist.ProcessGroup]]:
    """Yield ``(spatial_idx, tensor_axis, group)`` for each split axis."""
    n = len(spatial_groups)
    for idx, group in enumerate(spatial_groups):
        if group is not None and group.size() > 1:
            yield idx, idx - n, group


def _needs_halo(
    spatial_groups: tuple[dist.ProcessGroup | None, ...],
    padding: str | tuple[int, ...],
    dilation: tuple[int, ...],
) -> bool:
    """Return whether a split axis has padding or dilation, which requires a halo."""
    return isinstance(padding, tuple) and any(
        padding[idx] != 0 or dilation[idx] != 1
        for idx, _, _ in _split_axes(spatial_groups)
    )


def _reduce_scatter_channels(
    g_x: torch.Tensor, channel_group: dist.ProcessGroup, n: int
) -> tuple[torch.Tensor, dist.Work | None]:
    """Sum a full-channel gradient over the channel group and keep the local shard.

    The reduce-scatter is asynchronous, so the returned handle must be waited on
    before the tensor is read. A group of size one is the identity.
    """
    group_size = channel_group.size()
    if group_size == 1:
        return g_x.contiguous(), None

    channel_dim = -(n + 1)
    src = g_x.movedim(channel_dim, 0).contiguous()
    shard = src.new_empty((src.shape[0] // group_size, *src.shape[1:]))
    handle = dist.reduce_scatter_tensor(
        shard, src, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
    )
    return shard.movedim(0, channel_dim), handle


def _local_padding(
    padding: tuple[int, ...],
    spatial_groups: tuple[dist.ProcessGroup | None, ...],
) -> tuple[int, ...]:
    """Return the padding of the local convolution, zero on split axes."""
    split = {axis for _, axis, _ in _split_axes(spatial_groups)}
    n = len(padding)
    return tuple(0 if (idx - n) in split else pad for idx, pad in enumerate(padding))
