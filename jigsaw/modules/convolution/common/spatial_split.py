"""Spatial-split bookkeeping shared by the distributed conv / conv-transpose kernels.

Both :mod:`jigsaw.modules.convolution.conv.distributed` and
:mod:`jigsaw.modules.convolution.conv_transpose.distributed` describe their domain split with
``spatial_groups`` -- an ``n_spatial``-tuple aligned with the trailing spatial
axes, an entry ``None`` (or a size-1 group) where that axis is not split -- and
share the same helpers for walking it, deciding whether a halo exchange is
needed, computing the local-conv padding and slicing a full-channel gradient
back to this rank's channel shard.
"""

from typing import Iterator, Optional

import torch
import torch.distributed as dist


def _split_axes(
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
) -> Iterator[tuple[int, int, dist.ProcessGroup]]:
    """Yield ``(spatial_idx, tensor_axis, group)`` for the axes actually split.

    ``spatial_groups`` is aligned with the trailing spatial axes, so spatial
    index ``idx`` maps to tensor axis ``idx - len(spatial_groups)`` (e.g. the
    2-D ``(H, W)`` groups map to axes ``-2`` and ``-1``).
    """
    n = len(spatial_groups)
    for idx, group in enumerate(spatial_groups):
        if group is not None and group.size() > 1:
            yield idx, idx - n, group


def _needs_halo(
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
    padding: str | tuple[int, ...],
    dilation: tuple[int, ...],
) -> bool:
    """Whether a nominally no-halo call (``stride == kernel``) still needs a halo.

    ``stride == kernel_size`` only tiles a split axis with no overlap when it
    also has zero padding and unit dilation. If *any* split axis has non-trivial
    padding or dilation, neighbouring ranks' windows overlap (or their raw
    outputs overlap / are cropped) and the halo path is required.
    """
    return isinstance(padding, tuple) and any(
        padding[idx] != 0 or dilation[idx] != 1
        for idx, _, _ in _split_axes(spatial_groups)
    )


def _reduce_scatter_channels(
    g_x: torch.Tensor, channel_group: dist.ProcessGroup, n: int
) -> tuple[torch.Tensor, Optional[dist.Work]]:
    """Sum a full-channel grad across the channel group, keeping this rank's shard.

    ``g_x`` carries every input channel on axis ``-(n + 1)``; the channel group
    splits that axis evenly and this rank keeps its own block of the summed grad.

    Uses ``reduce_scatter`` rather than ``all_reduce`` + slice: an ``all_reduce``
    is a ``reduce_scatter`` followed by an ``all_gather`` whose result would be
    sliced away here, so this moves half the channel-group bytes. The scatter is
    launched with ``async_op=True``; wait on the returned handle (when it is not
    ``None``) before reading the tensor, so the caller can overlap it with the
    grad-weight compute. Group size one is the identity.
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
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...],
) -> tuple[int, ...]:
    """Local-conv padding: split axes are 0 (absorbed into the halo), else given."""
    split = {axis for _, axis, _ in _split_axes(spatial_groups)}
    n = len(padding)
    return tuple(0 if (idx - n) in split else pad for idx, pad in enumerate(padding))
