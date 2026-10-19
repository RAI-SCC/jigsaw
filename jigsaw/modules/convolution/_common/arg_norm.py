"""Normalization of constructor arguments."""

from collections.abc import Sequence
from typing import Callable, Optional

import torch.distributed as dist


def _ntuple(n: int) -> Callable[[int | tuple[int, ...]], tuple[int, ...]]:
    """Return a parser that converts an ``int`` or tuple to an ``n``-tuple.

    An ``int`` or a length-1 tuple is broadcast to all axes.
    """

    def parse(x: int | tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(x, int):
            return (x,) * n
        if len(x) == 1:
            return (x[0],) * n
        return tuple(x[i] for i in range(n))

    return parse


def _as_spatial_groups(
    spatial_group: "dist.ProcessGroup | Sequence[Optional[dist.ProcessGroup]]",
    n_spatial: int,
) -> tuple[Optional[dist.ProcessGroup], ...]:
    """Convert ``spatial_group`` to a tuple with one entry per spatial axis.

    A single group splits the last axis only. ``None`` or a group of size 1
    marks an axis that is not split.
    """
    if isinstance(spatial_group, dist.ProcessGroup):
        return (None,) * (n_spatial - 1) + (spatial_group,)
    groups = tuple(spatial_group)
    if len(groups) != n_spatial:
        raise ValueError(
            f"spatial_group as a sequence must have {n_spatial} entries "
            f"(one per spatial axis, innermost last); got {len(groups)}."
        )
    return groups
