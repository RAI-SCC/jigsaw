"""Constructor-argument normalisers shared by the conv / conv-transpose modules.

Both ``_DistributedConvNd`` and ``_DistributedConvTransposeNd`` accept
``kernel_size`` / ``stride`` / ``padding`` / ``dilation`` as an ``int`` or a
short tuple, and ``spatial_group`` as a single process group or a per-axis
sequence. These helpers turn either form into the ``n_spatial``-tuple the rest
of the implementation expects.
"""

from collections.abc import Sequence
from typing import Callable, Optional

import torch.distributed as dist


def _ntuple(n: int) -> Callable[[int | tuple[int, ...]], tuple[int, ...]]:
    """Return a parser that normalises an ``int`` / short tuple to an ``n``-tuple.

    An ``int`` or a length-1 tuple is broadcast to every axis; a longer tuple is
    taken axis-by-axis (and indexes past ``n`` are dropped). A tuple shorter than
    ``n`` but longer than 1 raises ``IndexError`` -- it is under-specified.
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
    """Normalise ``spatial_group`` to an ``n_spatial``-tuple aligned with the axes.

    A single process group keeps the historical behaviour -- the domain is split
    along the last (width) axis only. A sequence is taken axis-by-axis, so a
    full-length sequence splits several spatial axes; ``None`` (or a group of
    size 1) marks an axis that is not split.
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
