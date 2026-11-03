"""``extra_repr`` formatting of the process groups a layer is sharded over."""

from collections.abc import Sequence

import torch.distributed as dist

_Group = dist.ProcessGroup | None


def _size(group: _Group) -> int:
    """Return the size of ``group``; ``None`` counts as 1."""
    return 1 if group is None else group.size()


def shard_repr(**groups: _Group | Sequence[_Group] | int) -> str:
    """Return ``name=size`` for each process group, comma-separated.

    Sizes are read on every call, so the result follows ``reshard_parameters``.

    Parameters
    ----------
    **groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None] | int | None
        Group, sequence of groups (shown as a tuple of sizes) or size, by name.
        ``None`` counts as size 1.

    Returns
    -------
    str
        E.g. ``"spatial=(1, 2), channel=2"``.
    """
    parts = []
    for name, group in groups.items():
        size: int | tuple[int, ...]
        if isinstance(group, int):
            size = group
        elif group is None or isinstance(group, dist.ProcessGroup):
            size = _size(group)
        else:
            size = tuple(_size(g) for g in group)
        parts.append(f"{name}={size}")
    return ", ".join(parts)
