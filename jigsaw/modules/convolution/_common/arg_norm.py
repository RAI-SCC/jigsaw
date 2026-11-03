"""Normalization of constructor arguments."""

from collections.abc import Callable, Sequence

import torch.distributed as dist


def _ntuple(n: int) -> Callable[[int | tuple[int, ...]], tuple[int, ...]]:
    """Return a parser that converts an ``int`` or tuple to an ``n``-tuple.

    An ``int`` or a length-1 tuple is broadcast to all axes. Other lengths than
    1 or ``n`` raise ``ValueError``.
    """

    def parse(x: int | tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(x, int):
            return (x,) * n
        if len(x) == 1:
            return (x[0],) * n
        if len(x) != n:
            raise ValueError(
                f"expected an int or a tuple of 1 or {n} entries (one per "
                f"spatial axis); got {tuple(x)!r}."
            )
        return tuple(x)

    return parse


def _resolve_padding_str(
    padding: str,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
) -> str | tuple[int, ...]:
    """Convert ``"valid"`` or ``"same"`` padding to per-axis integers.

    ``"same"`` stays a string if an axis needs asymmetric padding (odd
    ``dilation * (kernel_size - 1)``), which only the native convolution supports.

    Raises
    ------
    ValueError
        If ``padding`` is neither ``"valid"`` nor ``"same"``, or ``"same"`` is
        combined with a stride other than 1, as in :class:`torch.nn.Conv2d`.
    """
    if padding == "valid":
        return (0,) * len(kernel_size)
    if padding != "same":
        raise ValueError(
            f"Invalid padding string {padding!r}, should be one of ('valid', 'same')"
        )
    if any(s != 1 for s in stride):
        raise ValueError("padding='same' is not supported for strided convolutions")
    total = [d * (k - 1) for k, d in zip(kernel_size, dilation, strict=True)]
    if any(t % 2 for t in total):
        return "same"
    return tuple(t // 2 for t in total)


def _as_spatial_groups(
    spatial_group: "dist.ProcessGroup | Sequence[dist.ProcessGroup | None]",
    n_spatial: int,
) -> tuple[dist.ProcessGroup | None, ...]:
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
