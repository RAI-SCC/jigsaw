"""Distributed roll kernels."""

from typing import Callable, Tuple

import torch
import torch.distributed as dist


def exchange_halo_1d(
    x: torch.Tensor,
    dim: int,
    window_size: int,
    group: dist.ProcessGroup,
    comm_halo_1d_fn: Callable,
) -> torch.Tensor:
    """Exchange a halo along one dimension of a tensor.

    For a positive ``window_size`` the leading slice is sent to the previous rank and
    replaced by the slice received from the next rank. For a negative ``window_size``
    the trailing slice is sent to the next rank and replaced by the slice received
    from the previous rank.

    Parameters
    ----------
    x : torch.Tensor
        Local tensor, modified in place.
    dim : int
        Dimension along which the halo is exchanged.
    window_size : int
        Signed size of the halo.
    group : dist.ProcessGroup
        Process group along which the halo is exchanged.
    comm_halo_1d_fn : Callable
        Function performing the communication.

    Returns
    -------
    torch.Tensor
        ``x`` with the received halo.
    """
    if window_size == 0:
        return x

    sl: list[slice] = [slice(None)] * x.ndim
    sl[dim] = slice(0, window_size) if window_size > 0 else slice(window_size, None)
    sl_tuple: tuple[slice, ...] = tuple(sl)

    halo_to_send = x[sl_tuple].contiguous()
    recv = torch.zeros_like(halo_to_send)
    offset = 1 if window_size > 0 else -1
    recv, _ = comm_halo_1d_fn(
        halo_to_send,
        recv,
        offset,
        group=group,
        async_op=False,
    )

    x[sl_tuple] = recv
    return x


def distributed_roll(
    x: torch.Tensor,
    shifts: Tuple[int, ...],
    dims: Tuple[int, ...],
    distributed_dims: list[int],
    spatial_process_groups: list[dist.ProcessGroup],
    exchange_halo_1d_fn: Callable,
) -> torch.Tensor:
    """Roll a tensor and exchange the halos of its distributed dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Local tensor.
    shifts : tuple[int, ...]
        Shift along each dimension of ``dims``.
    dims : tuple[int, ...]
        Dimensions to roll.
    distributed_dims : list[int]
        Dimensions across which ``x`` is partitioned. Each must appear in ``dims``,
        counted from either end.
    spatial_process_groups : list[dist.ProcessGroup]
        Process group of each entry of ``distributed_dims``.
    exchange_halo_1d_fn : Callable
        Function performing the communication of a halo.

    Returns
    -------
    torch.Tensor
        Rolled local tensor.

    Raises
    ------
    ValueError
        If an entry of ``distributed_dims`` is not in ``dims``.
    """
    x = torch.roll(x, shifts=shifts, dims=dims)

    # Compare the dimensions as counted from the front: ``-2`` is ``2`` in 4-D.
    dim_to_shift = {d % x.ndim: s for d, s in zip(dims, shifts, strict=True)}
    for d, pg in zip(distributed_dims, spatial_process_groups, strict=True):
        if d % x.ndim not in dim_to_shift:
            raise ValueError(f"distributed dim {d} not in dims {dims}")
        x = exchange_halo_1d(
            x,
            dim=d,
            window_size=dim_to_shift[d % x.ndim],
            group=pg,
            comm_halo_1d_fn=exchange_halo_1d_fn,
        )
    return x
