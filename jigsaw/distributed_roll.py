"""Implementation of distributed roll functions."""

from typing import Callable, Tuple

import torch
import torch.distributed as dist


def exchange_halo_1d(
    x: torch.Tensor,
    dim: int,
    window_size: int,
    group: dist.ProcessGroup,
    empty: torch.Tensor,
    comm_halo_1d_fn: Callable,
) -> torch.Tensor:
    """Exchange a halo of `window_size` along tensor `dim` over a 1D process group.

    Positive window_size: send leading slice to rank-1, receive into leading slice from rank+1.
    Negative window_size: send trailing slice to rank+1, receive into trailing slice from rank-1.

    Parameters
    ----------
    x: torch.Tensor
        Data tensor.
    dim: int
        Dimension whetere the data should be exchanged
    window_size: int
        Size of halo to be communicated.
    group: dist.ProcessGroup
        Process group where communication occurs.
    empty: torch.Tensor
        Empty tensor for computation reasons.

    Returns
    -------
        Data with halo already appended.
    """
    if window_size == 0:
        return x

    sl: list[slice] = [slice(None)] * x.ndim
    sl[dim] = slice(0, window_size) if window_size > 0 else slice(window_size, None)
    sl_tuple: tuple[slice, ...] = tuple(sl)

    halo_to_send = x[sl_tuple].contiguous()
    recv = torch.zeros_like(halo_to_send)

    recv = comm_halo_1d_fn(window_size, halo_to_send, recv, group, empty=empty)

    x[sl] = recv
    return x


def distributed_roll(
    x: torch.Tensor,
    shifts: Tuple[int, ...],
    dims: Tuple[int, ...],
    distributed_dims: list[int],
    spatial_process_groups: list[dist.ProcessGroup],
    empty: torch.Tensor,
    exchange_halo_1d_fn: Callable,
) -> torch.Tensor:
    """
    Distributed roll function.

    Roll `x` by `shifts` along `dims`; for each dim in `distributed_dims`,
    exchange a halo on the corresponding process group.

    Parameters
    ----------
    x: torch.Tensor
        Data to roll.
    shifts: Tuple[int, ...]
        The number of places by which the elements of the tensor are shifted. If shifts is a tuple, dims must be a tuple of the same size, and each dimension will be rolled by the corresponding value.
    dims: Tuple[int, ...]
        Axes along which to roll.
    distributed_dims: list[int]
        Axes to roll, upon which the data is spatially distributed
    spatial_process_groups: list[dist.ProcessGroup]
        Corresponding process groups to distributed_dims.
    empty: torch.Tensor
        Empty tensor for computation purposes.

    Returns
    -------
    x: torch.Tensor
        Rolled tensor.
    """
    x = torch.roll(x, shifts=shifts, dims=dims)

    dim_to_shift = dict(zip(dims, shifts))
    for d, pg in zip(distributed_dims, spatial_process_groups):
        if d not in dim_to_shift:
            raise ValueError(f"distributed dim {d} not in dims {dims}")
        x = exchange_halo_1d(
            x,
            dim=d,
            window_size=dim_to_shift[d],
            group=pg,
            empty=empty,
            comm_halo_1d_fn=exchange_halo_1d_fn,
        )
    return x
