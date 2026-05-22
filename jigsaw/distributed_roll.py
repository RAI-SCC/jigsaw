import torch
import torch.distributed as dist

from typing import Tuple

from jigsaw import distributed_convolutions as dist_conv


def exchange_halo_1d(
    x: torch.Tensor,
    dim: int,
    window_size: int,
    group: dist.ProcessGroup,
    device: torch.device,
    empty: torch.Tensor,
) -> torch.Tensor:
    """Exchange a halo of `window_size` along tensor `dim` over a 1D process group.

    Positive window_size: send leading slice to rank-1, receive into leading slice from rank+1.
    Negative window_size: send trailing slice to rank+1, receive into trailing slice from rank-1.
    """
    if window_size == 0:
        return x

    rank = group.rank()
    world = group.size()

    sl = [slice(None)] * x.ndim
    sl[dim] = slice(0, window_size) if window_size > 0 else slice(window_size, None)
    sl = tuple(sl)

    halo_to_send = x[sl].contiguous()
    recv = torch.zeros_like(halo_to_send)

    if window_size > 0:
        send_to = (rank - 1) % world
        recv_from = (rank + 1) % world
    else:
        send_to = (rank + 1) % world
        recv_from = (rank - 1) % world

    send_list = [empty] * world
    recv_list = [empty] * world
    send_list[send_to] = halo_to_send
    recv_list[recv_from] = recv

    dist.all_to_all(recv_list, send_list, group=group, async_op=False)

    x[sl] = recv
    return x


def distributed_roll(
    x: torch.Tensor,
    shifts: Tuple[int, ...],
    dims: Tuple[int, ...],
    distributed_dims: List[int],
    spatial_process_groups: List[dist.ProcessGroup],
    device: torch.device,
    empty: torch.Tensor,
) -> torch.Tensor:
    """Roll `x` by `shifts` along `dims`; for each dim in `distributed_dims`,
    exchange a halo on the corresponding process group."""
    x = torch.roll(x, shifts=shifts, dims=dims)

    dim_to_shift = dict(zip(dims, shifts))
    for d, pg in zip(distributed_dims, spatial_process_groups):
        if d not in dim_to_shift:
            raise ValueError(f"distributed dim {d} not in dims {dims}")
        x = exchange_halo_1d(
            x, dim=d, window_size=dim_to_shift[d], group=pg, device=device, empty=empty
        )
    return x