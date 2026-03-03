import torch
import torch.distributed as dist

from jigsaw import distributed_convolutions as dist_conv


def get_halo(
    x: torch.Tensor,
    window_size: int,
    rank: int,
    device: torch.device,
    parallelism: int,
    empty: torch.Tensor,
    spatial_group: dist.ProcessGroup,
):
    """Get halo for roll, 4way."""
    left_right_local_ranks = dist_conv.get_left_right_local_ranks(parallelism)
    if window_size > 0:
        halo_to_send = x[..., :window_size, :].contiguous()
    else:
        halo_to_send = x[..., window_size:, :].contiguous()
    recv = torch.zeros(*halo_to_send.shape, device=device, dtype=x.dtype)

    if rank in left_right_local_ranks[0]:  # I am holding the left spatial dimensions
        dist.all_to_all(
            [empty, recv], [empty, halo_to_send], group=spatial_group, async_op=False
        )

    else:
        dist.all_to_all(
            [recv, empty], [halo_to_send, empty], group=spatial_group, async_op=False
        )

    if window_size > 0:
        x[..., :window_size, :] = recv
    else:
        x[..., window_size:, :] = recv

    return x


def get_halo_8way(
    x: torch.Tensor,
    window_size: int,
    local_rank: int,
    device: torch.device,
    empty: torch.Tensor,
    spatial_group: dist.ProcessGroup,
):
    """Get halo for roll, 8way."""
    local_rank = spatial_group.rank()
    if window_size > 0:
        halo_to_send = x[..., :window_size, :].contiguous()
        recv = torch.zeros(*halo_to_send.shape, device=device, dtype=x.dtype)
        match local_rank:
            case 0:
                # recv, send
                dist.all_to_all(
                    [empty, empty, empty, recv],
                    [empty, halo_to_send, empty, empty],
                    group=spatial_group,
                    async_op=False,
                )
            case 1:
                dist.all_to_all(
                    [recv, empty, empty, empty],
                    [empty, empty, halo_to_send, empty],
                    group=spatial_group,
                    async_op=False,
                )
            case 2:
                dist.all_to_all(
                    [empty, recv, empty, empty],
                    [empty, empty, empty, halo_to_send],
                    group=spatial_group,
                    async_op=False,
                )
            case 3:
                dist.all_to_all(
                    [empty, empty, recv, empty],
                    [halo_to_send, empty, empty, empty],
                    group=spatial_group,
                    async_op=False,
                )
    else:
        halo_to_send = x[..., window_size:, :].contiguous()
        recv = torch.zeros(*halo_to_send.shape, device=device, dtype=x.dtype)
        match local_rank:
            case 0:
                # recv, send
                dist.all_to_all(
                    [empty, recv, empty, empty],
                    [empty, empty, empty, halo_to_send],
                    group=spatial_group,
                    async_op=False,
                )
            case 1:
                dist.all_to_all(
                    [empty, empty, recv, empty],
                    [halo_to_send, empty, empty, empty],
                    group=spatial_group,
                    async_op=False,
                )
            case 2:
                dist.all_to_all(
                    [empty, empty, empty, recv],
                    [empty, halo_to_send, empty, empty],
                    group=spatial_group,
                    async_op=False,
                )
            case 3:
                dist.all_to_all(
                    [recv, empty, empty, empty],
                    [empty, empty, halo_to_send, empty],
                    group=spatial_group,
                    async_op=False,
                )

    if window_size > 0:
        x[..., :window_size, :] = recv
    else:
        x[..., window_size:, :] = recv

    return x


def roll2d(
    x: torch.Tensor,
    shifts,
    dims,
    rank: int,
    device: torch.device,
    empty: torch.Tensor,
    spatial_group: dist.ProcessGroup,
):
    """Perform rolling in 2d, distributed across 4 GPUs."""
    x = torch.roll(x, shifts=shifts, dims=(-3, -2))

    x = get_halo(x, shifts[1], rank, device, 4, empty, spatial_group)

    return x


def roll2d_8way(
    x: torch.Tensor,
    shifts,
    dims,
    rank,
    device: torch.device,
    empty: torch.Tensor,
    spatial_group: dist.ProcessGroup,
):
    """Perform rolling in 2d, distributed across 8 GPUs."""
    x = torch.roll(x, shifts=shifts, dims=(-3, -2))

    x = get_halo_8way(x, shifts[1], rank, device, empty, spatial_group)

    return x
