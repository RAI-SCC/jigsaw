import torch
import torch.distributed as dist
from Jigsaw import distributed_convolutions as dist_conv

def get_halo(
    x: torch.Tensor,
    window_size: int,
    rank: int,
    device: torch.device,
    parallelism: int,
    empty: torch.Tensor,
    spatial_group: dist.ProcessGroup
):
    # get left rank
    # print(f'rank {rank} and window size is {window_size}')
    left_right_local_ranks = dist_conv.get_left_right_local_ranks(parallelism)
    if window_size > 0:
        halo_to_send = x[...,:window_size,:].contiguous()
    else:
        halo_to_send = x[...,window_size:,:].contiguous()
    recv = torch.zeros(*halo_to_send.shape, device=device, dtype=x.dtype)
    
    if rank in left_right_local_ranks[0]: # I am holding the left spatial dimensions
        dist.all_to_all(
            [empty, recv], [empty, halo_to_send], group=spatial_group, async_op=False
        )
        
    else:
        dist.all_to_all(
            [recv, empty], [halo_to_send, empty], group=spatial_group, async_op=False
        )
    
    if window_size > 0:
        x[...,:window_size,:] = recv
    else:
        x[...,window_size:,:] = recv
    
    return x
        

def roll2d(
    x: torch.Tensor,
    shifts,
    dims,
    rank: int,
    device: torch.device,
    empty: torch.Tensor,
    spatial_group: dist.ProcessGroup
):
    # For now, assume that dims is always (-3, -2)
    # roll first
    # TODO adjust rolling dimension
    # print(f'here! shifts is {shifts} and x has shape {x.shape}')
    x = torch.roll(x, shifts=shifts, dims=(-3,-2))
    
    x = get_halo(x, shifts[1], rank, device, 4, empty, spatial_group)
    
    return x
    
    
    