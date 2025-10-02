import os

import torch
import torch.distributed as dist

def conv1d_twoway(
    x: torch.tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    rank: int,
    model_group: dist.ProcessGroup    
) -> torch.tensor:
    """Distributed 1d convolution"""
    # x shape
    
    overlap = max(kernel_size + (kernel_size - 1) * (dilation - 1) - stride, 0)
    mod = rank % 2
    
    # global indices where you want to apply the convolutions:
    global_size = int(x.shape[-1]*2)
    global_steps = torch.arange(0, global_size-kernel_size, step=stride)
    len_global_size = global_steps.shape[-1] 
    
    # uh
    mid_index = len_global_size//2 - 1 if len_global_size % 2 == 0 else len_global_size//2 
    
    #print(global_steps[mid_index], x.shape[-1])
#    local_steps = global_steps[0:mid_index] # i.e., [0, 3, 6]
    #print(f'Mid index: {mid_index}, global steps {global_steps}')
    # lets say x has shape [c, 10], with stride 3 and kernel size 10
    # final size would be [1, 10, 4]
    # on rank 0: you need [0--10], [3--13]
    # on rank 1: [6 -- 16] --> append indices 6, 7, 8, 9
    
    right = (global_steps[mid_index]+kernel_size)-x.shape[-1] # left is how m uch you get from rank 1 --> 0
    #right = kernel_size - (global_steps[mid_index+1])
    left = overlap - right
    #print(f"from right perspective: {right}, {global_steps[mid_index+1]}")
    #print(f'Rank {rank}, Right {right}, left {left}, total overlap {overlap}')
    B, C, L = x.shape
    empty = torch.empty(0, device=x.device, dtype=x.dtype)

    match mod:
        case 0:
            send = x[..., -right:] if right > 0 else empty
            recv = torch.zeros((B, C, right), device=x.device, dtype=x.dtype) if right > 0 else empty
            send = send.contiguous()
            dist.all_to_all([recv, empty], [empty, send], group=model_group, async_op=False)
            x_augment = torch.cat((x, recv if right > 0 else empty), dim=-1)
        case 1:
            send = x[..., :left] if left > 0 else empty
            recv = torch.zeros((B, C, left), device=x.device, dtype=x.dtype) if left > 0 else empty
            send = send.contiguous()
            dist.all_to_all([empty, recv], [send, empty], group=model_group, async_op=False)
            x_augment = torch.cat((recv if left > 0 else empty, x), dim=-1)
    
    return x_augment, L

def conv2d_twoway(
    x: torch.tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    rank: int,
    model_group: dist.ProcessGroup    
) -> torch.tensor:
    """Distributed 1d convolution"""
    # x shape
    
    overlap = max(kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1) - stride[1], 0)
    mod = rank % 2
    
    # global indices where you want to apply the convolutions:
    global_size = int(x.shape[-1]*2)
    global_steps = torch.arange(0, global_size-kernel_size[1], step=stride[1])
    len_global_size = global_steps.shape[-1] 
    
    # uh
    mid_index = len_global_size//2 - 1 if len_global_size % 2 == 0 else len_global_size//2 
    
    #print(global_steps[mid_index], x.shape[-1])
#    local_steps = global_steps[0:mid_index] # i.e., [0, 3, 6]
    #print(f'Mid index: {mid_index}, global steps {global_steps}')
    # lets say x has shape [c, 10], with stride 3 and kernel size 10
    # final size would be [1, 10, 4]
    # on rank 0: you need [0--10], [3--13]
    # on rank 1: [6 -- 16] --> append indices 6, 7, 8, 9
    
    right_to_left = (global_steps[mid_index]+kernel_size[1])-x.shape[-1] # right is how much you get from rank 1 gets
    #right = kernel_size - (global_steps[mid_index+1])
    left_to_right = overlap - right_to_left
    #print(f"from right perspective: {right}, {global_steps[mid_index+1]}")
    #print(f'Rank {rank}, Right {right}, left {left}, total overlap {overlap}')
    B, C, H, W = x.shape
    
    empty = torch.empty(0, device=x.device, dtype=x.dtype)

    match mod:
        case 0:
            send = x[..., -left_to_right:] if left_to_right > 0 else empty
            recv = torch.zeros((B, C, H, right_to_left), device=x.device, dtype=x.dtype) if right_to_left > 0 else empty
            send = send.contiguous()
            #print(f'Rank {rank} recv shape {recv.shape}, send shape {send.shape}, left {left_to_right}, right {right_to_left}')
            dist.all_to_all([recv, empty], [empty, send], group=model_group, async_op=False)
            x_augment = torch.cat((x, recv if right_to_left > 0 else empty), dim=-1)
        case 1:
            send = x[..., :right_to_left] if right_to_left > 0 else empty
            recv = torch.zeros((B, C, H, left_to_right), device=x.device, dtype=x.dtype) if left_to_right > 0 else empty
            send = send.contiguous()
            dist.all_to_all([empty, recv], [send, empty], group=model_group, async_op=False)
            x_augment = torch.cat((recv if left_to_right > 0 else empty, x), dim=-1)
            #print(f'Rank {rank} recv shape {recv.shape}, send shape {send.shape}')
    return x_augment, W

def conv2d_fourway(
    x: torch.tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    rank: int,
    model_group: dist.ProcessGroup    
) -> torch.tensor:
    """Distributed 2d convolution"""
    # x shape
    
    x_overlap = max(kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1) - stride[0], 0)
    y_overlap = max(kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1) - stride[1], 0)
    
    mod = rank % 4
    
    # global indices where you want to apply the convolutions:
    global_size_x = int(x.shape[-2]*2)
    global_size_y = int(x.shape[-1]*2)
    global_steps_x = torch.arange(0, global_size_x-kernel_size[0], step=stride[0])
    global_steps_y = torch.arange(0, global_size_y-kernel_size[1], step=stride[1])
    len_global_size_x = global_steps_x.shape[-1] 
    len_global_size_y = global_steps_y.shape[-1] 
    
    # uh
    mid_index_x = len_global_size_x//2 - 1 if len_global_size_x % 2 == 0 else len_global_size_x//2 
    mid_index_y = len_global_size_y//2 - 1 if len_global_size_y % 2 == 0 else len_global_size_y//2 
    
    #print(global_steps[mid_index], x.shape[-1])
#    local_steps = global_steps[0:mid_index] # i.e., [0, 3, 6]
    #print(f'Mid index: {mid_index}, global steps {global_steps}')
    # lets say x has shape [c, 10], with stride 3 and kernel size 10
    # final size would be [1, 10, 4]
    # on rank 0: you need [0--10], [3--13]
    # on rank 1: [6 -- 16] --> append indices 6, 7, 8, 9
    
    x_right_to_left = (global_steps_x[mid_index_x]+kernel_size[0])-x.shape[-2] # right is how much you get from rank 1 gets
    #right = kernel_size - (global_steps[mid_index+1])
    x_left_to_right = x_overlap - x_right_to_left
    
    y_right_to_left = (global_steps_x[mid_index_x]+kernel_size[1])-x.shape[-1]
    y_left_to_right = y_overlap - y_right_to_left
    
    #print(f"from right perspective: {right}, {global_steps[mid_index+1]}")
    #print(f'Rank {rank}, Right {right}, left {left}, total overlap {overlap}')
    B, C, H, W = x.shape
    
    empty = torch.empty(0, device=x.device, dtype=x.dtype)

    match mod:
        case 0:
            send = x[..., -x_left_to_right, -y_left_to_right:] if y_left_to_right > 0 else empty
            recv = torch.zeros((B, C, x_right_to_left, y_right_to_left), device=x.device, dtype=x.dtype) if y_right_to_left > 0 else empty
            send = send.contiguous()
            #print(f'Rank {rank} recv shape {recv.shape}, send shape {send.shape}, left {left_to_right}, right {right_to_left}')
            dist.all_to_all([recv, empty], [empty, send], group=model_group, async_op=False)
            x_augment = torch.cat((x, recv if x_right_to_left > 0 else empty), dim=-1) # TODO: how to append multiple dimensions?
        case 1:
            send = x[..., :right_to_left] if right_to_left > 0 else empty
            recv = torch.zeros((B, C, H, left_to_right), device=x.device, dtype=x.dtype) if left_to_right > 0 else empty
            send = send.contiguous()
            dist.all_to_all([empty, recv], [send, empty], group=model_group, async_op=False)
            x_augment = torch.cat((recv if left_to_right > 0 else empty, x), dim=-1)
            #print(f'Rank {rank} recv shape {recv.shape}, send shape {send.shape}')
    return x_augment, W