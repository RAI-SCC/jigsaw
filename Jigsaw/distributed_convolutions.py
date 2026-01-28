import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Tuple


def calc_halo_sizes(
    x_shape: torch.tensor,
    kernel_size: int,
    stride: int,
):
    # global indices where you want to apply the convolutions:
    global_size = int(x_shape[-1] * 2)
    global_steps = torch.arange(0, global_size - kernel_size + 1, step=stride)
    len_global_size = global_steps.shape[-1]
    mid_index = len_global_size // 2 - 1

    right_to_left = max((global_steps[mid_index] + kernel_size) - x_shape[-1], 0)

    left_to_right = (
        global_steps[mid_index + 1] - x_shape[-1]
        if global_steps[mid_index + 1] < x_shape[-1]
        else 0
    )
    left_to_right = left_to_right * -1

    return right_to_left, left_to_right


def get_left_right_local_ranks(parallelism: int):
    if parallelism == 2:
        return [[0], [1]]
    elif parallelism == 4:
        return [[0, 1], [2, 3]]


def communicate_halos(
    x: torch.tensor,
    right_to_left: int,
    left_to_right: int,
    local_rank: int,
    parallelism: int,
    empty: torch.tensor,
    spatial_group: dist.ProcessGroup,
    async_op: bool,
):
    left_right_local_ranks = get_left_right_local_ranks(parallelism)
    if local_rank in left_right_local_ranks[0]:
        send = x[..., -left_to_right:] if left_to_right > 0 else empty
        recv = (
            torch.zeros((*x.shape[:-1], right_to_left), device=x.device, dtype=x.dtype)
            if right_to_left > 0
            else empty
        )
        send = send.contiguous()
        if async_op:
            wait_handle = dist.all_to_all(
                [empty, recv], [empty, send], group=spatial_group, async_op=async_op
            )
            return recv, wait_handle
        else:
            dist.all_to_all(
                [empty, recv], [empty, send], group=spatial_group, async_op=async_op
            )
            return recv, None
    elif local_rank in left_right_local_ranks[1]:
        send = x[..., :right_to_left] if right_to_left > 0 else empty
        recv = (
            torch.zeros((*x.shape[:-1], left_to_right), device=x.device, dtype=x.dtype)
            if left_to_right > 0
            else empty
        )
        send = send.contiguous()
        if async_op:
            wait_handle = dist.all_to_all(
                [recv, empty], [send, empty], group=spatial_group, async_op=async_op
            )
            return recv, wait_handle
        else:
            dist.all_to_all(
                [recv, empty], [send, empty], group=spatial_group, async_op=async_op
            )
            return recv, None


def conv1dTranspose_halocalc_twoway(
    x: torch.tensor,
    y: torch.tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    local_rank: int,
    spatial_group: dist.ProcessGroup,
    parallelism: int
) -> torch.tensor:
    """Distributed 1d convolution"""
    # x shape

    global_shape = (
        (x.shape[-1] * 2 - 1) * stride - 2 * padding + dilation * (kernel_size - 1)
    ) + 1
    # midway_index = global_shape//2-1
    y_shape = y.shape[-1]
    overlap_size = y_shape * 2 - global_shape
    if global_shape % 2 == 0:
        local_size_left, local_size_right = global_shape // 2, global_shape // 2
    else:
        local_size_left, local_size_right = global_shape // 2, global_shape // 2 + 1

    empty = torch.empty(0, device=x.device, dtype=x.dtype)
    
    left_right_local_ranks = get_left_right_local_ranks(parallelism)
    
    if local_rank in left_right_local_ranks[0]:
        send = y[..., -overlap_size:] if overlap_size > 0 else empty
        recv = (
            torch.zeros(
                (*y.shape[:-1], overlap_size), device=x.device, dtype=x.dtype
            )
            if overlap_size > 0
            else empty
        )
        send = send.contiguous()
        dist.all_to_all(
            [empty, recv], [empty, send], group=spatial_group, async_op=False
        )
    elif local_rank in left_right_local_ranks[1]:
        send = y[..., :overlap_size] if overlap_size > 0 else empty
        recv = (
            torch.zeros(
                (*y.shape[:-1], overlap_size), device=x.device, dtype=x.dtype
            )
            if overlap_size > 0
            else empty
        )
        send = send.contiguous()
        
        dist.all_to_all(
            [recv, empty], [send, empty], group=spatial_group, async_op=False
        )

    return recv, overlap_size, local_size_left, local_size_right




def append_halos(
    x: torch.tensor,
    halo: torch.tensor,
    right_to_left: int,
    left_to_right: int,
    local_rank: int,
    parallelism: int,
    empty: torch.tensor,
):
    left_right_local_ranks = get_left_right_local_ranks(parallelism)

    if local_rank in left_right_local_ranks[0]:
        x_with_halos = torch.cat((x, halo if right_to_left > 0 else empty), dim=-1)
    elif local_rank in left_right_local_ranks[1]:
        x_with_halos = torch.cat((halo if left_to_right > 0 else empty, x), dim=-1)
    return x_with_halos


def distributed_conv1d_over_channels(
    x: torch.tensor,
    w: torch.tensor,
    local_rank: int,
    in_channels_global: int,
    stride: int,
    padding: int,
    dilation: int,
    empty: torch.tensor,
    channel_group: dist.ProcessGroup,
):
    x_buffer = torch.zeros(x.shape, device=x.device, dtype=x.dtype)

    # TODO: will need adjustment after including BNN
    match local_rank:
        case 0 | 2:
            x_handle = dist.all_to_all(
                [empty, x_buffer],
                [empty, x],
                group=channel_group,
                async_op=True,
            )
            y = F.conv1d(
                x,
                w[:, : in_channels_global // 2],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            x_handle.wait()
            y = y + F.conv1d(
                x_buffer,
                w[:, in_channels_global // 2 :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            return y, x, x_buffer
        case 1 | 3:
            x_handle = dist.all_to_all(
                [x_buffer, empty],
                [x, empty],
                group=channel_group,
                async_op=True,
            )
            y = F.conv1d(
                x,
                w[:, in_channels_global // 2 :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            x_handle.wait()
            y = y + F.conv1d(
                x_buffer,
                w[:, : in_channels_global // 2],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            return y, x_buffer, x


def distributed_convTranspose1d_over_channels(
    x: torch.tensor,
    w: torch.tensor,
    local_rank: int,
    in_channels_global: int,
    stride: int,
    padding: int,
    dilation: int,
    empty: torch.tensor,
    channel_group: dist.ProcessGroup,
):
    x_buffer = torch.zeros(x.shape, device=x.device, dtype=x.dtype)

    # TODO: will need adjustment after including BNN
    match local_rank:
        case 0 | 2:
            x_handle = dist.all_to_all(
                [empty, x_buffer],
                [empty, x],
                group=channel_group,
                async_op=True,
            )
            y = F.conv_transpose1d(
                x,
                w[:, : in_channels_global // 2],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            x_handle.wait()
            y = y + F.conv_transpose1d(
                x_buffer,
                w[:, in_channels_global // 2 :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            return y, x, x_buffer
        case 1 | 3:
            x_handle = dist.all_to_all(
                [x_buffer, empty],
                [x, empty],
                group=channel_group,
                async_op=True,
            )
            y = F.conv_transpose1d(
                x,
                w[:, in_channels_global // 2 :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            x_handle.wait()
            y = y + F.conv_transpose1d(
                x_buffer,
                w[:, : in_channels_global // 2],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            return y, x_buffer, x


def distributed_conv2d_over_channels(
    x: torch.tensor,
    w: torch.tensor,
    local_rank: int,
    in_channels_global: int,
    stride: Tuple[int, int],
    padding: int,
    dilation: int,
    empty: torch.tensor,
    channel_group: dist.ProcessGroup,
):
    x_buffer = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
    if len(x.shape) == 5: # TODO: is there a better way to flag "if sampling for UQ is required"?
        conv_func = torch.vmap(F.conv2d)
    else:
        conv_func = F.conv2d
    match local_rank:
        case 0 | 2:
            x_handle = dist.all_to_all(
                [empty, x_buffer],
                [empty, x],
                group=channel_group,
                async_op=True,
            )
            y = conv_func(
                x,
                w[..., : in_channels_global // 2, :, :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            x_handle.wait()
            y = y + conv_func(
                x_buffer, # (sampling, batch, in_channels, x, y)
                w[..., in_channels_global // 2 :, :, :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            return y, x, x_buffer
        case 1 | 3:
            x_handle = dist.all_to_all(
                [x_buffer, empty],
                [x, empty],
                group=channel_group,
                async_op=True,
            )
            y = conv_func(
                x,
                w[..., in_channels_global // 2 :, :, :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            x_handle.wait()
            
            y = y + conv_func(
                x_buffer,
                w[..., : in_channels_global // 2, :, :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            
            return y, x_buffer, x


def send_g_x_channels(
    g_x: torch.tensor,
    buffer: torch.tensor,
    empty: torch.tensor,
    in_channels_global: int,
    channel_group: dist.ProcessGroup,
    local_rank: int,
    async_op: bool,
):
    match local_rank:
        case 0 | 2:
            send = g_x[..., in_channels_global // 2 :, :]
            send = send.contiguous()
            wait_handle = dist.all_to_all(
                [empty, buffer],
                [empty, send],
                group=channel_group,
                async_op=async_op,
            )

        case 1 | 3:
            send = g_x[..., : in_channels_global // 2, :]
            send = send.contiguous()
            wait_handle = dist.all_to_all(
                [buffer, empty],
                [send, empty],
                group=channel_group,
                async_op=async_op,
            )
    return buffer, wait_handle


def send_g_x_channels_2d(
    g_x: torch.tensor,
    buffer: torch.tensor,
    empty: torch.tensor,
    in_channels_global: int,
    channel_group: dist.ProcessGroup,
    local_rank: int,
    async_op: bool,
):
    match local_rank:
        case 0 | 2:
            send = g_x[..., in_channels_global // 2 :, :, :]
            send = send.contiguous()
            wait_handle = dist.all_to_all(
                [empty, buffer],
                [empty, send],
                group=channel_group,
                async_op=async_op,
            )

        case 1 | 3:
            send = g_x[..., : in_channels_global // 2, :, :]
            send = send.contiguous()
            wait_handle = dist.all_to_all(
                [buffer, empty],
                [send, empty],
                group=channel_group,
                async_op=async_op,
            )
    return buffer, wait_handle


def conv1d_twoway(
    x: torch.tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    rank: int,
    model_group: dist.ProcessGroup,
) -> torch.tensor:
    """Distributed 1d convolution"""
    # x shape
    mod = rank % 2

    # global indices where you want to apply the convolutions:
    global_size = int(x.shape[-1] * 2)
    global_steps = torch.arange(0, global_size - kernel_size + 1, step=stride)
    len_global_size = global_steps.shape[-1]

    # midpoint calculation
    mid_index = (
        len_global_size // 2
        - 1  # if len_global_size % 2 == 0 else len_global_size // 2
    )

    right_to_left = max((global_steps[mid_index] + kernel_size) - x.shape[-1], 0)

    # left is how much you get from rank 1 --> 0
    # right = kernel_size - (global_steps[mid_index+1])
    # left_to_right = overlap - right_to_left
    left_to_right = (
        global_steps[mid_index + 1] - x.shape[-1]
        if global_steps[mid_index + 1] < x.shape[-1]
        else 0
    )
    left_to_right = left_to_right * -1

    B, C, L = x.shape
    empty = torch.empty(0, device=x.device, dtype=x.dtype)

    match mod:
        case 0:
            send = x[..., -left_to_right:] if left_to_right > 0 else empty
            recv = (
                torch.zeros((B, C, right_to_left), device=x.device, dtype=x.dtype)
                if right_to_left > 0
                else empty
            )
            send = send.contiguous()
            dist.all_to_all(
                [empty, recv], [empty, send], group=model_group, async_op=False
            )
            x_augment = torch.cat((x, recv if right_to_left > 0 else empty), dim=-1)
        case 1:
            send = x[..., :right_to_left] if right_to_left > 0 else empty
            recv = (
                torch.zeros((B, C, left_to_right), device=x.device, dtype=x.dtype)
                if left_to_right > 0
                else empty
            )
            send = send.contiguous()
            dist.all_to_all(
                [recv, empty], [send, empty], group=model_group, async_op=False
            )
            x_augment = torch.cat((recv if left_to_right > 0 else empty, x), dim=-1)

    return x_augment, L, right_to_left, left_to_right


def conv1d_twoway_channel(
    x: torch.tensor,
    w: torch.tensor,
    in_channels: int,
    stride: int,
    padding: int,
    dilation: int,
    rank: int,
    model_group: dist.ProcessGroup,
) -> torch.tensor:
    """Distributed 1d convolution across channels"""
    # x shape
    mod = rank % 2
    # communicate x1, x0
    empty = torch.empty(0, device=x.device, dtype=x.dtype)
    x_buffer = torch.zeros(x.shape, device=x.device, dtype=x.dtype)

    match mod:
        case 0:
            y = F.conv1d(
                x,
                w[:, : in_channels // 2],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            dist.all_to_all(
                [empty, x_buffer], [empty, x], group=model_group, async_op=False
            )
            y = y + F.conv1d(
                x_buffer,
                w[:, in_channels // 2 :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            return y, x, x_buffer
        case 1:
            y = F.conv1d(
                x,
                w[:, in_channels // 2 :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            dist.all_to_all(
                [x_buffer, empty], [x, empty], group=model_group, async_op=False
            )
            y = y + F.conv1d(
                x_buffer,
                w[:, : in_channels // 2],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            return y, x_buffer, x


def convTranspose1d_twoway_channel(
    x: torch.tensor,
    w: torch.tensor,
    in_channels: int,
    stride: int,
    padding: int,
    dilation: int,
    rank: int,
    model_group: dist.ProcessGroup,
) -> torch.tensor:
    """Distributed 1d convolution across channels"""
    # x shape
    mod = rank % 2
    # communicate x1, x0
    empty = torch.empty(0, device=x.device, dtype=x.dtype)
    x_buffer = torch.zeros(x.shape, device=x.device, dtype=x.dtype)

    # TODO: dimension shapes may be modified after combining with torch_blue
    match mod:
        case 0:
            y = F.conv_transpose1d(
                x,
                w[: in_channels // 2],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            dist.all_to_all(
                [empty, x_buffer], [empty, x], group=model_group, async_op=False
            )
            y = y + F.conv_transpose1d(
                x_buffer,
                w[in_channels // 2 :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            return y, x, x_buffer
        case 1:
            y = F.conv_transpose1d(
                x,
                w[in_channels // 2 :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            dist.all_to_all(
                [x_buffer, empty], [x, empty], group=model_group, async_op=False
            )
            y = y + F.conv_transpose1d(
                x_buffer,
                w[: in_channels // 2],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            return y, x_buffer, x
        
def convTranspose2d_twoway_channel(
    x: torch.tensor,
    w: torch.tensor,
    in_channels: int,
    stride: int,
    padding: int,
    dilation: int,
    rank: int,
    channel_group: dist.ProcessGroup,
) -> torch.tensor:
    """Distributed 2d convolution across channels"""
    # x shape
    mod = rank % 2
    # communicate x1, x0
    empty = torch.empty(0, device=x.device, dtype=x.dtype)
    x_buffer = torch.zeros(x.shape, device=x.device, dtype=x.dtype)

    if len(x.shape) == 5:        
        conv_func = torch.vmap(F.conv_transpose2d)
    else:
        conv_func = F.conv_transpose2d

    match mod:
        case 0:
            y = conv_func(
                x,
                w[...,: in_channels // 2,:,:,:],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            dist.all_to_all(
                [empty, x_buffer], [empty, x], group=channel_group, async_op=False
            )
            y = y + conv_func(
                x_buffer,
                w[...,in_channels // 2 :,:,:,:],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            return y, x, x_buffer
        case 1:
            y = conv_func(
                x,
                w[...,in_channels // 2 :,:,:,:],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            dist.all_to_all(
                [x_buffer, empty], [x, empty], group=channel_group, async_op=False
            )
            y = y + conv_func(
                x_buffer,
                w[...,: in_channels // 2,:,:,:],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            return y, x_buffer, x


def conv1d_fourway_complete(
    x: torch.Tensor,
    w: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    in_channels: int,
    rank: int,
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    empty: torch.Tensor,
) -> torch.Tensor:
    """Distributed 1d convolution"""
    # Setup
    local_model_rank = rank % 4

    # communicate x1, x0
    right_to_left, left_to_right = calc_halo_sizes(x.shape, kernel_size, stride)

    # Spatial split
    halo, _ = communicate_halos(
        x,
        right_to_left,
        left_to_right,
        local_model_rank,
        4,
        empty,
        spatial_group,
        async_op=False,
    )
    x_with_halos = append_halos(
        x, halo, right_to_left, left_to_right, local_model_rank, 4, empty
    )

    (
        y,
        x_left,
        x_right,
    ) = distributed_conv1d_over_channels(
        x_with_halos,
        w,
        local_model_rank,
        in_channels,
        stride,
        padding,
        dilation,
        empty,
        channel_group,
    )
    return y, x_left, x_right, right_to_left, left_to_right


def convTranspose1d_fourway_complete(
    x: torch.Tensor,
    w: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    in_channels: int,
    rank: int,
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    empty: torch.Tensor,
) -> torch.Tensor:
    """Distributed 1d convolution"""
    # Setup
    local_model_rank = rank % 4

    # communicate x1, x0
    right_to_left, left_to_right = calc_halo_sizes(x.shape, kernel_size, stride)

    # Spatial split
    halo, _ = communicate_halos(
        x,
        right_to_left,
        left_to_right,
        local_model_rank,
        4,
        empty,
        spatial_group,
        async_op=False,
    )
    x_with_halos = append_halos(
        x, halo, right_to_left, left_to_right, local_model_rank, 4, empty
    )

    (
        y,
        x_left,
        x_right,
    ) = distributed_convTranspose1d_over_channels(
        x_with_halos,
        w,
        local_model_rank,
        in_channels,
        stride,
        padding,
        dilation,
        empty,
        channel_group,
    )
    return y, x_left, x_right, right_to_left, left_to_right


def conv2d_twoway_channel(
    x: torch.tensor,
    w: torch.tensor,
    in_channels: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    rank: int = 0,
    model_group: dist.ProcessGroup = None,
) -> torch.tensor:
    """Distributed 2d convolution across channels (two-way split)"""
    ## TODO : include bias as well
    # x shape
    mod = rank % 2
    # communicate x1, x0
    empty = torch.empty(0, device=x.device, dtype=x.dtype)
    x_buffer = torch.zeros(x.shape, device=x.device, dtype=x.dtype)

    match mod:
        case 0:
            y = F.conv2d(
                x,
                w[:, : in_channels // 2],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            dist.all_to_all(
                [empty, x_buffer], [empty, x], group=model_group, async_op=False
            )
            y = y + F.conv2d(
                x_buffer,
                w[:, in_channels // 2 :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            return y, x, x_buffer
        case 1:
            y = F.conv2d(
                x,
                w[:, in_channels // 2 :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            dist.all_to_all(
                [x_buffer, empty], [x, empty], group=model_group, async_op=False
            )
            y = y + F.conv2d(
                x_buffer,
                w[:, : in_channels // 2],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            return y, x_buffer, x


def conv2d_twoway(
    x: torch.tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    rank: int,
    model_group: dist.ProcessGroup,
) -> torch.tensor:
    """Distributed 1d convolution"""
    # x shape

    overlap = max(
        kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1) - stride[1], 0
    )
    mod = rank % 2

    # global indices where you want to apply the convolutions:
    global_size = int(x.shape[-1] * 2)
    global_steps = torch.arange(0, global_size - kernel_size[1], step=stride[1])
    len_global_size = global_steps.shape[-1]

    mid_index = (
        len_global_size // 2 - 1 if len_global_size % 2 == 0 else len_global_size // 2
    )

    right_to_left = (global_steps[mid_index] + kernel_size[1]) - x.shape[
        -1
    ]  # right is how much you get from rank 1 gets

    left_to_right = overlap - right_to_left
    B, C, H, W = x.shape

    empty = torch.empty(0, device=x.device, dtype=x.dtype)

    match mod:
        case 0:
            send = x[..., -left_to_right:] if left_to_right > 0 else empty
            recv = (
                torch.zeros((B, C, H, right_to_left), device=x.device, dtype=x.dtype)
                if right_to_left > 0
                else empty
            )
            send = send.contiguous()
            dist.all_to_all(
                [empty, recv], [empty, send], group=model_group, async_op=False
            )
            x_augment = torch.cat((x, recv if right_to_left > 0 else empty), dim=-1)
        case 1:
            send = x[..., :right_to_left] if right_to_left > 0 else empty
            recv = (
                torch.zeros((B, C, H, left_to_right), device=x.device, dtype=x.dtype)
                if left_to_right > 0
                else empty
            )
            send = send.contiguous()
            dist.all_to_all(
                [recv, empty], [send, empty], group=model_group, async_op=False
            )
            x_augment = torch.cat((recv if left_to_right > 0 else empty, x), dim=-1)
    return x_augment, H, W


def global_to_local_mapping(
    h_size: int,
    w_size: int,
    rank: int,
    world_size: int,
    global_index_h: int,
    global_index_w: int,
):
    if rank in [2, 3]:
        global_mapping_y = global_index_h - h_size
    else:
        global_mapping_y = global_index_h
    if rank in [1, 3]:
        global_mapping_x = global_index_w - w_size
    else:
        global_mapping_x = global_index_w
    global_mapping_x = 0 if global_mapping_x < 0 else global_mapping_x
    global_mapping_y = 0 if global_mapping_y < 0 else global_mapping_y
    return global_mapping_y, global_mapping_x


def conv2d_fourway(
    x: torch.tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    rank: int,
    model_group: dist.ProcessGroup,
) -> torch.tensor:
    """Distributed 2d convolution"""
    # x shape

    h_size, w_size = x.shape[-2], x.shape[-1]
    mod = rank % 4

    # global indices where you want to apply the convolutions:
    global_size_h = int(x.shape[-2] * 2)
    global_size_w = int(x.shape[-1] * 2)
    global_steps_h = torch.arange(0, global_size_h - kernel_size[0] + 1, step=stride[0])
    global_steps_w = torch.arange(0, global_size_w - kernel_size[1] + 1, step=stride[1])

    len_global_size_y = global_steps_h.shape[-1]
    len_global_size_x = global_steps_w.shape[-1]
    # midpoint calculation
    mid_index_h = (
        len_global_size_y // 2 - 1
        if len_global_size_y % 2 == 0
        else len_global_size_y // 2
    )
    mid_index_w = (
        len_global_size_x // 2 - 1
        if len_global_size_x % 2 == 0
        else len_global_size_x // 2
    )

    h_bottom_to_top = (global_steps_h[mid_index_h] + kernel_size[0]) - x.shape[-2]
    if h_bottom_to_top < 0:
        h_bottom_to_top = 0
    h_top_to_bottom = x.shape[-2] - global_steps_h[mid_index_h + 1]
    if h_top_to_bottom < 0:
        h_top_to_bottom = 0

    w_right_to_left = (global_steps_w[mid_index_w] + kernel_size[1]) - x.shape[-1]
    if w_right_to_left < 0:
        w_right_to_left = 0
    w_left_to_right = (x.shape[-1]) - global_steps_w[mid_index_w + 1]
    if w_left_to_right < 0:
        w_left_to_right = 0


    B, C, H, W = x.shape

    empty = torch.empty(0, device=x.device, dtype=x.dtype)

    match mod:
        case 0:
            send_1 = x[..., -w_left_to_right:] if w_left_to_right > 0 else empty
            send_2 = x[..., -h_top_to_bottom:, :] if h_top_to_bottom > 0 else empty
            send_3 = (
                x[..., -h_top_to_bottom:, -w_left_to_right:]
                if (w_left_to_right > 0 and h_top_to_bottom > 0)
                else empty
            )
            recv_1 = (
                torch.zeros(
                    (B, C, H, w_right_to_left),
                    device=x.device,
                    dtype=x.dtype,
                )
                if w_right_to_left > 0
                else empty
            )
            recv_2 = (
                torch.zeros(
                    (B, C, h_bottom_to_top, W),
                    device=x.device,
                    dtype=x.dtype,
                )
                if h_bottom_to_top > 0
                else empty
            )
            recv_3 = (
                torch.zeros(
                    (
                        B,
                        C,
                        h_bottom_to_top,
                        w_right_to_left,
                    ),
                    device=x.device,
                    dtype=x.dtype,
                )
                if (h_bottom_to_top > 0 and w_right_to_left > 0)
                else empty
            )
            send_1, send_2, send_3 = (
                send_1.contiguous(),
                send_2.contiguous(),
                send_3.contiguous(),
            )

            dist.all_to_all(
                [empty, recv_1, recv_2, recv_3],
                [empty, send_1, send_2, send_3],
                group=model_group,
                async_op=False,
            )
            x_augment_1 = torch.cat(
                (x, recv_1 if w_right_to_left > 0 else empty), dim=-1
            )
            x_augment_2 = torch.cat((recv_2, recv_3), dim=-1)
            x_augment = torch.cat((x_augment_1, x_augment_2), dim=-2).contiguous()
        case 1:
            send_0 = x[..., :w_right_to_left] if w_right_to_left > 0 else empty
            send_2 = (
                x[..., -h_top_to_bottom:, :w_right_to_left]
                if (h_top_to_bottom > 0 and w_right_to_left > 0)
                else empty
            )
            send_3 = x[..., -h_top_to_bottom:, :] if h_top_to_bottom > 0 else empty
            recv_0 = (
                torch.zeros(
                    (B, C, H, w_left_to_right),
                    device=x.device,
                    dtype=x.dtype,
                )
                if w_left_to_right > 0
                else empty
            )
            recv_2 = (
                torch.zeros(
                    (
                        B,
                        C,
                        h_bottom_to_top,
                        w_left_to_right,
                    ),
                    device=x.device,
                    dtype=x.dtype,
                )
                if (h_bottom_to_top > 0 and w_left_to_right > 0)
                else empty
            )
            recv_3 = (
                torch.zeros(
                    (B, C, h_bottom_to_top, W),
                    device=x.device,
                    dtype=x.dtype,
                )
                if h_bottom_to_top > 0
                else empty
            )
            send_0, send_2, send_3 = (
                send_0.contiguous(),
                send_2.contiguous(),
                send_3.contiguous(),
            )

            dist.all_to_all(
                [recv_0, empty, recv_2, recv_3],
                [send_0, empty, send_2, send_3],
                group=model_group,
                async_op=False,
            )
            x_augment_1 = torch.cat((recv_0, x), dim=-1)
            x_augment_2 = torch.cat((recv_2, recv_3), dim=-1)
            x_augment = torch.cat((x_augment_1, x_augment_2), dim=-2).contiguous()
            h_coord, w_coord = global_to_local_mapping(
                h_size,
                w_size,
                rank,
                4,
                global_steps_h[mid_index_h],
                global_steps_w[mid_index_w + 1],
            )
            x_augment = x_augment[..., w_coord:]
        case 2:
            send_0 = x[..., :h_bottom_to_top, :] if h_bottom_to_top > 0 else empty
            send_1 = (
                x[..., :h_bottom_to_top, -w_left_to_right:]
                if (h_bottom_to_top > 0 and w_left_to_right > 0)
                else empty
            )
            
            send_3 = x[..., -w_left_to_right:] if w_left_to_right > 0 else empty
            recv_0 = (
                torch.zeros(
                    (B, C, h_top_to_bottom, W),
                    device=x.device,
                    dtype=x.dtype,
                )
                if h_top_to_bottom > 0
                else empty
            )
            recv_1 = (
                torch.zeros(
                    (B, C, h_top_to_bottom, w_right_to_left),
                    device=x.device,
                    dtype=x.dtype,
                )
                if h_top_to_bottom > 0 and w_right_to_left > 0
                else empty
            )
            recv_3 = (
                torch.zeros(
                    (B, C, H, w_right_to_left),
                    device=x.device,
                    dtype=x.dtype,
                )
                if w_right_to_left > 0
                else empty
            )
            send_0, send_1, send_3 = (
                send_0.contiguous(),
                send_1.contiguous(),
                send_3.contiguous(),
            )

            dist.all_to_all(
                [recv_0, recv_1, empty, recv_3],
                [send_0, send_1, empty, send_3],
                group=model_group,
                async_op=False,
            )
            x_augment_1 = torch.cat((recv_0, recv_1), dim=-1)
            x_augment_2 = torch.cat((x, recv_3), dim=-1)
            x_augment = torch.cat((x_augment_1, x_augment_2), dim=-2).contiguous()
            h_coord, w_coord = global_to_local_mapping(
                h_size,
                w_size,
                rank,
                4,
                global_steps_h[mid_index_h + 1],
                global_steps_w[mid_index_w],
            )
            x_augment = x_augment[..., h_coord:, :]
        case 3:
            send_0 = (
                x[..., :h_bottom_to_top, :w_right_to_left]
                if (h_bottom_to_top > 0 and w_right_to_left > 0)
                else empty
            )
            send_1 = x[..., :h_bottom_to_top, :] if h_bottom_to_top > 0 else empty
            send_2 = x[..., :w_right_to_left] if w_right_to_left > 0 else empty
            recv_0 = (
                torch.zeros(
                    (B, C, h_top_to_bottom, w_left_to_right),
                    device=x.device,
                    dtype=x.dtype,
                )
                if (h_top_to_bottom > 0 and w_left_to_right > 0)
                else empty
            )
            recv_1 = (
                torch.zeros(
                    (B, C, h_top_to_bottom, W),
                    device=x.device,
                    dtype=x.dtype,
                )
                if h_top_to_bottom > 0
                else empty
            )
            recv_2 = (
                torch.zeros(
                    (B, C, H, w_left_to_right),
                    device=x.device,
                    dtype=x.dtype,
                )
                if w_left_to_right > 0
                else empty
            )
            send_0, send_1, send_2 = (
                send_0.contiguous(),
                send_1.contiguous(),
                send_2.contiguous(),
            )

            dist.all_to_all(
                [recv_0, recv_1, recv_2, empty],
                [send_0, send_1, send_2, empty],
                group=model_group,
                async_op=False,
            )

            x_augment_1 = torch.cat((recv_0, recv_1), dim=-1)
            x_augment_2 = torch.cat((recv_2, x), dim=-1)
            x_augment = torch.cat((x_augment_1, x_augment_2), dim=-2).contiguous()
            h_coord, w_coord = global_to_local_mapping(
                h_size,
                w_size,
                rank,
                4,
                global_steps_h[mid_index_h + 1],
                global_steps_w[mid_index_w + 1],
            )

            x_augment = x_augment[..., h_coord:, w_coord:]

    return x_augment, H, W


def conv2d_fourway_complete(
    x: torch.Tensor,
    w: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: int,
    padding: int,
    dilation: int,
    in_channels: int,
    rank: int,
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    empty: torch.Tensor,
) -> torch.Tensor:
    """Distributed 2d convolution"""
    # Setup
    local_model_rank = rank % 4

    # communicate x1, x0
    right_to_left, left_to_right = calc_halo_sizes(x.shape, kernel_size[1], stride[1])

    # Spatial split
    halo, _ = communicate_halos(
        x,
        right_to_left,
        left_to_right,
        local_model_rank,
        4,
        empty,
        spatial_group,
        async_op=False,
    )
    x_with_halos = append_halos(
        x, halo, right_to_left, left_to_right, local_model_rank, 4, empty
    )
    (
        y,
        x_left,
        x_right,
    ) = distributed_conv2d_over_channels(
        x_with_halos,
        w,
        local_model_rank,
        in_channels,
        stride,
        padding,
        dilation,
        empty,
        channel_group,
    )
    return y, x_left, x_right, right_to_left, left_to_right
