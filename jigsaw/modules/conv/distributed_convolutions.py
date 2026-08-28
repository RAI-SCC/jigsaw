"""Distributed convolution implementation."""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from jigsaw.modules.buffers import get_preallocated_buffer


def calc_halo_sizes(
    x_shape: torch.Size,
    kernel_size: int,
    stride: int,
) -> Any:
    """Calculate halo for 2-way spatial split."""
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


def calc_halo_sizes_8way(
    x_shape: torch.Size,
    kernel_size: int,
    stride: int,
) -> tuple[list[int], list[int]]:
    """Calculate halo for 4-way spatial split."""
    # global indices where you want to apply the convolutions:
    global_size = int(x_shape[-1] * 4)
    global_steps = range(0, global_size - kernel_size + 1, stride)
    len_global_size = len(global_steps)
    mid_indices = [
        len_global_size // 4 - 1,
        len_global_size // 2 - 1,
        len_global_size * 3 // 4 - 1,
    ]

    right_to_left = [
        max((global_steps[mid_indices[0]] + kernel_size) - x_shape[-1], 0),
        max((global_steps[mid_indices[1]] + kernel_size) - 2 * x_shape[-1], 0),
        max((global_steps[mid_indices[2]] + kernel_size) - 3 * x_shape[-1], 0),
    ]

    left_to_right = [
        (
            global_steps[mid_indices[0] + 1] - x_shape[-1]
            if global_steps[mid_indices[0] + 1] < x_shape[-1]
            else 0
        ),
        (
            global_steps[mid_indices[1] + 1] - 2 * x_shape[-1]
            if global_steps[mid_indices[1] + 1] < 2 * x_shape[-1]
            else 0
        ),
        (
            global_steps[mid_indices[2] + 1] - 3 * x_shape[-1]
            if global_steps[mid_indices[2] + 1] < 3 * x_shape[-1]
            else 0
        ),
    ]

    left_to_right = [left * -1 for left in left_to_right]

    return right_to_left, left_to_right


def get_left_right_local_ranks(parallelism: int) -> list:
    """Define left and right ranks."""
    if parallelism == 2:
        return [[0], [1]]
    elif parallelism == 4:
        return [[0, 1], [2, 3]]
    elif parallelism == 8:
        return [[0, 1], [2, 3], [4, 5], [6, 7]]
    else:
        raise ValueError(f"Unsupported parallelism: {parallelism}")


def communicate_halos(
    x: torch.Tensor,
    right_to_left: int,
    left_to_right: int,
    empty: torch.Tensor,
    spatial_group: dist.ProcessGroup,
    async_op: bool,
    comm_fn: Callable,
) -> tuple[torch.Tensor, list[Optional[dist.Work]]]:
    """Communicate halos over 2way spatial split."""
    left_right_local_ranks = get_left_right_local_ranks(spatial_group.size())
    local_spatial_rank = spatial_group.rank()
    if local_spatial_rank in left_right_local_ranks[0]:
        x_send = x[..., -left_to_right:] if left_to_right > 0 else empty
        x_recv = (
            get_preallocated_buffer(
                torch.Size((*x.shape[:-1], right_to_left)),
                x.dtype,
                local_spatial_rank,
                x.device,
            )
            if right_to_left > 0
            else empty
        )
        x_send = x_send.contiguous()
        buffer, handles = comm_fn(
            x_send, x_recv, empty, group=spatial_group, async_op=async_op
        )
    elif local_spatial_rank in left_right_local_ranks[1]:
        x_send = x[..., :right_to_left] if right_to_left > 0 else empty
        x_recv = (
            get_preallocated_buffer(
                torch.Size((*x.shape[:-1], left_to_right)),
                x.dtype,
                local_spatial_rank,
                x.device,
            )
            if left_to_right > 0
            else empty
        )
        x_send = x_send.contiguous()
        buffer, handles = comm_fn(
            x_send, x_recv, empty, group=spatial_group, async_op=async_op
        )
    else:
        raise ValueError(f"Unsupported parallelism: {spatial_group.size()}")
    return buffer, handles


def communicate_halos_4space(
    x: torch.Tensor,
    right_to_left: list[int],
    left_to_right: list[int],
    empty: torch.Tensor,
    spatial_group: dist.ProcessGroup,
    async_op: bool,
) -> tuple[list[torch.Tensor], Optional[dist.Work]]:
    """Halo calculation for 8-way split."""
    local_rank = spatial_group.rank()

    match spatial_group.rank():
        case 0:
            send_1 = x[..., -left_to_right[0] :] if left_to_right[0] > 0 else empty
            recv_1 = (
                get_preallocated_buffer(
                    torch.Size((*x.shape[:-1], right_to_left[0])),
                    x.dtype,
                    local_rank,
                    x.device,
                )
                if right_to_left[0] > 0
                else empty
            )
            send_1 = send_1.contiguous()
            if async_op:
                wait_handle = dist.all_to_all(
                    [empty, recv_1, empty, empty],
                    [empty, send_1, empty, empty],
                    group=spatial_group,
                    async_op=async_op,
                )
                return [empty, recv_1], wait_handle
            else:
                dist.all_to_all(
                    [empty, recv_1, empty, empty],
                    [empty, send_1, empty, empty],
                    group=spatial_group,
                    async_op=async_op,
                )
                return [empty, recv_1], None
        case 1:
            send_0 = x[..., : right_to_left[0]] if right_to_left[0] > 0 else empty
            send_2 = x[..., -left_to_right[1] :] if left_to_right[1] > 0 else empty
            recv_0 = (
                get_preallocated_buffer(
                    torch.Size((*x.shape[:-1], left_to_right[0])),
                    x.dtype,
                    local_rank,
                    x.device,
                )
                if left_to_right[0] > 0
                else empty
            )
            recv_2 = (
                get_preallocated_buffer(
                    torch.Size((*x.shape[:-1], right_to_left[2])),
                    x.dtype,
                    local_rank,
                    x.device,
                )
                if left_to_right[2] > 0
                else empty
            )
            send_0, send_2 = send_0.contiguous(), send_2.contiguous()
            if async_op:
                wait_handle = dist.all_to_all(
                    [recv_0, empty, recv_2, empty],
                    [send_0, empty, send_2, empty],
                    group=spatial_group,
                    async_op=async_op,
                )
                return [recv_0, recv_2], wait_handle
            else:
                dist.all_to_all(
                    [recv_0, empty, recv_2, empty],
                    [send_0, empty, send_2, empty],
                    group=spatial_group,
                    async_op=async_op,
                )
                return [recv_0, recv_2], None
        case 2:
            send_1 = x[..., : right_to_left[1]] if right_to_left[1] > 0 else empty
            send_3 = x[..., -left_to_right[2] :] if left_to_right[2] > 0 else empty
            recv_1 = (
                get_preallocated_buffer(
                    torch.Size((*x.shape[:-1], left_to_right[1])),
                    x.dtype,
                    local_rank,
                    x.device,
                )
                if left_to_right[1] > 0
                else empty
            )
            recv_3 = (
                get_preallocated_buffer(
                    torch.Size((*x.shape[:-1], right_to_left[2])),
                    x.dtype,
                    local_rank,
                    x.device,
                )
                if left_to_right[2] > 0
                else empty
            )
            send_1, send_3 = send_1.contiguous(), send_3.contiguous()
            if async_op:
                wait_handle = dist.all_to_all(
                    [empty, recv_1, empty, recv_3],
                    [empty, send_1, empty, send_3],
                    group=spatial_group,
                    async_op=async_op,
                )
                return [recv_1, recv_3], wait_handle
            else:
                dist.all_to_all(
                    [empty, recv_1, empty, recv_3],
                    [empty, send_1, empty, send_3],
                    group=spatial_group,
                    async_op=async_op,
                )
                return [recv_1, recv_3], None
        case 3:
            send_2 = x[..., -left_to_right[2] :] if right_to_left[2] > 0 else empty
            recv_2 = (
                get_preallocated_buffer(
                    torch.Size((*x.shape[:-1], left_to_right[2])),
                    x.dtype,
                    local_rank,
                    x.device,
                )
                if right_to_left[2] > 0
                else empty
            )
            send_2 = send_2.contiguous()
            if async_op:
                wait_handle = dist.all_to_all(
                    [empty, empty, recv_2, empty],
                    [empty, empty, send_2, empty],
                    group=spatial_group,
                    async_op=async_op,
                )
                return [recv_2, empty], wait_handle
            else:
                dist.all_to_all(
                    [empty, empty, recv_2, empty],
                    [empty, empty, send_2, empty],
                    group=spatial_group,
                    async_op=async_op,
                )
                return [recv_2, empty], None
        case _:
            raise ValueError(f"Unsupported local_rank: {local_rank}")


def append_halos(
    x: torch.Tensor,
    halo: torch.Tensor,
    right_to_left: int,
    left_to_right: int,
    local_rank: int,
    parallelism: int,
    empty: torch.Tensor,
) -> torch.Tensor:
    """Append halos for 2-way spatial splits."""
    left_right_local_ranks = get_left_right_local_ranks(parallelism)

    if local_rank in left_right_local_ranks[0]:
        x_with_halos = torch.cat((x, halo if right_to_left > 0 else empty), dim=-1)
    elif local_rank in left_right_local_ranks[1]:
        x_with_halos = torch.cat((halo if left_to_right > 0 else empty, x), dim=-1)
    return x_with_halos


def append_halos_4space(
    x: torch.Tensor,
    halo: torch.Tensor,
    spatial_group: dist.ProcessGroup,
    empty: torch.Tensor,
) -> torch.Tensor:
    """Append halos for 8-way split."""
    halo_l, halo_r = halo[0], halo[1]
    x_with_halos = torch.cat(
        (
            halo_l if halo_l.nelement() > 0 else empty,
            x,
            halo_r if halo_r.nelement() > 0 else empty,
        ),
        dim=-1,
    )

    return x_with_halos


def distributed_conv2d_over_channels(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    local_channel_rank: int,
    in_channels_global: int,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    empty: torch.Tensor,
    channel_group: dist.ProcessGroup,
    conv_func: Callable,
    comm_fn: Callable,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distributed Conv2d over channels."""
    x_buffer = get_preallocated_buffer(
        x.shape, x.dtype, local_channel_rank, x.device
    )

    x_buffer, x_handles = comm_fn(
        data=x, buffer=x_buffer, empty=empty, group=channel_group, async_op=True
    )
    match local_channel_rank % 2:
        case 0:
            y = conv_func(
                x,
                w[..., : in_channels_global // 2, :, :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            for x_handle in x_handles:
                x_handle.wait()
            if b is not None:
                y = (
                    y
                    + conv_func(
                        x_buffer,  # (sampling, batch, in_channels, x, y)
                        w[..., in_channels_global // 2 :, :, :],
                        b,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    )
                )
            else:
                y = (
                    y
                    + conv_func(
                        x_buffer,  # (sampling, batch, in_channels, x, y)
                        w[..., in_channels_global // 2 :, :, :],
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    )
                )
            return y, x, x_buffer
        case 1:
            y = conv_func(
                x,
                w[..., in_channels_global // 2 :, :, :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            for x_handle in x_handles:
                x_handle.wait()
            if b is not None:
                y = y + conv_func(
                    x_buffer,
                    w[..., : in_channels_global // 2, :, :],
                    b,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            else:
                y = y + conv_func(
                    x_buffer,
                    w[..., : in_channels_global // 2, :, :],
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            return y, x_buffer, x
        case _:
            raise ValueError(f"Invalid local_rank % 2: {local_channel_rank % 2}")


def send_g_x_channels(
    g_x: torch.Tensor,
    buffer: torch.Tensor,
    empty: torch.Tensor,
    in_channels_global: int,
    channel_group: dist.ProcessGroup,
    async_op: bool,
) -> tuple[torch.Tensor, Optional[dist.Work]]:
    """Perform all_reduce of g_x over channel splits."""
    match channel_group.rank():
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
    g_x: torch.Tensor,
    buffer: torch.Tensor,
    empty: torch.Tensor,
    in_channels_global: int,
    channel_group: dist.ProcessGroup,
    async_op: bool,
    comm_fn: Callable,
) -> tuple[torch.Tensor, list[Optional[dist.Work]]]:
    """Communicate g_x across channel-parallel processes."""
    match channel_group.rank():
        case 0:
            send_g_x = g_x[..., in_channels_global // 2 :, :, :]

        case 1:
            send_g_x = g_x[..., : in_channels_global // 2, :, :]

    send_g_x = send_g_x.contiguous()
    buffer, handles = comm_fn(send_g_x, buffer, empty, channel_group, async_op=async_op)
    return buffer, handles


def conv2d_fourway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    in_channels: int,
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    empty: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Distributed 2d convolution."""
    # Setup
    local_channel_rank = channel_group.rank()

    # communicate x1, x0
    right_to_left, left_to_right = calc_halo_sizes(x.shape, kernel_size[1], stride[1])

    # Spatial split
    x_with_halos = config.communicate_halos_forward(
        x,
        right_to_left,
        left_to_right,
        local_channel_rank,
        4,
        empty,
        spatial_group,
        channel_group,
        async_op=False,
    )
    (
        y,
        x_left,
        x_right,
    ) = distributed_conv2d_over_channels(
        x_with_halos,
        w,
        b,
        local_channel_rank,
        in_channels,
        stride,
        padding,
        dilation,
        empty,
        channel_group,
        config.conv2d_fn,
        config.comm_fn,
    )
    return y, x_left, x_right, right_to_left, left_to_right


def conv2d_eightway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    in_channels: int,
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    empty: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Distributed 2d convolution."""
    # Setup

    local_channel_rank = channel_group.rank()
    # communicate x1, x0
    right_to_left, left_to_right = calc_halo_sizes_8way(
        x.shape, kernel_size[1], stride[1]
    )
    # Spatial split
    x_with_halos = config.communicate_halos_forward(
        x,
        right_to_left,
        left_to_right,
        8,
        empty,
        spatial_group,
        channel_group,
        False,
    )
    (
        y,
        x_left,
        x_right,
    ) = distributed_conv2d_over_channels(
        x_with_halos,
        w,
        b,
        local_channel_rank,
        in_channels,
        stride,
        padding,
        dilation,
        empty,
        channel_group,
        config.conv2d_fn,
        config.comm_fn,
    )

    return y, x_left, x_right, right_to_left, left_to_right
