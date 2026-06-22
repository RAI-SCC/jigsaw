"""Distributed convolution implementation."""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist


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
            torch.zeros((*x.shape[:-1], right_to_left), device=x.device, dtype=x.dtype)
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
            torch.zeros((*x.shape[:-1], left_to_right), device=x.device, dtype=x.dtype)
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
                torch.zeros(
                    (*x.shape[:-1], right_to_left[0]), device=x.device, dtype=x.dtype
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
                torch.zeros(
                    (*x.shape[:-1], left_to_right[0]), device=x.device, dtype=x.dtype
                )
                if left_to_right[0] > 0
                else empty
            )
            recv_2 = (
                torch.zeros(
                    (*x.shape[:-1], right_to_left[2]), device=x.device, dtype=x.dtype
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
                torch.zeros(
                    (*x.shape[:-1], left_to_right[1]), device=x.device, dtype=x.dtype
                )
                if left_to_right[1] > 0
                else empty
            )
            recv_3 = (
                torch.zeros(
                    (*x.shape[:-1], right_to_left[2]), device=x.device, dtype=x.dtype
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
                torch.zeros(
                    (*x.shape[:-1], left_to_right[2]), device=x.device, dtype=x.dtype
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


def conv_transpose_halocalc_twoway(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    spatial_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, int, int, int]:
    """Halo calculation for 2way spatial split."""
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

    left_right_local_ranks = get_left_right_local_ranks(parallelism=4)
    local_spatial_rank = spatial_group.rank()
    if local_spatial_rank in left_right_local_ranks[0]:
        send = y[..., -overlap_size:] if overlap_size > 0 else empty
        recv = (
            torch.zeros((*y.shape[:-1], overlap_size), device=x.device, dtype=x.dtype)
            if overlap_size > 0
            else empty
        )
        send = send.contiguous()
        dist.all_to_all(
            [empty, recv], [empty, send], group=spatial_group, async_op=False
        )
    elif local_spatial_rank in left_right_local_ranks[1]:
        send = y[..., :overlap_size] if overlap_size > 0 else empty
        recv = (
            torch.zeros((*y.shape[:-1], overlap_size), device=x.device, dtype=x.dtype)
            if overlap_size > 0
            else empty
        )
        send = send.contiguous()

        dist.all_to_all(
            [recv, empty], [send, empty], group=spatial_group, async_op=False
        )

    return recv, overlap_size, local_size_left, local_size_right


def conv_transpose_halocalc_fourway(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    spatial_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Halo calculation for 4way spatial split."""
    global_shape = (
        (x.shape[-1] * 4 - 1) * stride - 2 * padding + dilation * (kernel_size - 1)
    ) + 1
    y_shape = y.shape[-1]

    overlap_size = y_shape * 4 - global_shape
    # overlaps need to be communicated between partner processes

    match overlap_size % 3:  # .............
        case 0:
            overlap = [overlap_size // 3, overlap_size // 3, overlap_size // 3]
        case 1:
            overlap = [overlap_size // 3, overlap_size // 3, overlap_size // 3 + 1]
        case 2:
            overlap = [overlap_size // 3 + 1, overlap_size // 3, overlap_size // 3 + 1]
    match global_shape % 4:
        case 0:
            local_size = [
                global_shape // 4,
                global_shape // 4,
                global_shape // 4,
                global_shape // 4,
            ]
        case 1:
            local_size = [
                global_shape // 4,
                global_shape // 4,
                global_shape // 4,
                global_shape // 4 + 1,
            ]
        case 2:
            local_size = [
                global_shape // 4,
                global_shape // 4 + 1,
                global_shape // 4,
                global_shape // 4 + 1,
            ]
        case 3:
            local_size = [
                global_shape // 4,
                global_shape // 4 + 1,
                global_shape // 4 + 1,
                global_shape // 4 + 1,
            ]

    empty = torch.empty(0, device=x.device, dtype=x.dtype)

    local_rank = spatial_group.rank()

    match local_rank:
        case 0:
            send_1 = y[..., -overlap[0] :] if overlap[0] > 0 else empty
            recv_1 = (
                torch.zeros((*y.shape[:-1], overlap[0]), device=x.device, dtype=x.dtype)
                if overlap[0] > 0
                else empty
            )
            send_1 = send_1.contiguous()
            dist.all_to_all(
                [empty, recv_1, empty, empty],
                [empty, empty, empty, send_1],
                group=spatial_group,
                async_op=False,
            )
            recv_a, recv_b = empty, recv_1
        case 1:
            send_0 = y[..., : overlap[0]] if overlap[0] > 0 else empty
            send_2 = y[..., -overlap[1]] if overlap[1] > 0 else empty
            recv_0 = (
                torch.zeros((*y.shape[:-1], overlap[0]), device=x.device, dtype=x.dtype)
                if overlap[0] > 0
                else empty
            )
            recv_2 = (
                torch.zeros((*y.shape[:-1], overlap[1]), device=x.device, dtype=x.dtype)
                if overlap[1] > 0
                else empty
            )
            send_0, send_2 = send_0.contiguous(), send_2.contiguous()

            dist.all_to_all(
                [recv_0, empty, recv_2, empty],
                [send_0, empty, send_2, empty],
                group=spatial_group,
                async_op=False,
            )
            recv_a, recv_b = recv_0, recv_2
        case 2:
            send_1 = y[..., : overlap[1]] if overlap[1] > 0 else empty
            send_3 = y[..., -overlap[2]] if overlap[2] > 0 else empty
            recv_1 = (
                torch.zeros((*y.shape[:-1], overlap[1]), device=x.device, dtype=x.dtype)
                if overlap[1] > 0
                else empty
            )
            recv_3 = (
                torch.zeros((*y.shape[:-1], overlap[2]), device=x.device, dtype=x.dtype)
                if overlap[2] > 0
                else empty
            )
            send_1, send_3 = send_1.contiguous(), send_3.contiguous()

            dist.all_to_all(
                [empty, recv_1, empty, recv_3],
                [empty, recv_1, empty, recv_3],
                group=spatial_group,
                async_op=False,
            )
            recv_a, recv_b = recv_1, recv_3
        case 3:
            send_2 = y[..., : overlap[2]] if overlap[2] > 0 else empty
            recv_2 = (
                torch.zeros((*y.shape[:-1], overlap[2]), device=x.device, dtype=x.dtype)
                if overlap[2] > 0
                else empty
            )
            send_2 = send_2.contiguous()

            dist.all_to_all(
                [empty, empty, send_2, empty],
                [empty, empty, recv_2, empty],
                group=spatial_group,
                async_op=False,
            )
            recv_a, recv_b = recv_2, empty

    return recv_a, recv_b, overlap, local_size


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
    x_buffer = torch.zeros(x.shape, device=x.device, dtype=x.dtype)

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


def distributed_conv_transpose2d_over_channels(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    in_channels: int,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    channel_group: dist.ProcessGroup,
    conv_transpose2d_fn: Callable,
    comm_fn: Callable,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distributed 2d convolution across channels."""
    # x shape
    # communicate x1, x0
    empty = torch.empty(0, device=x.device, dtype=x.dtype)
    x_buffer = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
    x = x.contiguous()
    x_buffer, handles = comm_fn(x, x_buffer, empty, channel_group, async_op=True)

    match channel_group.rank() % 2:
        case 0:
            y = conv_transpose2d_fn(
                x,
                w[..., : in_channels // 2, :, :, :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            for x_handle in handles:
                x_handle.wait()
            if b is not None:
                y = y + conv_transpose2d_fn(
                    x_buffer,
                    w[..., in_channels // 2 :, :, :, :],
                    b,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            else:
                y = y + conv_transpose2d_fn(
                    x_buffer,
                    w[..., in_channels // 2 :, :, :, :],
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            return y, x, x_buffer
        case 1:
            y = conv_transpose2d_fn(
                x,
                w[..., in_channels // 2 :, :, :, :],
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            for x_handle in handles:
                x_handle.wait()
            if b is not None:
                y = y + conv_transpose2d_fn(
                    x_buffer,
                    w[..., : in_channels // 2, :, :, :],
                    b,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            else:
                y = y + conv_transpose2d_fn(
                    x_buffer,
                    w[..., : in_channels // 2, :, :, :],
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            return y, x_buffer, x

    raise ValueError(f"Unsupported local channel rank: {channel_group.rank()}")


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
    selector: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Distributed 2d convolution."""
    # Setup
    local_channel_rank = channel_group.rank()

    # communicate x1, x0
    right_to_left, left_to_right = calc_halo_sizes(x.shape, kernel_size[1], stride[1])

    # Spatial split
    x_with_halos = selector.communicate_halos_forward(
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
        selector.conv2d_fn,
        selector.comm_fn,
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
    selector: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Distributed 2d convolution."""
    # Setup

    local_channel_rank = channel_group.rank()
    # communicate x1, x0
    right_to_left, left_to_right = calc_halo_sizes_8way(
        x.shape, kernel_size[1], stride[1]
    )
    # Spatial split
    x_with_halos = selector.communicate_halos_forward(
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
        selector.conv2d_fn,
        selector.comm_fn,
    )

    return y, x_left, x_right, right_to_left, left_to_right


def conv_transpose2d_twoway(
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
    selector: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distributed 2d convolution."""
    # Setup

    y, x0, x1 = distributed_conv_transpose2d_over_channels(
        x,
        w,
        b,
        in_channels,
        stride=stride,
        padding=padding,
        dilation=dilation,
        channel_group=channel_group,
        conv_transpose2d_fn=selector.conv_transpose_2d_fn,
        comm_fn=selector.comm_fn,
    )

    # communicate x1, x0
    y, halo_sizes = selector.communicate_halos_forward(
        x,
        y,
        kernel_size[1],
        stride=stride[1],
        padding=padding[1],
        dilation=dilation[1],
        spatial_group=spatial_group,
        parallelism=4,
    )

    x_aug = torch.cat([x0, x1], dim=-3)

    return y, x_aug, x, halo_sizes


def conv_transpose2d_fourway(
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
    selector: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distributed 2d convolution."""
    # Setup
    y, x0, x1 = distributed_conv_transpose2d_over_channels(
        x,
        w,
        b,
        in_channels,
        stride=stride,
        padding=padding,
        dilation=dilation,
        channel_group=channel_group,
        conv_transpose2d_fn=selector.conv_transpose2d_fn,
        comm_fn=selector.comm_fn,
    )

    y, halo_sizes = selector.communicate_halos_forward(
        x,
        y,
        kernel_size[1],
        stride=stride[1],
        padding=padding[1],
        dilation=dilation[1],
        spatial_group=spatial_group,
        parallelism=8,
    )

    x_aug = torch.cat([x0, x1], dim=-3)

    return y, x_aug, x, halo_sizes


def conv_transpose2d_fourway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    halo_sizes: torch.Tensor,
    in_channels_global: int,
    stride: int | tuple[int, ...],
    padding: int | tuple[int, ...],
    dilation: int | tuple[int, ...],
    empty: torch.Tensor,
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    selector: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for conv_transpose2d distribtuted 4ways."""
    right_to_left, left_to_right = halo_sizes[0], halo_sizes[1]
    grad_output = selector.communicate_halos_backward(
        grad_output,
        right_to_left,
        left_to_right,
        4,
        empty,
        spatial_group=spatial_group,
        async_op=False,
    )

    # handle autocasting by matching dtype of grad_output to weight dtype
    g_x, g_w, _ = selector.conv_transpose2d_grad_fn(
        grad_output,
        x_aug,
        weight,
        None,
        stride,
        padding,
        dilation,
        True,
        [0],
        1,
        [True, True, False],
    )

    g_x_buffer = torch.zeros(
        (*g_x.shape[:-3], g_x.shape[-3] // 2, *g_x.shape[-2:]),
        device=g_x.device,
        dtype=g_x.dtype,
    )

    g_x_recv, g_x_handles = send_g_x_channels_2d(
        g_x,
        g_x_buffer,
        empty,
        in_channels_global,
        channel_group,
        async_op=True,
        comm_fn=selector.comm_fn,
    )

    for g_x_handle in g_x_handles:
        if g_x_handle is not None:
            g_x_handle.wait()

    g_w_handle = dist.all_reduce(
        g_w, op=dist.ReduceOp.SUM, group=spatial_group, async_op=True
    )

    g_x = (
        g_x[..., : in_channels_global // 2, :, :] + g_x_recv
        if channel_group.rank() == 0  # TODO: left/right definitions [0, 2]
        else g_x[..., in_channels_global // 2 :, :, :] + g_x_recv
    )
    g_w_handle.wait()

    return g_x, g_w


def conv_transpose2d_eightway_backward(
    grad_output: torch.Tensor,
    x_aug: torch.Tensor,
    weight: torch.Tensor,
    halo_sizes: torch.Tensor,
    in_channels_global: int,
    stride: int | tuple[int, ...],
    padding: int | tuple[int, ...],
    dilation: int | tuple[int, ...],
    empty: torch.Tensor,
    spatial_group: dist.ProcessGroup,
    channel_group: dist.ProcessGroup,
    selector: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for conv_transpose2d distributed 8ways."""
    right_to_left, left_to_right = halo_sizes[0], halo_sizes[1]
    grad_output = selector.communicate_halos_backward(
        grad_output,
        right_to_left,
        left_to_right,
        8,
        empty,
        spatial_group=spatial_group,
        async_op=False,
    )

    g_x, g_w, _ = selector.conv_transpose2d_grad_fn(
        grad_output.to(weight.dtype),
        x_aug,
        weight,
        None,
        stride,
        padding,
        dilation,
        True,
        [0],
        1,
        [True, True, False],
    )

    g_x_buffer = torch.zeros(
        (*g_x.shape[:-3], g_x.shape[-3] // 2, *g_x.shape[-2:]),
        device=g_x.device,
        dtype=g_x.dtype,
    )

    g_x_recv, g_x_handles = send_g_x_channels_2d(
        g_x,
        g_x_buffer,
        empty,
        in_channels_global,
        channel_group,
        async_op=True,
        comm_fn=selector.comm_fn,
    )

    # make grads identical on all ranks
    g_w_handle = dist.all_reduce(
        g_w, op=dist.ReduceOp.SUM, group=spatial_group, async_op=True
    )

    for g_x_handle in g_x_handles:
        if g_x_handle is not None:
            g_x_handle.wait()

    g_x = (
        g_x[..., : in_channels_global // 2, :, :] + g_x_recv
        if channel_group.rank() % 2 == 0  # TODO: left/right definitions
        else g_x[..., in_channels_global // 2 :, :, :] + g_x_recv
    )

    g_w_handle.wait()
    return g_x, g_w
