"""Helper functions for distributed transpose convolution."""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist


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
    config: Any,
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
        conv_transpose2d_fn=config.conv_transpose_2d_fn,
        comm_fn=config.comm_fn,
    )

    # communicate x1, x0
    y, halo_sizes = config.communicate_halos_forward(
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
    config: Any,
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
        conv_transpose2d_fn=config.conv_transpose_2d_fn,
        comm_fn=config.comm_fn,
    )

    y, halo_sizes = config.communicate_halos_forward(
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
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for conv_transpose2d distribtuted 4ways."""
    right_to_left, left_to_right = halo_sizes[0], halo_sizes[1]
    grad_output = config.communicate_halos_backward(
        grad_output,
        right_to_left,
        left_to_right,
        4,
        empty,
        spatial_group=spatial_group,
        async_op=False,
    )

    # handle autocasting by matching dtype of grad_output to weight dtype
    g_x, g_w, _ = config.conv_transpose2d_grad_fn(
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
        comm_fn=config.comm_fn,
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
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for conv_transpose2d distributed 8ways."""
    right_to_left, left_to_right = halo_sizes[0], halo_sizes[1]
    grad_output = config.communicate_halos_backward(
        grad_output,
        right_to_left,
        left_to_right,
        8,
        empty,
        spatial_group=spatial_group,
        async_op=False,
    )

    g_x, g_w, _ = config.conv_transpose2d_grad_fn(
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
        comm_fn=config.comm_fn,
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
