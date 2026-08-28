"""Distributed convolution implementation.

Only the no-halo case (``stride == kernel_size``) is supported: every spatial
rank convolves its own domain slice independently, so the spatial split needs no
communication. The channel-parallel group cooperates through
:func:`distributed_conv2d_over_channels_nway`, which works for any group size.
"""

from typing import Any, Callable

import torch
import torch.distributed as dist


def distributed_conv2d_over_channels_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    channel_group: dist.ProcessGroup,
    conv_func: Callable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-parallel Conv2d for an arbitrary channel-group size.

    Each rank owns a contiguous ``in_channels // channel_group.size()`` slice of
    the input channels and the matching slice of the weight's *output* channels
    (``w`` still carries every input channel). The ranks all-gather their input
    slices, rebuild the full-channel input and run the local convolution, so the
    result already holds this rank's output-channel slice. Group sizes of one
    (all-gather is the identity) and two are handled by the same code path.

    Parameters
    ----------
    x : torch.Tensor
        This rank's input-channel slice, shape ``(..., C_in / G, H, W)``.
    w : torch.Tensor
        This rank's weight slice, shape ``(C_out / G, C_in, kH, kW)``.
    b : torch.Tensor
        This rank's bias slice, or ``None``.
    stride : tuple[int, int]
        Convolution stride.
    padding : str | tuple[int, int]
        Convolution padding.
    dilation : tuple[int, int]
        Convolution dilation.
    channel_group : dist.ProcessGroup
        Channel-parallel process group.
    conv_func : Callable
        Convolution primitive (``F.conv2d`` or a vmapped variant).

    Returns
    -------
    y : torch.Tensor
        This rank's output-channel slice.
    x_full : torch.Tensor
        The gathered full-channel input, needed for the backward pass.
    """
    group_size = channel_group.size()
    local_channel_rank = channel_group.rank()

    x = x.contiguous()
    slices = [torch.empty_like(x) for _ in range(group_size)]
    dist.all_gather(slices, x, group=channel_group)
    # Keep this rank's own slice wired into the autograd graph.
    slices[local_channel_rank] = x
    x_full = torch.cat(slices, dim=-3)

    if b is not None:
        y = conv_func(x_full, w, b, stride=stride, padding=padding, dilation=dilation)
    else:
        y = conv_func(x_full, w, stride=stride, padding=padding, dilation=dilation)

    return y, x_full


def conv2d_nway(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int],
    padding: str | tuple[int, int],
    dilation: tuple[int, int],
    channel_group: dist.ProcessGroup,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed 2d convolution for any spatial- and channel-group size.

    Requires ``stride == kernel_size``, so the spatial split needs no halo
    exchange: every spatial rank convolves its own domain slice independently
    while the channel group cooperates through
    :func:`distributed_conv2d_over_channels_nway`.

    Returns ``(y, x_full)`` where ``x_full`` is the gathered full-channel input
    kept for the backward pass.
    """
    return distributed_conv2d_over_channels_nway(
        x,
        w,
        b,
        stride,
        padding,
        dilation,
        channel_group,
        config.conv2d_fn,
    )
