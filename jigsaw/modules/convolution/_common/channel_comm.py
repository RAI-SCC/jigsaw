"""Channel-parallel primitives of the convolution kernels."""

from collections.abc import Callable

import torch
import torch.distributed as dist

from jigsaw._core.comm import CommOp, resolve_comm_op, select_comm_1d

# Valid ``channel_comm`` names besides ``"auto"``, in preference order.
CHANNEL_COMMS: tuple[str, ...] = (
    "all_gather",
    *(op.value for op in CommOp if op is not CommOp.AUTO),
)

# A convolution primitive: ``fn(x, w[, b], stride=, padding=, dilation=) -> y``.
ConvFn = Callable[..., torch.Tensor]

# The send of one ring step: ``fn(data, buffer, direction=, group=, async_op=) ->
# (recv_buffer, wait_handle)``. Satisfied by the ``jigsaw._core.comm.comm_1d_*``
# exchanges.
_HopFn = Callable[..., tuple[torch.Tensor, dist.Work | None]]


def resolve_channel_comm(channel_comm: CommOp | str, device: torch.device) -> str:
    """Resolve the channel communication strategy.

    ``"all_gather"`` is kept; the streaming strategies are resolved as
    :class:`~jigsaw._core.comm.CommOp`, so ``"auto"`` becomes ``"all_to_all"``
    on CUDA and ``"batch_isend_irecv"`` otherwise.

    Parameters
    ----------
    channel_comm : CommOp or str
        Channel communication strategy: ``"all_gather"`` or a
        :class:`~jigsaw._core.comm.CommOp` (``"auto"``, ``"all_to_all"``,
        ``"isend"``, ``"batch_isend_irecv"``).
    device : torch.device
        Device of the module.

    Returns
    -------
    str
        Resolved strategy.

    Raises
    ------
    ValueError
        If ``channel_comm`` is not a valid strategy, or ``"all_to_all"`` is
        requested off CUDA.
    """
    if channel_comm == "all_gather":
        return "all_gather"
    if str(channel_comm).lower() not in {op.value for op in CommOp}:
        raise ValueError(
            f"channel_comm must be 'auto' or one of {list(CHANNEL_COMMS)}, "
            f"got {channel_comm!r}"
        )
    return resolve_comm_op(channel_comm, device).value


def resolve_conv_groups(
    groups: int, in_channels: int, out_channels: int, channel_group_size: int
) -> tuple[bool, int]:
    """Validate ``groups`` against the channel group size ``G``.

    Parameters
    ----------
    groups : int
        Number of convolution groups.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    channel_group_size : int
        Size ``G`` of the channel group.

    Returns
    -------
    channel_local : bool
        Whether ``groups`` is a multiple of ``G``, so that each rank convolves
        its own channels without channel communication.
    groups_local : int
        Number of groups of the local convolution.

    Raises
    ------
    ValueError
        If ``groups`` is invalid or does not divide the channel counts.
    NotImplementedError
        If ``groups > 1`` is not a multiple of ``G``.
    """
    if groups < 1:
        raise ValueError(f"groups must be >= 1, got {groups}")
    if in_channels % groups or out_channels % groups:
        raise ValueError(
            f"in_channels ({in_channels}) and out_channels ({out_channels}) must "
            f"both be divisible by groups ({groups})"
        )
    if groups != 1 and groups % channel_group_size != 0:
        raise NotImplementedError(
            "Distributed grouped convolution needs groups == 1 or groups to be a "
            f"multiple of the channel-parallel group size ({channel_group_size}); "
            f"got groups={groups}."
        )
    channel_local = groups % channel_group_size == 0
    groups_local = groups // channel_group_size if channel_local else 1
    return channel_local, groups_local


def _resolve_channel_hop(channel_comm: str, channel_group: dist.ProcessGroup) -> _HopFn:
    """Return the communication function of a streaming ``channel_comm``.

    Raises
    ------
    RuntimeError
        If ``"all_to_all"`` is requested on a gloo group.
    """
    if channel_comm == "all_to_all" and dist.get_backend(channel_group) == "gloo":
        raise RuntimeError(
            "channel_comm='all_to_all' needs a backend with all-to-all support "
            "(e.g. NCCL); gloo has none. Use channel_comm='batch_isend_irecv' "
            "instead."
        )
    return select_comm_1d(CommOp(channel_comm))


def gather_full_channels(
    x: torch.Tensor, channel_group: dist.ProcessGroup, channel_dim: int
) -> torch.Tensor:
    """All-gather the channel slices into a full-channel tensor.

    The local slice is kept in the autograd graph. A group of size one is the
    identity.

    Parameters
    ----------
    x : torch.Tensor
        Local channel slice.
    channel_group : dist.ProcessGroup
        Process group splitting the channels.
    channel_dim : int
        Dimension holding the channels.

    Returns
    -------
    torch.Tensor
        Full-channel tensor.
    """
    group_size = channel_group.size()
    if group_size == 1:
        return x

    local_channel_rank = channel_group.rank()
    slices = [torch.empty_like(x) for _ in range(group_size)]
    dist.all_gather(slices, x, group=channel_group)
    slices[local_channel_rank] = x
    return torch.cat(slices, dim=channel_dim)


def _in_channel_block(
    w: torch.Tensor, channel_dim: int, block: int, width: int
) -> torch.Tensor:
    """Return input-channel block ``block`` of the weight."""
    return w.narrow(channel_dim, block * width, width).contiguous()
