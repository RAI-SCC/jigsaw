"""Channel-parallel primitives shared by the distributed conv / conv-transpose kernels.

Each rank owns a contiguous ``C_in / G`` slice of the input channels and the
matching slice of the weight's output channels. The strategy that reassembles the
full-channel input -- ``"all_gather"`` (one shot), ``"p2p"`` (batched ``isend`` /
``irecv`` ring) or ``"all_to_all"`` (masked all-to-all ring) -- is selected by
``config.channel_comm``; the streaming drivers live in
:mod:`jigsaw.modules.convolution.conv.channel_comm` /
:mod:`jigsaw.modules.convolution.conv_transpose.channel_comm`. This module holds
the strategy resolution, the per-hop transport lookup, the full-channel gather and
the weight input-channel slicer they share.
"""

from typing import Callable, Optional

import torch
import torch.distributed as dist

from jigsaw.comm import comm_1d_alltoall, comm_1d_p2p

# Valid ``channel_comm`` names, in preference order.
CHANNEL_COMMS: tuple[str, ...] = ("all_gather", "p2p", "all_to_all")

# A convolution primitive: ``fn(x, w[, b], stride=, padding=, dilation=) -> y``.
ConvFn = Callable[..., torch.Tensor]

# A per-hop transport: ``fn(data, buffer, direction=, group=, async_op=) ->
# (recv_buffer, wait_handle)``. Satisfied by ``comm_1d_p2p`` / ``comm_1d_alltoall``.
_HopFn = Callable[..., tuple[torch.Tensor, Optional[dist.Work]]]

# The per-hop transport behind each streaming ``channel_comm``.
_CHANNEL_HOP_FNS: dict[str, _HopFn] = {
    "p2p": comm_1d_p2p,
    "all_to_all": comm_1d_alltoall,
}


def resolve_channel_comm(channel_comm: str, device: torch.device) -> str:
    """Resolve ``"auto"`` from ``device`` and validate the result.

    ``"auto"`` -> ``"all_to_all"`` on CUDA (NCCL has a real all-to-all),
    ``"p2p"`` (batched ``isend`` / ``irecv``) everywhere else.
    """
    if channel_comm == "auto":
        channel_comm = "all_to_all" if device.type == "cuda" else "p2p"
    if channel_comm not in CHANNEL_COMMS:
        raise ValueError(
            f"channel_comm must be 'auto' or one of {list(CHANNEL_COMMS)}, "
            f"got {channel_comm!r}"
        )
    return channel_comm


def resolve_conv_groups(
    groups: int, in_channels: int, out_channels: int, channel_group_size: int
) -> tuple[bool, int]:
    """Validate ``groups`` against the channel-parallel split and resolve the regime.

    A grouped convolution only mixes channels *within* a group, so once the
    channel-parallel group owns whole conv-groups the convolution needs no
    cross-rank channel exchange at all -- every rank runs a plain (grouped) native
    conv on its own channel shard. That holds exactly when ``groups`` is a
    multiple of the channel-parallel group size ``G`` (``G == 1`` included).

    Returns ``(channel_local, groups_local)``:

    * ``channel_local`` -- ``True`` when ``groups % G == 0``: the forward/backward
      run entirely on this rank's channel shard, with ``groups_local`` groups and
      no channel-group communication. ``False`` only for ``groups == 1`` with
      ``G > 1`` -- the historical full-channel-gather path.
    * ``groups_local`` -- the ``groups=`` value the local (sharded) native conv is
      called with: ``groups // G`` in the local regime, ``1`` in the gather one.

    ``groups > 1`` that is not a multiple of ``G`` (so several channel ranks would
    share one conv-group) is not implemented.
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
    """Per-hop transport for a streaming ``channel_comm`` (``"p2p"`` / ``"all_to_all"``).

    ``"all_to_all"`` needs a backend with all-to-all support; gloo has none, so it
    raises there -- pass ``"p2p"`` instead.
    """
    if channel_comm == "all_to_all" and dist.get_backend(channel_group) == "gloo":
        raise RuntimeError(
            "channel_comm='all_to_all' needs a backend with all-to-all support "
            "(e.g. NCCL); gloo has none. Use channel_comm='p2p' instead."
        )
    return _CHANNEL_HOP_FNS[channel_comm]


def gather_full_channels(
    x: torch.Tensor, channel_group: dist.ProcessGroup, channel_dim: int
) -> torch.Tensor:
    """All-gather the channel-parallel input slices into a full-channel tensor.

    Each rank owns a contiguous ``C_in / G`` slice of the input channels; the
    group all-gathers them and concatenates along ``channel_dim``. This rank's
    own slice is spliced back in so it stays wired into the autograd graph.
    Group size one is the identity.
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
    """Contiguous ``width``-wide input-channel block ``block`` of the weight."""
    return w.narrow(channel_dim, block * width, width).contiguous()
