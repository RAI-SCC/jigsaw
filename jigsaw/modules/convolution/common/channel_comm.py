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
) -> tuple[bool, bool, int]:
    """Validate ``groups`` against the channel-parallel split and resolve the regime.

    A grouped convolution only mixes channels *within* a group. Three regimes
    fall out of how ``groups`` relates to the channel-parallel group size ``G``:

    Returns ``(channel_local, channel_grouped, groups_local)``:

    * ``channel_local`` -- ``True`` when ``groups % G == 0``: the channel-parallel
      group owns whole conv-groups, so the forward/backward run entirely on this
      rank's channel shard with ``groups_local = groups // G`` groups and no
      channel-group communication. ``G == 1`` is included here.
    * ``channel_grouped`` -- ``True`` when ``1 < groups`` is *not* a multiple of
      ``G``: a conv-group's channels straddle rank boundaries. Handled by the
      full-channel-gather path running one plain (``groups=1``) conv per
      conv-group the local output shard touches -- see :func:`conv_group_blocks`.
    * neither flag -- ``groups == 1`` with ``G > 1``: the historical
      full-channel-gather path (one fused ``groups=1`` conv).

    ``channel_local`` and ``channel_grouped`` are mutually exclusive.
    ``groups_local`` -- the ``groups=`` value the native conv is called with:
    ``groups // G`` in the local regime, ``1`` otherwise.
    """
    if groups < 1:
        raise ValueError(f"groups must be >= 1, got {groups}")
    if in_channels % groups or out_channels % groups:
        raise ValueError(
            f"in_channels ({in_channels}) and out_channels ({out_channels}) must "
            f"both be divisible by groups ({groups})"
        )
    channel_local = groups % channel_group_size == 0
    channel_grouped = groups != 1 and not channel_local
    groups_local = groups // channel_group_size if channel_local else 1
    if channel_grouped and (
        in_channels % channel_group_size or out_channels % channel_group_size
    ):
        # The channel-grouped per-group loop shards the output channels evenly
        # over the channel group and reduce-scatters the full-channel input grad;
        # both need an exact split.
        raise ValueError(
            f"a grouped convolution whose groups ({groups}) is not a multiple of "
            f"the channel-parallel group size ({channel_group_size}) needs "
            f"in_channels ({in_channels}) and out_channels ({out_channels}) both "
            f"divisible by that group size"
        )
    return channel_local, channel_grouped, groups_local


def conv_group_blocks(
    out_channels: int,
    in_channels: int,
    groups: int,
    channel_rank: int,
    channel_size: int,
) -> list[tuple[int, int, int, int]]:
    """Per conv-group blocks for the channel-grouped (uneven ``groups``) regime.

    The channel-parallel group shards the ``out_channels`` evenly; this rank owns
    the contiguous slice ``[r * C_out / G, (r + 1) * C_out / G)``. A grouped conv
    only connects output channels of conv-group ``g`` to its ``C_in / groups``
    input channels, so this rank's output shard is computed one conv-group at a
    time against the matching slice of the gathered full-channel input.

    Returns one ``(out_lo, out_hi, in_lo, in_hi)`` tuple per conv-group the local
    output shard touches, where ``out_lo:out_hi`` indexes this rank's *local*
    output channels (and the local weight's output axis) and ``in_lo:in_hi``
    indexes the *full* input-channel axis of the gathered input / weight.
    """
    out_local = out_channels // channel_size
    out_per_group = out_channels // groups
    in_per_group = in_channels // groups
    shard_lo = channel_rank * out_local
    shard_hi = shard_lo + out_local

    blocks: list[tuple[int, int, int, int]] = []
    g_lo = shard_lo // out_per_group
    g_hi = (shard_hi - 1) // out_per_group
    for g in range(g_lo, g_hi + 1):
        o_lo = max(g * out_per_group, shard_lo)
        o_hi = min((g + 1) * out_per_group, shard_hi)
        blocks.append(
            (o_lo - shard_lo, o_hi - shard_lo, g * in_per_group, (g + 1) * in_per_group)
        )
    return blocks


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
