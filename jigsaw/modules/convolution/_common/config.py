"""Layer properties a convolution config selects its kernels from."""

from typing import Any

import torch
import torch.distributed as dist


def layer_key(
    channel_group: dist.ProcessGroup,
    bias: torch.Tensor | None,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    groups: int,
) -> dict[str, Any]:
    """Return the layer properties a convolution config selects its kernels from.

    Parameters
    ----------
    channel_group : dist.ProcessGroup
        Channel process group.
    bias : torch.Tensor | None
        Bias, if any.
    kernel_size : tuple[int, ...]
        Kernel size.
    stride : tuple[int, ...]
        Stride.
    groups : int
        Number of convolution groups.

    Returns
    -------
    dict[str, Any]
        Spatial rank, bias presence, whether ``kernel_size == stride``, groups
        and channel group size.
    """
    return {
        "n_spatial": len(kernel_size),
        "bias": bias is not None,
        "kernel_size == stride": tuple(kernel_size) == tuple(stride),
        "groups": groups,
        "channel group size": channel_group.size(),
    }
