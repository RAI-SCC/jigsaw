"""Check that a kernel configuration fits the layer it is passed to."""

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


def check_config(config: Any, expected: dict[str, Any]) -> None:
    """Check that ``config`` was built for a layer with the properties ``expected``.

    A mismatched config selects the wrong kernels; a missing bias, for example,
    leaves the bias without a gradient.

    Parameters
    ----------
    config : ConvNdConfig or ConvTransposeNdConfig
        Config passed to the layer.
    expected : dict[str, Any]
        :func:`layer_key` of the layer.

    Raises
    ------
    ValueError
        If ``config`` was built for a layer with other properties.
    """
    diff = [
        f"{k}={config._layer_key[k]!r} (layer: {v!r})"
        for k, v in expected.items()
        if config._layer_key[k] != v
    ]
    if diff:
        raise ValueError(
            "config was built for a different layer: " + ", ".join(diff) + "."
        )
