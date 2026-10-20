"""Kernel selection for the distributed linear layers."""

from typing import Callable

import torch

from jigsaw._core.vmap import VmapConfig

from . import _distributed_matmul as dist_mm
from . import _distributed_matmul_p2p as dist_mm_p2p

#: Kernel module of each device type: NCCL collectives on CUDA, p2p on CPU.
_BACKENDS = {"cuda": dist_mm, "cpu": dist_mm_p2p}
#: Kernel-name suffix of each supported model parallelism.
_WAYS = {2: "twoway", 4: "fourway", 8: "eightway"}
#: Kernel-name prefixes, in ``mm_kernels`` order.
_KERNELS = ("xw_transpose", "xw", "x_transpose_w", "fused_xw_xtw")
#: Single-process kernels (no communication), in ``mm_kernels`` order.
_ONEWAY: tuple[Callable, ...] = (
    lambda x, w, *args, **kwargs: x @ w.mT,
    lambda x, w, *args, **kwargs: x @ w,
    lambda x, w, *args, **kwargs: x.mT @ w,
    dist_mm.fused_xw_xtw_oneway,
)


class LinearConfig(VmapConfig):
    """Selection of the kernels of the distributed linear operation."""

    def __init__(
        self,
        parallelism: int,
        *,
        linear_layer: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """Initialize the configuration.

        Parameters
        ----------
        parallelism : int
            Number of model-parallel processes.
        linear_layer : bool
            Whether the operation belongs to a linear layer.
        device : torch.device
            Device of the module.
        """
        self.linear_layer = linear_layer
        self.mm_kernels = _select_mm_kernels(parallelism, device)


def _select_mm_kernels(parallelism: int, device: torch.device) -> list[Callable]:
    """Select the ``xwt``, ``xw``, ``xtw`` and fused ``xw``/``xtw`` kernels.

    Raises
    ------
    ValueError
        If the device type or the parallelism is unsupported.
    """
    backend = _BACKENDS.get(device.type)
    if backend is None:
        raise ValueError(f"Unsupported device {device.type}")
    if parallelism == 1:
        return list(_ONEWAY)
    if parallelism not in _WAYS:
        raise ValueError(f"Unsupported parallelism: {parallelism}")
    return [getattr(backend, f"{name}_{_WAYS[parallelism]}") for name in _KERNELS]
