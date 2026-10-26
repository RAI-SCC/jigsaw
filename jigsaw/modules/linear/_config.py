"""Kernel selection for the distributed linear layers."""

from collections.abc import Callable

import torch
import torch.distributed as dist

from jigsaw._core.vmap import VmapConfig

from . import _distributed_matmul as dist_mm
from . import _distributed_matmul_batchedp2p as dist_mm_batchedp2p
from . import _distributed_matmul_p2p as dist_mm_p2p

#: Kernel module of each ``comm`` strategy.
_COMMS = {
    "all_to_all": dist_mm,
    "p2p": dist_mm_p2p,
    "batched_p2p": dist_mm_batchedp2p,
}
#: Strategy ``"auto"`` resolves to on each device type: collectives on CUDA, p2p on CPU.
_AUTO_COMMS = {"cuda": "all_to_all", "cpu": "p2p"}
#: Kernel-name suffix of each supported model group size.
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
    """Selection of the kernels of the distributed linear operation.

    Parameters
    ----------
    model_group : dist.ProcessGroup or None
        Process group of one model-parallel instance; its size selects the
        kernels. ``None`` means a single process.
    linear_layer : bool
        Whether the operation belongs to a linear layer.
    device : torch.device
        Device of the module.
    comm : str
        Communication strategy over the model group: ``"all_to_all"`` (needs a
        backend with all-to-all support, such as NCCL), ``"p2p"`` (``isend`` /
        ``irecv``), ``"batched_p2p"`` (one ``batch_isend_irecv`` per phase) or
        ``"auto"``, which resolves to ``"all_to_all"`` on CUDA and ``"p2p"`` on
        CPU.
    """

    def __init__(
        self,
        model_group: dist.ProcessGroup | None,
        *,
        linear_layer: bool = True,
        device: torch.device = torch.device("cuda"),
        comm: str = "auto",
    ):
        """Initialize the configuration."""
        self.linear_layer = linear_layer
        self.comm = _resolve_comm(comm, device)
        size = 1 if model_group is None else model_group.size()
        self.mm_kernels = _select_mm_kernels(size, self.comm)


def _resolve_comm(comm: str, device: torch.device) -> str:
    """Resolve ``"auto"`` to the default strategy of ``device``.

    Raises
    ------
    ValueError
        If the device type is unsupported or ``comm`` is not a valid strategy.
    """
    if device.type not in _AUTO_COMMS:
        raise ValueError(f"Unsupported device {device.type}")
    if comm == "auto":
        return _AUTO_COMMS[device.type]
    if comm not in _COMMS:
        raise ValueError(f"comm must be 'auto' or one of {list(_COMMS)}, got {comm!r}")
    return comm


def _select_mm_kernels(model_group_size: int, comm: str) -> list[Callable]:
    """Select the ``xwt``, ``xw``, ``xtw`` and fused ``xw``/``xtw`` kernels.

    Raises
    ------
    ValueError
        If the model group size is unsupported.
    """
    backend = _COMMS[comm]
    if model_group_size == 1:
        return list(_ONEWAY)
    if model_group_size not in _WAYS:
        raise ValueError(f"Unsupported model group size: {model_group_size}")
    return [getattr(backend, f"{name}_{_WAYS[model_group_size]}") for name in _KERNELS]
