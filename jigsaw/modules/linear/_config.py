"""Kernel selection for the distributed linear layers."""

from collections.abc import Callable

import torch
import torch.distributed as dist

from jigsaw._core.comm import CommOp, resolve_comm_op
from jigsaw._core.vmap import VmapConfig

from . import _distributed_matmul as dist_mm
from . import _distributed_matmul_batchedp2p as dist_mm_batchedp2p
from . import _distributed_matmul_p2p as dist_mm_p2p

#: Kernel module of each resolved transport.
_BACKENDS = {
    CommOp.ALL_TO_ALL: dist_mm,
    CommOp.ISEND: dist_mm_p2p,
    CommOp.BATCH_ISEND_IRECV: dist_mm_batchedp2p,
}
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
    comm : CommOp or str
        Transport of the kernels. ``"auto"`` picks all-to-all on CUDA and
        ``batch_isend_irecv`` on CPU.
    """

    def __init__(
        self,
        model_group: dist.ProcessGroup | None,
        *,
        linear_layer: bool = True,
        device: torch.device = torch.device("cuda"),
        comm: CommOp | str = CommOp.AUTO,
    ):
        """Initialize the configuration."""
        self.linear_layer = linear_layer
        self.comm = resolve_comm_op(comm, device)
        size = 1 if model_group is None else model_group.size()
        self.mm_kernels = _select_mm_kernels(size, self.comm)


def _select_mm_kernels(model_group_size: int, comm: CommOp) -> list[Callable]:
    """Select the ``xwt``, ``xw``, ``xtw`` and fused ``xw``/``xtw`` kernels.

    Raises
    ------
    ValueError
        If the model group size is unsupported.
    """
    backend = _BACKENDS[comm]
    if model_group_size == 1:
        return list(_ONEWAY)
    if model_group_size not in _WAYS:
        raise ValueError(f"Unsupported model group size: {model_group_size}")
    return [getattr(backend, f"{name}_{_WAYS[model_group_size]}") for name in _KERNELS]
