"""Kernel selection for the distributed linear layers."""

from typing import Callable

import torch

from jigsaw._core.vmap import VmapConfig

from . import _distributed_matmul as dist_mm
from . import _distributed_matmul_p2p as dist_mm_p2p


class LinearConfig(VmapConfig):
    """Selection of the kernels of the distributed linear operation."""

    def __init__(
        self,
        parallelism: int,
        bias: torch.Tensor | None,
        *,
        linear_layer: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """Initialize the configuration.

        Parameters
        ----------
        parallelism : int
            Number of model-parallel processes.
        bias : torch.Tensor | None
            Unused; kept for backward compatibility. The Functions add the bias
            and reduce its gradient whenever one is passed.
        linear_layer : bool
            Whether the operation belongs to a linear layer.
        device : torch.device
            Device of the module.
        """
        self.linear_layer = linear_layer
        self.mm_kernels = self._mm_kernel_selection(parallelism, device)

    def _mm_kernel_selection(
        self, parallelism: int, device: torch.device
    ) -> list[Callable]:
        """Select the ``xwt``, ``xw`` and ``xtw`` matmul kernels.

        Returns
        -------
        list[Callable]
            The three kernels.
        """
        xw_t_kernel = self._select_xw_transpose_kernel(parallelism, device)
        xw_kernel = self._select_xw_kernel(parallelism, device)
        x_t_w_kernel = self._select_x_transpose_w_kernel(parallelism, device)
        xw_xtw_fused_kernel = self._select_xw_xtw_fused_kernel(parallelism, device)

        return [xw_t_kernel, xw_kernel, x_t_w_kernel, xw_xtw_fused_kernel]

    def _select_xw_transpose_kernel(
        self, parallelism: int, device: torch.device = torch.device("cuda")
    ) -> Callable:
        """Select the ``xwt`` kernel."""
        if device.type == "cuda":
            match parallelism:
                case 1:
                    return lambda x, w, *args, **kwargs: x @ w.mT
                case 2:
                    return dist_mm.xw_transpose_twoway
                case 4:
                    return dist_mm.xw_transpose_fourway
                case 8:
                    return dist_mm.xw_transpose_eightway
                case _:
                    raise ValueError(f"Unsupported parallelism: {parallelism}")
        elif device.type == "cpu":
            match parallelism:
                case 1:
                    return lambda x, w, *args, **kwargs: x @ w.mT
                case 2:
                    return dist_mm_p2p.xw_transpose_twoway
                case 4:
                    return dist_mm_p2p.xw_transpose_fourway
                case 8:
                    return dist_mm_p2p.xw_transpose_eightway
                case _:
                    raise ValueError(f"Unsupported parallelism: {parallelism}")
        else:
            raise ValueError(f"Unsupported device {device.type}")

    def _select_xw_kernel(
        self, parallelism: int, device: torch.device = torch.device("cuda")
    ) -> Callable:
        """Select the ``xw`` kernel."""
        if device.type == "cuda":
            match parallelism:
                case 1:
                    return lambda x, w, *args, **kwargs: x @ w
                case 2:
                    return dist_mm.xw_twoway
                case 4:
                    return dist_mm.xw_fourway
                case 8:
                    return dist_mm.xw_eightway
                case _:
                    raise ValueError(f"Unsupported parallelism: {parallelism}")
        elif device.type == "cpu":
            match parallelism:
                case 1:
                    return lambda x, w, *args, **kwargs: x @ w
                case 2:
                    return dist_mm_p2p.xw_twoway
                case 4:
                    return dist_mm_p2p.xw_fourway
                case 8:
                    return dist_mm_p2p.xw_eightway
                case _:
                    raise ValueError(f"Unsupported parallelism: {parallelism}")
        else:
            raise ValueError(f"Unsupported device {device.type}")

    def _select_x_transpose_w_kernel(
        self, parallelism: int, device: torch.device = torch.device("cuda")
    ) -> Callable:
        """Select the ``xtw`` kernel."""
        if device.type == "cuda":
            match parallelism:
                case 1:
                    return lambda x, w, *args, **kwargs: x.mT @ w
                case 2:
                    return dist_mm.x_transpose_w_twoway
                case 4:
                    return dist_mm.x_transpose_w_fourway
                case 8:
                    return dist_mm.x_transpose_w_eightway
                case _:
                    raise ValueError(f"Unsupported parallelism: {parallelism}")
        elif device.type == "cpu":
            match parallelism:
                case 1:
                    return lambda x, w, *args, **kwargs: x.mT @ w
                case 2:
                    return dist_mm_p2p.x_transpose_w_twoway
                case 4:
                    return dist_mm_p2p.x_transpose_w_fourway
                case 8:
                    return dist_mm_p2p.x_transpose_w_eightway
                case _:
                    raise ValueError(f"Unsupported parallelism: {parallelism}")
        else:
            raise ValueError(f"Unsupported device {device.type}")

    def _select_xw_xtw_fused_kernel(
        self, parallelism: int, device: torch.device = torch.device("cuda")
    ) -> Callable:
        """Select the fused ``xw`` and ``xtw`` kernel.

        Parameters
        ----------
        parallelism : int
            Number of model-parallel processes.
        device : torch.device
            Device of the module.

        Returns
        -------
        Callable
            Fused kernel.
        """
        if device.type == "cuda":
            match parallelism:
                case 1:
                    return dist_mm.fused_xw_xtw_oneway
                case 2:
                    return dist_mm.fused_xw_xtw_twoway
                case 4:
                    return dist_mm.fused_xw_xtw_fourway
                case 8:
                    return dist_mm.fused_xw_xtw_eightway
                case _:
                    raise ValueError(f"Unsupported parallelism: {parallelism}")
        elif device.type == "cpu":
            match parallelism:
                case 1:
                    return dist_mm.fused_xw_xtw_oneway
                case 2:
                    return dist_mm_p2p.fused_xw_xtw_twoway
                case 4:
                    return dist_mm_p2p.fused_xw_xtw_fourway
                case 8:
                    return dist_mm_p2p.fused_xw_xtw_eightway
                case _:
                    raise ValueError(f"Unsupported parallelism: {parallelism}")
        else:
            raise ValueError(f"Unsupported device {device.type}")
