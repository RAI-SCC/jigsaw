"""Kernel selection for the distributed linear layers."""

from typing import Callable, Optional, Union

import torch
from torch.nn.parameter import Parameter

from jigsaw._core.vmap import VmapConfig

from . import _distributed_matmul as dist_mm
from . import _distributed_matmul_p2p as dist_mm_p2p


class LinearConfig(VmapConfig):
    """Selection of the kernels of the distributed linear operation."""

    def __init__(
        self,
        parallelism: int,
        bias: torch.Tensor | None,
        linear_layer: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """Initialize the configuration.

        Parameters
        ----------
        parallelism : int
            Number of model-parallel processes.
        bias : torch.Tensor | None
            Bias, if any.
        linear_layer : bool
            Whether the operation belongs to a linear layer.
        device : torch.device
            Device of the module.
        """
        self.linear_layer = linear_layer
        self.mm_kernels = self._mm_kernel_selection(parallelism, device)
        self.grad_bias = self._grad_bias_selection(bias)
        self.add_bias: Callable = (
            self._return_identity if bias is None else self._add_bias
        )
        self.reshape_weights: Callable = self._no_reshape_weights
        self.grad_sum = self._sum_grad_weights_selection(linear_layer)

    def _select_vmapped(self) -> None:
        """Swap in the members-axis kernels on the vmapped copy.

        The batching rule of the linear Functions passes ``x`` as
        ``(members, batch, seq, in)``, the weight as ``(members, out, in)`` and
        the bias as ``(members, out)``. The weight and bias get singleton
        ``batch``/``seq`` axes, so the kernels broadcast each member's weight
        over its input; the gradients are reduced over ``batch``/``seq`` only.
        """
        if self.linear_layer:
            self.reshape_weights = self._reshape_weights
            if self.add_bias == self._add_bias:
                self.add_bias = self._add_bias_vmapped

    # reshape weights
    def _reshape_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Add a batch dimension to the weights for vmap."""
        w = w.unsqueeze(1)
        return w

    def _no_reshape_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Return the weights unchanged."""
        return w

    def _sum_grad_weights_selection(self, linear_layer: bool) -> Callable:
        """Select the function reducing the weight gradient over the batch dimension."""
        if linear_layer:
            return self._grad_sum
        else:
            return self._no_grad_sum

    def _no_grad_sum(self, grad_weight: torch.Tensor) -> torch.Tensor:
        """Return the weight gradient unchanged."""
        return grad_weight

    def _grad_sum(self, grad_weight: torch.Tensor) -> torch.Tensor:
        """Sum the weight gradient over the batch dimensions.

        The leading (members) axis of a vmapped weight is kept.
        """
        first = 1 if self.vmapped else 0
        dims = tuple(range(first, grad_weight.ndim - 2))
        return grad_weight.sum(dim=dims) if dims else grad_weight

    # Adding xwt + b
    def _add_bias_vmapped(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add the per-member bias to ``x`` of shape ``(members, batch, seq, out)``."""
        b = b.unsqueeze(1).unsqueeze(1)
        return x.add(b)

    def _add_bias(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add the bias to ``x``."""
        return x.add(b)

    def _return_identity(
        self, x: torch.Tensor, b: Optional[Parameter] = None
    ) -> torch.Tensor:
        """Return ``x`` unchanged."""
        return x

    # Reduce bias terms over spatial groups
    def _grad_bias_selection(self, bias: Union[torch.Tensor, None]) -> Callable:
        """Select the function computing the bias gradient."""
        if bias is None:
            return self._no_bias
        else:
            return self._bias_sum

    def _bias_sum(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Sum ``grad_output`` over the space and batch dimensions.

        The leading (members) axis of a vmapped bias is kept.
        """
        first = 1 if self.vmapped else 0
        dims = tuple(range(first, grad_output.ndim - 1))
        return grad_output.sum(dim=dims) if dims else grad_output

    def _no_bias(
        self,
        grad_output: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Return ``None`` since there is no bias."""
        return None

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
