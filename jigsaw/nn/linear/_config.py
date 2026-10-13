"""Kernel selection for the distributed linear layers."""

from typing import Callable, Optional, Union

import torch
from torch.nn.parameter import Parameter

from . import _distributed_matmul as dist_mm
from . import _distributed_matmul_p2p as dist_mm_p2p


class LinearConfig:
    """Selection of the kernels of the distributed linear operation."""

    def __init__(
        self,
        parallelism: int,
        bias: torch.Tensor | None,
        bayesian: bool,
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
        bayesian : bool
            Whether the model is Bayesian.
        linear_layer : bool
            Whether the operation belongs to a linear layer.
        device : torch.device
            Device of the module.
        """
        self.mm_kernels = self._mm_kernel_selection(parallelism, device)
        self.grad_bias = self._grad_bias_selection(parallelism, bias, bayesian)
        self.add_bias = self._add_bias_selection(bias, bayesian)
        self.reshape_weights = self._reshape_weights_selection(bayesian, linear_layer)
        self.grad_sum = self._sum_grad_weights_selection(linear_layer)

    # reshape weights
    def _reshape_weights_selection(
        self, bayesian: bool, linear_layer: bool
    ) -> Callable:
        """Select the weight-reshape function."""
        # If the model is Bayesian and the module is a linear layer,
        # weight reshape will be necessary.
        if linear_layer and bayesian:
            return self._reshape_weights
        # If it is a distributed matrix operation with no explicit
        # weight sampling, weight reshape is not needed.
        else:
            return self._no_reshape_weights

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

    def _no_grad_sum(
        self, grad_weight: torch.Tensor, batch_dim: torch.Tensor
    ) -> torch.Tensor:
        """Return the weight gradient unchanged."""
        return grad_weight

    def _grad_sum(
        self, grad_weight: torch.Tensor, batch_dim: torch.Tensor
    ) -> torch.Tensor:
        """Sum the weight gradient over the batch dimension."""
        if len(grad_weight.shape) > 2:
            return grad_weight.sum(dim=int(batch_dim))
        else:
            return grad_weight

    # Adding xwt + b
    def _add_bias_selection(
        self, bias: Union[torch.Tensor, None], bayesian: bool
    ) -> Callable:
        """Select the function adding the bias."""
        if bias is None:
            return self._return_identity
        elif bayesian:
            return self._add_bias_bayesian
        else:
            return self._add_bias

    def _add_bias_bayesian(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add the reshaped bias to ``x``."""
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
    def _grad_bias_selection(
        self,
        parallelism: int,
        bias: Union[torch.Tensor, None],
        bayesian: bool,
    ) -> Callable:
        """Select the function computing the bias gradient."""
        # return identity if bias is None
        if bias is None:
            return self._no_bias
        # all_reduce is not required for 2-way parallel models
        else:
            return self._bias_sum

    def _bias_sum(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Sum ``grad_output`` over the space and batch dimensions."""
        if len(grad_output.shape) > 2:
            grad_b = grad_output.sum(dim=(-3, -2))
        else:
            grad_b = grad_output.sum(dim=(-2))
        return grad_b

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
