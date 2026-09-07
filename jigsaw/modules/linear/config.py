"""Config definition for linear layers."""

from typing import Callable, Optional, Union

import torch
from torch.nn.parameter import Parameter

from . import distributed_matmul as dist_mm
from . import distributed_matmul_p2p as dist_mm_p2p


class LinearConfig:
    """Selection strategy for distributed linear operation."""

    def __init__(
        self,
        parallelism: int,
        bias: torch.Tensor | None,
        linear_layer: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """Initialize the object.

        Parameters
        ----------
        parallelism : int
            Number of processes participating in parallel execution.
        bias : torch.Tensor | None
            Optional bias parameter.
        linear_layer : bool
            Whether the operation belongs to a linear layer.
        device : torch.device
            Device on which tensors and communication are placed.

        """
        self.mm_kernels = self._mm_kernel_selection(parallelism, device)
        self.grad_bias = self._grad_bias_selection(parallelism, bias)
        self.add_bias = self._add_bias_selection(bias)
        self.grad_sum = self._sum_grad_weights_selection(linear_layer)

    def _sum_grad_weights_selection(self, linear_layer: bool) -> Callable:
        """Determine if summation over batch dimension is necessary."""
        if linear_layer:
            return self._grad_sum
        else:
            return self._no_grad_sum

    def _no_grad_sum(
        self, grad_weight: torch.Tensor, members: bool = False
    ) -> torch.Tensor:
        """Identity function to return gradient weights."""
        return grad_weight

    def _grad_sum(
        self, grad_weight: torch.Tensor, members: bool = False
    ) -> torch.Tensor:
        """Reduce the per-sample weight grad to the parameter shape.

        Plain call: ``grad_weight`` is ``(B, out, in)`` -> sum the batch axis.
        Under the ``vmap`` rule (:func:`_linear_vmap`) an ensemble/members axis
        rides in front and the weight carries a singleton batch axis
        (``members`` is then true), so ``grad_weight`` is ``(E, B, out, in)`` ->
        reduce only ``B``, keeping the members axis and the singleton to match
        the ``(E, 1, out, in)`` weight.
        """
        if members:
            return grad_weight.sum(dim=1, keepdim=True)
        if len(grad_weight.shape) > 2:
            return grad_weight.sum(dim=0)
        return grad_weight

    # Adding xwt + b
    def _add_bias_selection(self, bias: Union[torch.Tensor, None]) -> Callable:
        """
        Selector function to add bias to weights.

        If the bias is None, return identity. Else, if bias exists, add bias.

        Parameters
        ----------
        bias: torch.Tensor
            Value of the bias term. Not the boolean flag.
        """
        if bias is None:
            return self._return_identity
        else:
            return self._add_bias

    def _add_bias(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add bias to x."""
        return x.add(b)

    def _return_identity(
        self, x: torch.Tensor, b: Optional[Parameter] = None
    ) -> torch.Tensor:
        """Return x as identity since bias does not exist."""
        return x

    # Reduce bias terms over spatial groups
    def _grad_bias_selection(
        self,
        parallelism: int,
        bias: Union[torch.Tensor, None],
    ) -> Callable:
        """
        Selector function for calculating gradient of bias.

        Returns None when bias term is None.
        If parallelism is 2, all model instances hold different and independent bias parameters.
        Otherwise, reduction over the spatial groups is necessary, since spatial groups will "share" output channels.
        If we have a synchronized layer in the case of SWIN, the bias synchronization will also need to occur across spatial groups.
        """
        # return identity if bias is None
        if bias is None:
            return self._no_bias
        # all_reduce is not required for 2-way parallel models
        else:
            return self._bias_sum

    def _bias_sum(self, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Calculate grad wrt bias.

        grad_b is equal to the gradient output summed over space and batch dim.

        Parameters
        ----------
        grad_output: torch.Tensor
            gradient wrt loss
        """
        if len(grad_output.shape) > 2:
            grad_b = grad_output.sum(dim=(-3, -2))
        else:
            grad_b = grad_output.sum(dim=(-2))
        return grad_b

    def _no_bias(
        self,
        grad_output: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Return identity when bias does not exist."""
        return None

    def _mm_kernel_selection(
        self, parallelism: int, device: torch.device
    ) -> list[Callable]:
        """
        Select distriuted matrix multiplication kernels.

        Depends on number of model-parallel processes.


        Returns
        -------
        List[Callable, Callable, Callable]: pointers to xwt, xw, and xtw kernels.
        """
        xw_t_kernel = self._select_xw_transpose_kernel(parallelism, device)
        xw_kernel = self._select_xw_kernel(parallelism, device)
        x_t_w_kernel = self._select_x_transpose_w_kernel(parallelism, device)
        xw_xtw_fused_kernel = self._select_xw_xtw_fused_kernel(parallelism, device)

        return [xw_t_kernel, xw_kernel, x_t_w_kernel, xw_xtw_fused_kernel]

    def _select_xw_transpose_kernel(
        self, parallelism: int, device: torch.device = torch.device("cuda")
    ) -> Callable:
        """Return xw_t kernel based on parallelism and device."""
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
        """Return xw kernel based on parallelism."""
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
        """Return x_tw kernel based on parallelism."""
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
        """Select the fused matrix multiplication kernel.

        Parameters
        ----------
        parallelism : int
            Number of processes participating in parallel execution.
        device : torch.device
            Device on which tensors and communication are placed.

        Returns
        -------
        Callable
            Result returned by the operation.
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
