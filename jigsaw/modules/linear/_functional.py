"""Autograd functions of the distributed linear layers."""

from typing import Any, Optional, Protocol, Union, cast

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from jigsaw._core.autocast import cast_to_autocast_dtype

from ._config import LinearConfig


class LinearCtx(Protocol):
    """Attributes stored on the autograd context."""

    saved_tensors: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    local_rank: int
    device: torch.Tensor
    config: LinearConfig
    model_group: dist.ProcessGroup
    spatial_group: dist.ProcessGroup
    bayesian: bool
    b: Optional[Parameter]


def _setup_linear_ctx(ctx: torch.autograd.function.FunctionCtx, inputs: Any) -> None:
    """Save the tensors and metadata of :class:`_XWT`, :class:`_XW` and :class:`_XTW`."""
    x, w, b, local_rank, device, config, model_group, bayesian = inputs
    ctx.save_for_backward(x, w, b)
    c = cast(LinearCtx, ctx)
    c.local_rank = local_rank
    c.device = device
    c.config = config
    c.model_group = model_group
    c.b = b
    c.bayesian = bayesian


def _linear_vmap(
    fn: Any,
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    local_rank: int,
    device: torch.device,
    config: LinearConfig,
    model_group: dist.ProcessGroup,
    bayesian: bool,
) -> tuple[torch.Tensor, Optional[int]]:
    """Apply ``fn`` under ``torch.vmap`` with the vmapped dimension at position 0."""
    x = x.contiguous()
    out = fn(x, w, b, local_rank, device, config, model_group, bayesian)
    return out, 0


class _XWT(torch.autograd.Function):
    """Distributed batched matmul ``x @ w.mT``."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        local_rank: int,
        device: torch.device,
        config: LinearConfig,
        model_group: dist.ProcessGroup,
        bayesian: bool = False,
    ) -> torch.Tensor:
        """Compute the local output of ``x @ w.mT``.

        Parameters
        ----------
        x : torch.Tensor
            Local input.
        w : torch.Tensor
            Local weight.
        b : torch.Tensor | None
            Local bias.
        local_rank : int
            Rank within the model group.
        device : torch.device
            Device of the computation.
        config : LinearConfig
            Kernel selection.
        model_group : dist.ProcessGroup
            Process group of one model-parallel instance.
        bayesian : bool
            Whether the model is Bayesian.

        Returns
        -------
        torch.Tensor
            Local output.
        """
        w = config.reshape_weights(w)
        # Opt into autocast: this Function hides the inner matmul from autocast's
        # op-level policy. ``_XWT`` casts x/w here (its kernel does not); casting
        # the bias too keeps ``add_bias`` from promoting the output back to fp32.
        x, w, b = cast_to_autocast_dtype(x, w, b)
        xwt = config.mm_kernels[0](x, w, local_rank, device, model_group)
        xwt = config.add_bias(xwt, b)

        xwt = cast(torch.Tensor, xwt)

        return xwt

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx, inputs: Any, output: torch.Tensor
    ) -> None:
        """Save the tensors and metadata for the backward pass."""
        _setup_linear_ctx(ctx, inputs)

    @staticmethod
    def backward(
        ctx: LinearCtx, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        """Compute the gradients of the linear operation.

        Parameters
        ----------
        ctx : LinearCtx
            Autograd context.
        grad_output : torch.Tensor
            Gradient of the output.

        Returns
        -------
        tuple[torch.Tensor | None, ...]
            Gradients of the input, weight and bias, followed by ``None`` for
            the non-differentiable arguments.
        """
        input, weight, _ = ctx.saved_tensors
        local_rank = ctx.local_rank
        device = ctx.device
        config = ctx.config
        model_group = ctx.model_group

        weight = config.reshape_weights(weight)
        # handle conversion if necessary here
        grad_output = grad_output.to(weight.dtype)
        input = input.to(weight.dtype)
        grad_input, grad_weight = config.mm_kernels[3](
            grad_output,
            weight,
            input,
            local_rank=local_rank,
            device=device,
            model_group=model_group,
        )
        grad_weight = config.grad_sum(
            grad_weight, ctx.bayesian
        )  # sum along batch dimension

        grad_b = config.grad_bias(grad_output)

        return grad_input, grad_weight, grad_b, None, None, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Union[torch.Tensor, None],
        local_rank: int,
        device: torch.device,
        config: LinearConfig,
        model_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Batching rule for ``torch.vmap``."""
        return _linear_vmap(
            _XWT.apply, x, w, b, local_rank, device, config, model_group, bayesian
        )


class _XW(torch.autograd.Function):
    """Distributed batched matmul ``x @ w``."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        local_rank: int,
        device: torch.device,
        config: LinearConfig,
        model_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> torch.Tensor:
        """Compute the local output of ``x @ w``.

        Parameters
        ----------
        x : torch.Tensor
            Local input.
        w : torch.Tensor
            Local weight.
        b : torch.Tensor | None
            Local bias.
        local_rank : int
            Rank within the model group.
        device : torch.device
            Device of the computation.
        config : LinearConfig
            Kernel selection.
        model_group : dist.ProcessGroup
            Process group of one model-parallel instance.
        bayesian : bool
            Whether the model is Bayesian.

        Returns
        -------
        torch.Tensor
            Local output.
        """
        w = config.reshape_weights(w)
        # Cast the bias to the autocast dtype (the kernel casts x/w itself) so
        # ``add_bias`` does not promote the output back to fp32 under autocast.
        (b,) = cast_to_autocast_dtype(b)
        xw = config.mm_kernels[1](x, w, local_rank, device, model_group)
        xw = config.add_bias(xw, b)
        xw = cast(torch.Tensor, xw)
        return xw

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx, inputs: Any, output: torch.Tensor
    ) -> None:
        """Save the tensors and metadata for the backward pass."""
        _setup_linear_ctx(ctx, inputs)

    @staticmethod
    def backward(
        ctx: LinearCtx, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None, None
    ]:
        """Compute the gradients of the linear operation.

        Parameters
        ----------
        ctx : LinearCtx
            Autograd context.
        grad_output : torch.Tensor
            Gradient of the output.

        Returns
        -------
        tuple[torch.Tensor | None, ...]
            Gradients of the input, weight and bias, followed by ``None`` for
            the non-differentiable arguments.
        """
        input, weight, bias = ctx.saved_tensors
        local_rank = ctx.local_rank
        device = ctx.device
        config = ctx.config
        model_group = ctx.model_group

        weight = config.reshape_weights(weight)

        grad_input = config.mm_kernels[0](
            grad_output,
            weight,
            local_rank=local_rank,
            device=device,
            model_group=model_group,
        )
        grad_weight = config.mm_kernels[2](
            input,
            grad_output,
            local_rank=local_rank,
            device=device,
            model_group=model_group,
        )

        grad_weight = config.grad_sum(grad_weight, ctx.bayesian)
        grad_b = config.grad_bias(grad_output)

        return grad_input, grad_weight, grad_b, None, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        local_rank: int,
        device: torch.device,
        config: LinearConfig,
        model_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Batching rule for ``torch.vmap``."""
        return _linear_vmap(
            _XW.apply, x, w, b, local_rank, device, config, model_group, bayesian
        )


class _XTW(torch.autograd.Function):
    """Distributed batched matmul ``x.mT @ w``."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        local_rank: int,
        device: torch.device,
        config: LinearConfig,
        model_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> torch.Tensor:
        """Compute the local output of ``x.mT @ w``.

        Parameters
        ----------
        x : torch.Tensor
            Local input.
        w : torch.Tensor
            Local weight.
        b : torch.Tensor | None
            Local bias.
        local_rank : int
            Rank within the model group.
        device : torch.device
            Device of the computation.
        config : LinearConfig
            Kernel selection.
        model_group : dist.ProcessGroup
            Process group of one model-parallel instance.
        bayesian : bool
            Whether the model is Bayesian.

        Returns
        -------
        torch.Tensor
            Local output.
        """
        w = config.reshape_weights(w)
        # Cast the bias to the autocast dtype (the kernel casts x/w itself) so
        # ``add_bias`` does not promote the output back to fp32 under autocast.
        (b,) = cast_to_autocast_dtype(b)
        xtw = config.mm_kernels[2](x, w, local_rank, device, model_group)
        xtw = config.add_bias(xtw, b)
        xtw = cast(torch.Tensor, xtw)
        return xtw

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx, inputs: Any, output: torch.Tensor
    ) -> None:
        """Save the tensors and metadata for the backward pass."""
        _setup_linear_ctx(ctx, inputs)

    @staticmethod
    def backward(
        ctx: LinearCtx, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None, None
    ]:
        """Compute the gradients of the linear operation.

        Parameters
        ----------
        ctx : LinearCtx
            Autograd context.
        grad_output : torch.Tensor
            Gradient of the output.

        Returns
        -------
        tuple[torch.Tensor | None, ...]
            Gradients of the input, weight and bias, followed by ``None`` for
            the non-differentiable arguments.
        """
        input, weight, _ = ctx.saved_tensors
        local_rank = ctx.local_rank
        device = ctx.device
        config = ctx.config
        model_group = ctx.model_group

        weight = config.reshape_weights(weight)

        grad_input = config.mm_kernels[0](
            weight,
            grad_output,
            local_rank=local_rank,
            device=device,
            model_group=model_group,
        )
        grad_weight = config.mm_kernels[1](
            input,
            grad_output,
            local_rank=local_rank,
            device=device,
            model_group=model_group,
        )

        grad_weight = config.grad_sum(grad_weight, ctx.bayesian)
        grad_b = config.grad_bias(grad_output)

        return grad_input, grad_weight, grad_b, None, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        local_rank: int,
        device: torch.device,
        config: LinearConfig,
        model_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Batching rule for ``torch.vmap``."""
        return _linear_vmap(
            _XTW.apply, x, w, b, local_rank, device, config, model_group, bayesian
        )
