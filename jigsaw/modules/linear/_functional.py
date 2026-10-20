"""Autograd functions of the distributed linear layers."""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from jigsaw._core.autocast import cast_to_autocast_dtype
from jigsaw._core.vmap import check_not_vmapped, to_front

from ._config import LinearConfig


class LinearCtx(Protocol):
    """Attributes stored on the autograd context."""

    saved_tensors: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    local_rank: int
    device: torch.Tensor
    config: LinearConfig
    model_group: dist.ProcessGroup
    spatial_group: dist.ProcessGroup
    b: Optional[Parameter]


def _setup_linear_ctx(ctx: torch.autograd.function.FunctionCtx, inputs: Any) -> None:
    """Save the tensors and metadata of :class:`_XWT`, :class:`_XW` and :class:`_XTW`."""
    x, w, b, local_rank, device, config, model_group = inputs
    ctx.save_for_backward(x, w, b)
    c = cast(LinearCtx, ctx)
    c.local_rank = local_rank
    c.device = device
    c.config = config
    c.model_group = model_group
    c.b = b


def _linear_vmap(
    fn: Any,
    info: Any,
    in_dims: tuple[Optional[int], ...],
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    local_rank: int,
    device: torch.device,
    config: LinearConfig,
    model_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, Optional[int]]:
    """Batching rule shared by :class:`_XWT`, :class:`_XW` and :class:`_XTW`.

    Makes one un-vmapped ``fn`` call with the vmapped axis as a leading batch
    axis, so every collective runs once.

    * Matmul of two operands (no layer weight): both get the leading axis.
    * Layer weight and bias not vmapped: the vmapped axis is folded into the
      batch axis.
    * Layer weight or bias vmapped (e.g. stacked ensemble members): ``fn`` runs
      with :meth:`LinearConfig.for_vmap`, which broadcasts each member's weight
      over that member's ``(batch, seq)`` input.
    """
    check_not_vmapped(config)
    size = info.batch_size
    x_dim, w_dim, b_dim = in_dims[:3]

    if not config.linear_layer:
        x = cast(torch.Tensor, to_front(x, x_dim, size)).contiguous()
        w = cast(torch.Tensor, to_front(w, w_dim, size)).contiguous()
        return fn(x, w, b, local_rank, device, config, model_group), 0

    x = cast(torch.Tensor, to_front(x, x_dim, size))
    if w_dim is None and b_dim is None:
        if x.ndim == 2:
            return fn(x.contiguous(), w, b, local_rank, device, config, model_group), 0
        out = fn(
            x.flatten(0, 1).contiguous(), w, b, local_rank, device, config, model_group
        )
        return out.unflatten(0, (size, -1)), 0

    w = cast(torch.Tensor, to_front(w, w_dim, size))
    b = to_front(b, b_dim, size)
    # The vmapped config expects ``x`` as ``(members, batch, seq, in)``.
    lead = x.shape[1:-1]
    x = x.reshape(size, -1, lead[-1] if lead else 1, x.shape[-1]).contiguous()
    out = fn(x, w, b, local_rank, device, config.for_vmap(), model_group)
    return out.reshape(size, *lead, out.shape[-1]), 0


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
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None
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
        grad_weight = config.grad_sum(grad_weight)  # sum along batch dimension

        grad_b = config.grad_bias(grad_output)

        return grad_input, grad_weight, grad_b, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        *args: Any,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Batching rule for ``torch.vmap``."""
        return _linear_vmap(_XWT.apply, info, in_dims, *args)


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
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None
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

        grad_weight = config.grad_sum(grad_weight)
        grad_b = config.grad_bias(grad_output)

        return grad_input, grad_weight, grad_b, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        *args: Any,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Batching rule for ``torch.vmap``."""
        return _linear_vmap(_XW.apply, info, in_dims, *args)


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
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None
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

        grad_weight = config.grad_sum(grad_weight)
        grad_b = config.grad_bias(grad_output)

        return grad_input, grad_weight, grad_b, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        *args: Any,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Batching rule for ``torch.vmap``."""
        return _linear_vmap(_XTW.apply, info, in_dims, *args)
