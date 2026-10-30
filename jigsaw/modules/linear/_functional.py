"""Autograd functions of the distributed linear layers."""

from typing import Any, Protocol, cast

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from jigsaw._core import comm_trace
from jigsaw._core.autocast import cast_to_autocast_dtype
from jigsaw._core.vmap import check_not_vmapped, to_front

from ._config import LinearConfig


class LinearCtx(Protocol):
    """Attributes stored on the autograd context."""

    saved_tensors: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
    needs_input_grad: tuple[bool, ...]
    local_rank: int
    device: torch.Tensor
    config: LinearConfig
    model_group: dist.ProcessGroup
    spatial_group: dist.ProcessGroup
    b: Parameter | None
    w_dtype: torch.dtype
    w_shape: torch.Size
    comm_layer: str | None


def _setup_linear_ctx(ctx: torch.autograd.function.FunctionCtx, inputs: Any) -> None:
    """Save the tensors and metadata of :class:`_XWT`, :class:`_XW` and :class:`_XTW`.

    The input gradient needs only ``w`` and the weight gradient only ``x``, so
    an operand is saved only when the other one needs a gradient: a frozen
    weight does not keep its layer's input alive.
    """
    x, w, b, local_rank, device, config, model_group = inputs
    c = cast(LinearCtx, ctx)
    need_x, need_w = c.needs_input_grad[:2]
    # save_for_backward accepts None; torch's stub types only Tensor.
    ctx.save_for_backward(x if need_w else None, w if need_x else None, b)  # type: ignore[arg-type]
    c.local_rank = local_rank
    c.device = device
    c.config = config
    c.model_group = model_group
    c.b = b
    c.w_dtype = w.dtype
    c.w_shape = w.shape
    # Runs inside ``forward``'s trace scope; ``backward`` reuses the layer label.
    c.comm_layer = comm_trace.current_layer()


def _linear_vmap(
    fn: Any,
    info: Any,
    in_dims: tuple[int | None, ...],
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    local_rank: int,
    device: torch.device,
    config: LinearConfig,
    model_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, int | None]:
    """Batching rule shared by :class:`_XWT`, :class:`_XW` and :class:`_XTW`.

    Makes one un-vmapped ``fn`` call with the vmapped axis as a leading batch
    axis, so every collective runs once.

    * Matmul of two operands (no layer weight): both get the leading axis.
    * Layer weight and bias not vmapped: the vmapped axis is folded into the
      batch axis.
    * Layer weight or bias vmapped (e.g. stacked ensemble members): ``x`` runs
      as ``(members, batch, seq, in)``, the weight as ``(members, 1, out, in)``
      and the bias as ``(members, 1, 1, out)``, so each member's parameters
      broadcast over that member's input. The backward sums the gradients back
      to those shapes, and autograd drops the singleton axes.
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

    w = cast(torch.Tensor, to_front(w, w_dim, size)).unsqueeze(1)
    b = to_front(b, b_dim, size)
    if b is not None:
        b = b[:, None, None]
    # The kernels broadcast each member's weight over its ``(batch, seq)`` input.
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
        b: torch.Tensor | None,
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
        # Opt into autocast: this Function hides the inner matmul from autocast's
        # op-level policy. ``_XWT`` casts x/w here (its kernel does not); casting
        # the bias too keeps the bias add from promoting the output back to fp32.
        x, w, b = cast_to_autocast_dtype(x, w, b)
        xwt = config.mm_kernels[0](x, w, local_rank, device, model_group)
        if b is not None:
            xwt = xwt.add(b)

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
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
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
        input, weight, bias = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        kwargs = dict(
            local_rank=ctx.local_rank, device=ctx.device, model_group=ctx.model_group
        )
        mm_kernels = ctx.config.mm_kernels

        # Under autocast, grad_output comes in the autocast dtype; run the
        # backward in the weight's dtype so grad_weight matches the parameter.
        grad_output = grad_output.to(ctx.w_dtype)
        if input is not None:
            input = input.to(ctx.w_dtype)
        # needs_input_grad is the same on every rank of the model group (all
        # ranks freeze the same parameters), so every rank skips the same
        # collectives.
        grad_input = grad_weight = grad_b = None
        with comm_trace.scope(ctx.comm_layer, "backward"):
            if need_x and need_w:
                grad_input, grad_weight = mm_kernels[3](
                    grad_output, weight, input, **kwargs
                )
            elif need_x:
                grad_input = mm_kernels[1](grad_output, weight, **kwargs)
            elif need_w:
                grad_weight = mm_kernels[2](grad_output, input, **kwargs)
        # Sum over the batch axes the weight and bias broadcast over; a
        # vmapped weight/bias keeps its members axis.
        if grad_weight is not None:
            grad_weight = grad_weight.sum_to_size(ctx.w_shape)
        if need_b and bias is not None:
            grad_b = grad_output.sum_to_size(bias.shape)

        return grad_input, grad_weight, grad_b, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        *args: Any,
    ) -> tuple[torch.Tensor, int | None]:
        """Batching rule for ``torch.vmap``."""
        return _linear_vmap(_XWT.apply, info, in_dims, *args)


class _XW(torch.autograd.Function):
    """Distributed batched matmul ``x @ w``."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor | None,
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
        # Cast the bias to the autocast dtype (the kernel casts x/w itself) so
        # the bias add does not promote the output back to fp32 under autocast.
        (b,) = cast_to_autocast_dtype(b)
        xw = config.mm_kernels[1](x, w, local_rank, device, model_group)
        if b is not None:
            xw = xw.add(b)
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
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
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
        input, weight, bias = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        kwargs = dict(
            local_rank=ctx.local_rank, device=ctx.device, model_group=ctx.model_group
        )
        mm_kernels = ctx.config.mm_kernels

        # Same skip on every rank of the model group; see _XWT.backward.
        grad_input = grad_weight = grad_b = None
        with comm_trace.scope(ctx.comm_layer, "backward"):
            if need_x:
                grad_input = mm_kernels[0](grad_output, weight, **kwargs)
            if need_w:
                grad_weight = mm_kernels[2](input, grad_output, **kwargs)
        if grad_weight is not None:
            # Sum over the batch axes the weight broadcasts over; a vmapped
            # weight keeps its members axis.
            grad_weight = grad_weight.sum_to_size(ctx.w_shape)
        if need_b and bias is not None:
            grad_b = grad_output.sum_to_size(bias.shape)

        return grad_input, grad_weight, grad_b, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        *args: Any,
    ) -> tuple[torch.Tensor, int | None]:
        """Batching rule for ``torch.vmap``."""
        return _linear_vmap(_XW.apply, info, in_dims, *args)


class _XTW(torch.autograd.Function):
    """Distributed batched matmul ``x.mT @ w``."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor | None,
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
        # Cast the bias to the autocast dtype (the kernel casts x/w itself) so
        # the bias add does not promote the output back to fp32 under autocast.
        (b,) = cast_to_autocast_dtype(b)
        xtw = config.mm_kernels[2](x, w, local_rank, device, model_group)
        if b is not None:
            xtw = xtw.add(b)
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
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
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
        input, weight, bias = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        kwargs = dict(
            local_rank=ctx.local_rank, device=ctx.device, model_group=ctx.model_group
        )
        mm_kernels = ctx.config.mm_kernels

        # Same skip on every rank of the model group; see _XWT.backward.
        grad_input = grad_weight = grad_b = None
        with comm_trace.scope(ctx.comm_layer, "backward"):
            if need_x:
                grad_input = mm_kernels[0](weight, grad_output, **kwargs)
            if need_w:
                grad_weight = mm_kernels[1](input, grad_output, **kwargs)
        if grad_weight is not None:
            # Sum over the batch axes the weight broadcasts over; a vmapped
            # weight keeps its members axis.
            grad_weight = grad_weight.sum_to_size(ctx.w_shape)
        if need_b and bias is not None:
            grad_b = grad_output.sum_to_size(bias.shape)

        return grad_input, grad_weight, grad_b, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        *args: Any,
    ) -> tuple[torch.Tensor, int | None]:
        """Batching rule for ``torch.vmap``."""
        return _linear_vmap(_XTW.apply, info, in_dims, *args)
