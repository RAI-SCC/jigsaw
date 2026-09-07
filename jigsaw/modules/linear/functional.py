"""Autograd function definition for distributed linear layers."""

from typing import Any, Optional, Protocol, Union, cast

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from jigsaw.autocast import cast_to_autocast_dtype

from .config import LinearConfig


class LinearCtx(Protocol):
    """Protocol for types stored within ctx."""

    saved_tensors: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    local_rank: int
    device: torch.Tensor
    config: LinearConfig
    model_group: dist.ProcessGroup
    spatial_group: dist.ProcessGroup
    b: Optional[Parameter]


def _setup_linear_ctx(
    ctx: torch.autograd.function.FunctionCtx, inputs: Any
) -> None:
    """
    Populate ``ctx`` for the distributed linear autograd Functions.

    Shared by :class:`XWT`, :class:`XW` and :class:`XTW`; their ``setup_context``
    hooks differ in nothing but this bookkeeping. Runs once per forward call, so
    the extra function call is not on any per-element hot path.

    ctx: FunctionCtx
    inputs: tuple(x, w, b, local_rank, device, config, model_group)
    """
    x, w, b, local_rank, device, config, model_group = inputs
    ctx.save_for_backward(x, w, b)
    c = cast(LinearCtx, ctx)
    c.local_rank = local_rank
    c.device = device
    c.config = config
    c.model_group = model_group
    c.b = b


def _linear_vmap(
    info: Any,
    in_dims: tuple[Optional[int], ...],
    fn: Any,
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    local_rank: int,
    device: torch.device,
    config: LinearConfig,
    model_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, Optional[int]]:
    """
    Shared ``vmap`` rule for the distributed linear autograd Functions.

    The staticmethod protocol has already exposed the mapped (ensemble / members)
    dimension as a plain leading axis. Move it to 0 on every operand, give the
    weight and bias singleton batch axes so the kernels broadcast one member's
    parameters over the whole real batch, and make a *single* un-vmapped ``fn``
    call: the members axis rides through as an extra leading batch dimension, so
    every collective still runs exactly once (E x the payload, not E rounds).
    The backward reduces the weight grad over the real batch axis only
    (``LinearConfig._grad_sum`` keys off the extra leading dim), and the output's
    mapped dimension sits at position 0.
    """
    xd, wd, bd = in_dims[0], in_dims[1], in_dims[2]
    x = x.movedim(xd, 0) if xd is not None else x.expand(info.batch_size, *x.shape)
    w = w.movedim(wd, 0) if wd is not None else w.expand(info.batch_size, *w.shape)
    if b is not None:
        b = b.movedim(bd, 0) if bd is not None else b.expand(info.batch_size, *b.shape)
    # (E, *) -> broadcast one member's weight/bias over the real batch.
    w = w.unsqueeze(1)
    if b is not None:
        b = b.unsqueeze(1).unsqueeze(1)
    out = fn(x.contiguous(), w, b, local_rank, device, config, model_group)
    return out, 0


class XWT(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

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
        """
        Forward pass of linear layer (XWT) distributed across parallelism processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim) or (S, B, x_in, in_dim).
        w : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim) or (S, in_dim, out_dim).
        b : torch.Tensor
            Local bias tensor of shape (out_dim) or (S, out_dim).
        local_rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        config : LinearConfig
            Instance of LinearConfig class.
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.


        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim) or (S, B, out_dim, in_dim).
        """
        # Opt into autocast: this Function hides the inner matmul from autocast's
        # op-level policy. ``XWT`` casts x/w here (its kernel does not); casting
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
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, w, b, local_rank, device, config, model_group)
        output: xwt
        """
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
    ]:
        """
        Backward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : tensor
            The gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        input, weight, _ = ctx.saved_tensors
        local_rank = ctx.local_rank
        device = ctx.device
        config = ctx.config
        model_group = ctx.model_group

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
        # sum along the batch dimension (weight.ndim > 2 -> vmap members path)
        grad_weight = config.grad_sum(grad_weight, weight.ndim > 2)

        grad_b = config.grad_bias(grad_output)

        return grad_input, grad_weight, grad_b, None, None, None, None, None

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
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Define XWT behavior under torch.vmap (see :func:`_linear_vmap`)."""
        return _linear_vmap(
            info, in_dims, XWT.apply, x, w, b, local_rank, device, config, model_group
        )


class XW(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XW) multiplication distributed across two processes."""

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
        """
        Forward pass of linear layer (XW) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
            B, x_in are global dimensions.
            in_dim is local dimension.
        w : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim).
            in_dim is global dimension.
        b : torch.Tensor
            Local bias tensor of shape (out_dim)
        local_rank : int
            Local model rank
        device : torch.device
            Device to perform computation on.
        config: LinearConfig
            Instance of LinearConfig class.
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        C: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim) or (S, B, x_in, out_dim).
        """
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
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, w, b, local_rank, device, parallelism, model_group)
        output: xwt
        """
        _setup_linear_ctx(ctx, inputs)

    @staticmethod
    def backward(
        ctx: LinearCtx, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None
    ]:
        """
        Backward pass of linear layer (XW) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        input, weight, bias = ctx.saved_tensors
        local_rank = ctx.local_rank
        device = ctx.device
        config = ctx.config
        model_group = ctx.model_group

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

        grad_weight = config.grad_sum(grad_weight, weight.ndim > 2)
        grad_b = config.grad_bias(grad_output)

        return grad_input, grad_weight, grad_b, None, None, None, None

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
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Define XW behavior under torch.vmap (see :func:`_linear_vmap`)."""
        return _linear_vmap(
            info, in_dims, XW.apply, x, w, b, local_rank, device, config, model_group
        )


class XTW(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XW) multiplication distributed across two processes."""

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
        """
        Forward pass of linear layer (XTW).

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
            B, x_in are global dimensions.
            in_dim is local dimension.
        w : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim).
            in_dim is global dimension.
        b : torch.Tensor
            Local bias tensor of shape (out_dim)
        local_rank : int
            Local model rank
        device : torch.device
            Device to perform computation on.
        config: LinearConfig
            Instance of LinearConfig class.
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        C: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim) or (S, B, x_in, out_dim).
        """
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
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, w, b, local_rank, device, parallelism, model_group)
        output: xwt
        """
        _setup_linear_ctx(ctx, inputs)

    @staticmethod
    def backward(
        ctx: LinearCtx, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None
    ]:
        """
        Backward pass of linear layer (XW) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        input, weight, _ = ctx.saved_tensors
        local_rank = ctx.local_rank
        device = ctx.device
        config = ctx.config
        model_group = ctx.model_group

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

        grad_weight = config.grad_sum(grad_weight, weight.ndim > 2)
        grad_b = config.grad_bias(grad_output)

        return grad_input, grad_weight, grad_b, None, None, None, None

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
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Define XTW behavior under torch.vmap (see :func:`_linear_vmap`)."""
        return _linear_vmap(
            info, in_dims, XTW.apply, x, w, b, local_rank, device, config, model_group
        )
