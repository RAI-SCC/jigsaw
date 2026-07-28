"""Autograd function definition for distributed linear layers."""

from typing import Any, Optional, Protocol, Union, cast

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from .config import LinearConfig


class LinearCtx(Protocol):
    """Protocol for types stored within ctx."""

    saved_tensors: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    local_rank: int
    device: torch.Tensor
    config: LinearConfig
    model_group: dist.ProcessGroup
    spatial_group: dist.ProcessGroup
    bayesian: bool
    b: Optional[Parameter]


class XWT(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        local_rank: int,
        device: torch.device,
        config: LinearConfig,
        model_group: dist.ProcessGroup,
        bayesian: bool = False,
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
        bayesian: bool
            Flag for whether the model is bayesian.


        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim) or (S, B, out_dim, in_dim).
        """
        w = config.reshape_weights(w)
        if torch.is_autocast_enabled(device.type):
            dtype = torch.get_autocast_dtype(device.type)
            x = x.to(dtype)
            w = w.to(dtype)
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
        inputs: tuple(x, w, b, local_rank, device, config, model_group, bayesian)
        output: xwt
        """
        x, w, b, local_rank, device, config, model_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        c = cast(LinearCtx, ctx)
        c.local_rank = local_rank
        c.device = device
        c.config = config
        c.model_group = model_group
        c.bayesian = bayesian

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
        spatial_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XWT behavior under torch.vmap.

        Parameters
        ----------
        info:
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims:
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, rank, device, config, model_group, spatial_group, bayesian).
        x: torch.Tensor
            Possibly batched input.
        w: torch.Tensor
            Possibly shared or batched weights.
        b: torch.Tensor | None
            Possibly shared or batched bias.
        local_rank: int
            Local rank.
        device: torch.device
            Device where computation occurs.
        config: LinearConfig
            Instance of selection class with function pointers based on specific config.
        model_group: dist.ProcessGroup
            Domain- and tensor-parallel process group.
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.
        bayesian: bool
            Flag to see whether model is bayesian.

        Returns
        -------
        output : Tensor
            Batched output with vmapped dimension in front.
        out_dim : Optional[int]
            Index of vmapped dimension in output (0 if batched, None if not).
        """
        x = x.contiguous()
        out = XWT.apply(
            x, w, b, local_rank, device, config, model_group, spatial_group, bayesian
        )

        # out has vmapped dimension at dim 0
        return out, 0


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
        bayesian: bool,
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
        bayesian : bool
            Flag to indicate if the model is bayesian.

        Returns
        -------
        C: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim) or (S, B, x_in, out_dim).
        """
        w = config.reshape_weights(w)
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
        x, w, b, local_rank, device, config, model_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        c = cast(LinearCtx, ctx)
        c.local_rank = local_rank
        c.device = device
        c.config = config
        c.model_group = model_group
        c.b = b
        c.bayesian = bayesian

    @staticmethod
    def backward(
        ctx: LinearCtx, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None, None
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
        config: int,
        model_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XW behavior under torch.vmap.

        Parameters
        ----------
        info:
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims:
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, local_rank, device, parallelism, model_group).
        x : Tensor
            Batched input.
        w : Tensor
            Batched weights.
        b : Tensor or None
            Batched bias.
        local_rank : int
            Local model rank
        device : torch.device
            Device to perform computation on.
        config: LinearConfig
            Instance of LinearConfig class.
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        bayesian : bool
            Flag to indicate if the model is bayesian.

        Returns
        -------
        output : Tensor
            Batched output with vmapped dimension in front.
        out_dim : Optional[int]
            Index of vmapped dimension in output (0 if batched, None if not).
        """
        x = x.contiguous()
        out = XW.apply(x, w, b, local_rank, device, config, model_group, bayesian)

        # out has vmapped dimension at dim 0
        return out, 0


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
        bayesian: bool,
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
        bayesian : bool
            Flag to indicate if the model is bayesian.

        Returns
        -------
        C: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim) or (S, B, x_in, out_dim).
        """
        w = config.reshape_weights(w)
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
        x, w, b, local_rank, device, config, model_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        c = cast(LinearCtx, ctx)
        c.local_rank = local_rank
        c.device = device
        c.config = config
        c.model_group = model_group
        c.b = b
        c.bayesian = bayesian

    @staticmethod
    def backward(
        ctx: LinearCtx, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None, None
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
        config: int,
        model_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XW behavior under torch.vmap.

        Parameters
        ----------
        info:
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims:
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, local_rank, device, parallelism, model_group).
        x : Tensor
            Batched input.
        w : Tensor
            Batched weights.
        b : Tensor or None
            Batched bias.
        local_rank : int
            Local model rank
        device : torch.device
            Device to perform computation on.
        config: LinearConfig
            Instance of LinearConfig class.
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        bayesian : bool
            Flag to indicate if the model is bayesian.

        Returns
        -------
        output : Tensor
            Batched output with vmapped dimension in front.
        out_dim : Optional[int]
            Index of vmapped dimension in output (0 if batched, None if not).
        """
        x = x.contiguous()
        out = XWT.apply(x, w, b, local_rank, device, config, model_group, bayesian)

        # out has vmapped dimension at dim 0
        return out, 0
