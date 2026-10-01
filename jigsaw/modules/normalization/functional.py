"""Autograd functions of the distributed normalization layers."""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from jigsaw.autocast import promote_to_fp32_under_autocast


class DistributedLNFunction(torch.autograd.Function):
    """Autograd function of the distributed LayerNorm."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize ``x`` over a last dimension sharded across ``channel_group``.

        Parameters
        ----------
        x : torch.Tensor
            Local input.
        weight : torch.Tensor
            Local scale.
        bias : torch.Tensor | None
            Local shift.
        spatial_group : dist.ProcessGroup
            Process group over which the parameter gradients are reduced.
        channel_group : dist.ProcessGroup
            Process group across which the last dimension is partitioned.
        eps : float
            Term added to the variance for numerical stability.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Output, normalized input, scale and variance plus ``eps``. All but the
            output are kept for the backward pass.
        """
        x, weight, bias = promote_to_fp32_under_autocast(x, weight, bias)
        # Compute mean and variance locally
        local_mean = x.mean(dim=-1, keepdim=True)
        local_var = x.var(dim=-1, keepdim=True, unbiased=False)

        mean = local_mean.clone()
        dist.all_reduce(mean, op=dist.ReduceOp.AVG, group=channel_group)

        var = local_var + (local_mean - mean) ** 2
        dist.all_reduce(var, op=dist.ReduceOp.AVG, group=channel_group)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + eps)

        # Apply affine transformation
        y = weight * x_norm + (bias if bias is not None else 0)  # type: ignore[operator]
        return y, x_norm, weight, var + eps

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple,
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Save the tensors and metadata for the backward pass."""
        x, weight, bias, spatial_group, channel_group, eps = inputs
        _, x_norm, weight, var = output

        ctx.mark_non_differentiable(x_norm, weight, var)
        ctx.spatial_group = spatial_group  # type: ignore[attr-defined]
        ctx.channel_group = channel_group  # type: ignore[attr-defined]

        # Save for backward
        ctx.save_for_backward(x_norm, weight, torch.sqrt(var))

    @staticmethod
    def vmap(
        info: Any,  # type: ignore[no-untyped-def]
        in_dims: Optional[tuple[int, ...]],
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        eps: float,
    ) -> tuple[torch.Tensor, int]:
        """Batching rule for ``torch.vmap``."""
        out = DistributedLNFunction.apply(
            x, weight, bias, spatial_group, channel_group, eps
        )

        return out, 0

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
        x_norm_grad: None,
        weight_grad: None,
        var_grad: None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None]:
        """Compute the gradients of the distributed LayerNorm.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context.
        grad_output : torch.Tensor
            Gradient of the output.
        x_norm_grad, weight_grad, var_grad : None
            Unused; the corresponding outputs are non-differentiable.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, None, None]
            Gradients of the input, weight and bias, followed by ``None`` for the
            non-differentiable arguments.
        """
        x_norm, weight, std = ctx.saved_tensors  # type: ignore[attr-defined]
        # Match the forward compute dtype: under autocast the saved tensors are
        # fp32 while ``grad_output`` arrives in the autocast dtype.
        grad_output = grad_output.to(x_norm.dtype)
        spatial_group = ctx.spatial_group  # type: ignore[attr-defined]
        channel_group = ctx.channel_group  # type: ignore[attr-defined]
        n = (
            x_norm.shape[-1] * channel_group.size()
        )  # normalized dimension size (global)

        # Gradients for weight and bias
        # Sum over batch and sequence dimensions, keep feature dim

        grad_weight = (grad_output * x_norm).sum(
            dim=tuple(range(grad_output.dim() - 1))
        )
        grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        # Synchronize parameter gradients across the process group
        g_w_handle = dist.all_reduce(
            grad_weight, op=dist.ReduceOp.SUM, group=spatial_group, async_op=True
        )
        g_b_handle = dist.all_reduce(
            grad_bias, op=dist.ReduceOp.SUM, group=spatial_group, async_op=True
        )

        # Gradient for input (standard LayerNorm backward)
        grad_x_norm = grad_output * weight

        grad_x_norm_mean = grad_x_norm.mean(dim=-1, keepdim=True)
        g_gx_handle = dist.all_reduce(
            grad_x_norm_mean, op=dist.ReduceOp.AVG, group=channel_group, async_op=True
        )

        x_grad_x_term = (grad_x_norm * x_norm).sum(dim=-1, keepdim=True)
        x_g_x_handle = dist.all_reduce(
            x_grad_x_term, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
        )

        g_gx_handle.wait()
        x_g_x_handle.wait()

        # LayerNorm backward formula
        grad_x = (1.0 / std) * (
            grad_x_norm - grad_x_norm_mean - x_norm * x_grad_x_term / n
        )

        g_w_handle.wait()
        g_b_handle.wait()

        return grad_x, grad_weight, grad_bias, None, None, None  # type: ignore[return-value]


class NoParamLNCtx(Protocol):
    """Attributes stored on the autograd context."""

    x_norm: torch.Tensor
    std: torch.Tensor


class DistributedNoParamLNFunction(torch.autograd.Function):
    """Autograd function of the distributed LayerNorm without parameters."""

    @staticmethod
    def forward(
        x: torch.Tensor, channel_group: dist.ProcessGroup, eps: float
    ) -> tuple[torch.Tensor, float]:
        """Normalize ``x`` over a last dimension sharded across ``channel_group``.

        Parameters
        ----------
        x : torch.Tensor
            Local input.
        channel_group : dist.ProcessGroup
            Process group across which the last dimension is partitioned.
        eps : float
            Term added to the variance for numerical stability.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Normalized input and the variance plus ``eps``, which is kept for the
            backward pass.
        """
        (x,) = promote_to_fp32_under_autocast(x)
        # Compute mean and variance locally

        n_l = x.shape[-1]  # normalized dimension size (local)
        n_g = n_l * channel_group.size()  # normalized dimension size (global)

        local_mean = x.mean(dim=-1, keepdim=True)
        local_var = x.var(dim=-1, keepdim=True)

        mean = local_mean.clone()
        dist.all_reduce(mean, op=dist.ReduceOp.AVG, group=channel_group)

        var = ((n_l - 1) / (n_g - 1)) * local_var + (n_l / (n_g - 1)) * (
            mean - local_mean
        ) ** 2
        dist.all_reduce(var, op=dist.ReduceOp.SUM, group=channel_group)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + eps)

        # Apply affine transformation
        return x_norm, var + eps  # type: ignore[return-value]

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[torch.Tensor, dist.ProcessGroup, float],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Save the tensors and metadata for the backward pass."""
        x, channel_group, eps = inputs
        x_norm, var = output

        ctx.mark_non_differentiable(var)
        ctx.channel_group = channel_group  # type: ignore[attr-defined]
        std = torch.sqrt(var)
        # Save for backward
        ctx.save_for_backward(x_norm, std)

    @staticmethod
    def vmap(
        info: Any,
        in_dims: Optional[tuple[int, ...]],
        x: torch.Tensor,
        channel_group: dist.ProcessGroup,
        eps: float,
    ) -> tuple[torch.Tensor, int]:
        """Batching rule for ``torch.vmap``."""
        out = DistributedNoParamLNFunction.apply(x, channel_group, eps)

        return out, 0

    @staticmethod
    def backward(
        ctx: NoParamLNCtx, grad_output: torch.Tensor, var_grad: None
    ) -> tuple[torch.Tensor | None, ...]:
        """Compute the gradient of the distributed LayerNorm without parameters.

        Parameters
        ----------
        ctx : NoParamLNCtx
            Autograd context.
        grad_output : torch.Tensor
            Gradient of the output.
        var_grad : None
            Unused; the variance output is non-differentiable.

        Returns
        -------
        tuple[torch.Tensor | None, ...]
            Gradient of the input, followed by ``None`` for the non-differentiable
            arguments.
        """
        x_norm, std = ctx.saved_tensors  # type: ignore[attr-defined]
        # Match the forward compute dtype (fp32 under autocast).
        grad_output = grad_output.to(x_norm.dtype)
        channel_group = ctx.channel_group  # type: ignore[attr-defined]

        n = (
            x_norm.shape[-1] * channel_group.size()
        )  # normalized dimension size (global)

        # Gradients for weight and bias
        # Sum over batch and sequence dimensions, keep feature dim

        grad_x_norm_mean = grad_output.mean(dim=-1, keepdim=True)
        g_gx_handle = dist.all_reduce(
            grad_x_norm_mean, op=dist.ReduceOp.AVG, group=channel_group, async_op=True
        )

        # Gradient for input (standard LayerNorm backward)
        x_grad_x_term = (grad_output * x_norm).sum(dim=-1, keepdim=True)
        x_g_x_handle = dist.all_reduce(
            x_grad_x_term, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
        )

        g_gx_handle.wait()
        x_g_x_handle.wait()

        # LayerNorm backward formula
        grad_x = (1.0 / std) * (
            grad_output - grad_x_norm_mean - x_norm * x_grad_x_term / (n - 1)
        )

        return grad_x, None, None


class SoftMaxCtx(Protocol):
    """Attributes stored on the autograd context."""

    dim: int
    group: dist.ProcessGroup


class _DistributedSoftmax(torch.autograd.Function):
    """Autograd function of the distributed softmax."""

    @staticmethod
    def forward(x: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
        """Compute the softmax over a dimension sharded across ``group``."""
        (x,) = promote_to_fp32_under_autocast(x)
        # local exponentials
        local_max = x.amax(dim=dim, keepdim=True)
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)

        y = torch.exp(x - local_max)

        local_sum = y.sum(dim=dim, keepdim=True)
        # global sum
        global_sum = local_sum.clone()

        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=group)

        # final softmax
        out = y / global_sum
        return out

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
    ) -> None:
        """Save the softmax and metadata for the backward pass."""
        _, dim, group = inputs
        out = output
        ctx.save_for_backward(out)
        c = cast(SoftMaxCtx, ctx)

        c.dim = dim
        c.group = group

    @staticmethod
    def backward(
        ctx: SoftMaxCtx, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        """Compute the gradient of the input."""
        (out,) = ctx.saved_tensors  # type: ignore[attr-defined]
        # Match the forward compute dtype (fp32 under autocast).
        grad_out = grad_out.to(out.dtype)
        dim = ctx.dim
        group = ctx.group

        # local grad numerator
        local_grad = grad_out * out
        local_sum = local_grad.sum(dim=dim, keepdim=True)
        # global sum for gradient
        global_grad_sum = local_sum.clone()
        dist.all_reduce(global_grad_sum, op=dist.ReduceOp.SUM, group=group)

        grad_x = out * (grad_out - global_grad_sum)

        return grad_x, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        x: torch.Tensor,
        dim: int,
        group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, int]:
        """Batching rule for ``torch.vmap``."""
        # The batch dimension is in front, so a non-negative ``dim`` moves by one.
        out = _DistributedSoftmax.apply(x, dim + 1 if dim >= 0 else dim, group)

        return out, 0
