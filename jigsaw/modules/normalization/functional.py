"""Autograd definition for autograd functions."""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist


def _fp32_under_autocast(
    *tensors: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], ...]:
    """Promote operands to (at least) fp32 while autocast is active.

    ``torch.autocast`` keeps ``layer_norm`` and ``softmax`` in fp32 by op-level
    policy -- their reductions (mean / variance, ``exp``) are too precision
    sensitive for bf16/fp16. A :class:`torch.autograd.Function` is opaque to that
    policy, so the LayerNorm and softmax Functions here opt in explicitly: under
    autocast every operand is cast up to ``promote_types(dtype, float32)``
    (fp16/bf16 -> fp32; fp32/fp64 unchanged), which also makes their output land
    in fp32 the way stock AMP does. Outside autocast the operands pass straight
    through.
    """
    ref = next((t for t in tensors if t is not None), None)
    if ref is None or not torch.is_autocast_enabled(ref.device.type):
        return tensors
    return tuple(
        None if t is None else t.to(torch.promote_types(t.dtype, torch.float32))
        for t in tensors
    )


class DistributedLNFunction(torch.autograd.Function):
    """Autograd function for distributed LayerNorm."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform forward pass for distributed layer norm."""
        x, weight, bias = _fp32_under_autocast(x, weight, bias)  # type: ignore[assignment]
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
        """Set up context for distributed layer norm."""
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
        """Define vmap for distributed layer norm."""
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
        """Perform backward pass for distributed layer norm."""
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
    """Context object for NoParamLN."""

    x_norm: torch.Tensor
    std: torch.Tensor


class DistributedNoParamLNFunction(torch.autograd.Function):
    """Autograd function for distributed LayerNorm."""

    @staticmethod
    def forward(
        x: torch.Tensor, channel_group: dist.ProcessGroup, eps: float
    ) -> tuple[torch.Tensor, float]:
        """Perform forward pass for distributed layer norm with no parameters."""
        (x,) = _fp32_under_autocast(x)  # type: ignore[assignment]
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
        """Set up context for distributed layer norm with no params."""
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
        """Define vmap for distributed NoParameter LN."""
        out = DistributedNoParamLNFunction.apply(x, channel_group, eps)

        return out, 0

    @staticmethod
    def backward(
        ctx: NoParamLNCtx, grad_output: torch.Tensor, var_grad: None
    ) -> tuple[torch.Tensor | None, ...]:
        """Define backward pass for layer norm with no parameters."""
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
    """Protocol for types stored within ctx."""

    dim: int
    group: dist.ProcessGroup


class _DistributedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
        """Overloaded Autograd function for distributed SoftMax."""
        (x,) = _fp32_under_autocast(x)  # type: ignore[assignment]
        # local exponentials
        local_max = x.amax(dim=-1, keepdim=True)
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
        """
        Set up context for distributed Softmax.

        ctx: FunctionCtx
        inputs: tuple(x, dim, group)
        output: softmax
        """
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
        """
        Backward pass of distributed SoftMax.

        Inputs
        ------
        ctx: Function context
        grad_out: gradient wrt output
        """
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
        """Vmap rule for softmax."""
        out = _DistributedSoftmax.apply(x, dim, group)

        return out, 0
