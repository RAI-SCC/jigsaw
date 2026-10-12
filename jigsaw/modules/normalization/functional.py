"""Autograd functions of the distributed normalization layers."""

from typing import Any, Optional, Protocol

import torch
import torch.distributed as dist

from jigsaw._core.autocast import promote_to_fp32_under_autocast


class _LayerNorm(torch.autograd.Function):
    """Autograd function of the distributed LayerNorm."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
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
        x, weight, bias, channel_group, eps = inputs
        _, x_norm, weight, var = output

        ctx.mark_non_differentiable(x_norm, weight, var)
        ctx.channel_group = channel_group  # type: ignore[attr-defined]
        ctx.has_bias = bias is not None  # type: ignore[attr-defined]

        # Save for backward
        ctx.save_for_backward(x_norm, weight, torch.sqrt(var))

    @staticmethod
    def vmap(
        info: Any,  # type: ignore[no-untyped-def]
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        channel_group: dist.ProcessGroup,
        eps: float,
    ) -> tuple[torch.Tensor, int]:
        """Batching rule for ``torch.vmap``."""
        out = _LayerNorm.apply(x, weight, bias, channel_group, eps)

        return out, 0

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
        x_norm_grad: None,
        weight_grad: None,
        var_grad: None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None]:
        """Compute the gradients of the distributed LayerNorm.

        The weight and bias gradients are local; the layer sums them over its
        replicas with :func:`jigsaw.parallel.sync_grad`.

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
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, None]
            Gradients of the input, weight and bias (``None`` without a bias),
            followed by ``None`` for the non-differentiable arguments.
        """
        x_norm, weight, std = ctx.saved_tensors  # type: ignore[attr-defined]
        # Match the forward compute dtype: under autocast the saved tensors are
        # fp32 while ``grad_output`` arrives in the autocast dtype.
        grad_output = grad_output.to(x_norm.dtype)
        channel_group = ctx.channel_group  # type: ignore[attr-defined]
        n = (
            x_norm.shape[-1] * channel_group.size()
        )  # normalized dimension size (global)

        # Gradients for weight and bias
        # Sum over batch and sequence dimensions, keep feature dim

        grad_weight = (grad_output * x_norm).sum(
            dim=tuple(range(grad_output.dim() - 1))
        )
        grad_bias = (
            grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            if ctx.has_bias  # type: ignore[attr-defined]
            else None
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

        return grad_x, grad_weight, grad_bias, None, None  # type: ignore[return-value]


class LayerNormNoAffineCtx(Protocol):
    """Attributes stored on the autograd context."""

    x_norm: torch.Tensor
    std: torch.Tensor


class _LayerNormNoAffine(torch.autograd.Function):
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

        local_mean = x.mean(dim=-1, keepdim=True)
        local_var = x.var(dim=-1, keepdim=True, unbiased=False)

        mean = local_mean.clone()
        dist.all_reduce(mean, op=dist.ReduceOp.AVG, group=channel_group)

        # Combine each shard's *biased* variance (matching plain LayerNorm, which
        # divides by n rather than n-1) via the shifted-mean identity, then average
        # across equally-sized shards -- same combination as _LayerNorm,
        # minus the affine transform.
        var = local_var + (local_mean - mean) ** 2
        dist.all_reduce(var, op=dist.ReduceOp.AVG, group=channel_group)

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
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        channel_group: dist.ProcessGroup,
        eps: float,
    ) -> tuple[torch.Tensor, int]:
        """Batching rule for ``torch.vmap``."""
        out = _LayerNormNoAffine.apply(x, channel_group, eps)

        return out, 0

    @staticmethod
    def backward(
        ctx: LayerNormNoAffineCtx, grad_output: torch.Tensor, var_grad: None
    ) -> tuple[torch.Tensor | None, ...]:
        """Compute the gradient of the distributed LayerNorm without parameters.

        Parameters
        ----------
        ctx : LayerNormNoAffineCtx
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
            grad_output - grad_x_norm_mean - x_norm * x_grad_x_term / n
        )

        return grad_x, None, None
