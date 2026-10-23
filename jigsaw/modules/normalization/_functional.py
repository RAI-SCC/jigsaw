"""Autograd functions of the distributed normalization layers."""

from collections.abc import Sequence
from typing import Any, Protocol

import torch
import torch.distributed as dist

from jigsaw._core.autocast import promote_to_fp32_under_autocast
from jigsaw._core.vmap import to_front


class _LayerNorm(torch.autograd.Function):
    """Autograd function of the distributed LayerNorm."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
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
        local_mean = x.mean(dim=-1, keepdim=True)
        local_var = x.var(dim=-1, keepdim=True, unbiased=False)

        mean = local_mean.clone()
        dist.all_reduce(mean, op=dist.ReduceOp.AVG, group=channel_group)

        var = local_var + (local_mean - mean) ** 2
        dist.all_reduce(var, op=dist.ReduceOp.AVG, group=channel_group)

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

        ctx.save_for_backward(x_norm, weight, torch.sqrt(var))

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        channel_group: dist.ProcessGroup,
        eps: float,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int | None, ...]]:
        """Batching rule for ``torch.vmap``.

        The vmapped axis of ``x`` is moved to the front; the normalization is
        independent over every leading axis. A vmapped weight or bias (e.g.
        stacked ensemble members) is viewed as ``(members, 1, ..., 1, C)`` so
        each member's parameters broadcast over its own input; the backward
        reduces their gradients back to that shape.
        """
        size = info.batch_size
        x_dim, w_dim, b_dim = in_dims[:3]
        x = to_front(x, x_dim, size)  # type: ignore[assignment]
        params_batched = w_dim is not None or b_dim is not None
        if params_batched:
            shape = (size, *([1] * (x.ndim - 2)), x.shape[-1])
            weight = to_front(weight, w_dim, size).reshape(shape)  # type: ignore[union-attr]
            bias = to_front(bias, b_dim, size)
            if bias is not None:
                bias = bias.reshape(shape)
        out = _LayerNorm.apply(x, weight, bias, channel_group, eps)
        return out, (0, 0, 0 if params_batched else None, 0)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
        x_norm_grad: None,
        weight_grad: None,
        var_grad: None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, None]:
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

        # (``sum_to_size`` keeps the members axis of a vmapped weight/bias, which
        # the batching rule shapes like the weight.)
        grad_weight = (grad_output * x_norm).sum_to_size(weight.shape)
        grad_bias = (
            grad_output.sum_to_size(weight.shape)
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
        local_mean = x.mean(dim=-1, keepdim=True)
        local_var = x.var(dim=-1, keepdim=True, unbiased=False)

        mean = local_mean.clone()
        dist.all_reduce(mean, op=dist.ReduceOp.AVG, group=channel_group)

        # Combine each shard's *biased* variance (matching plain LayerNorm, which
        # divides by n rather than n-1) via the shifted-mean identity, then average
        # across equally-sized shards, as in _LayerNorm.
        var = local_var + (local_mean - mean) ** 2
        dist.all_reduce(var, op=dist.ReduceOp.AVG, group=channel_group)

        x_norm = (x - mean) / torch.sqrt(var + eps)

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
        ctx.save_for_backward(x_norm, std)

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        x: torch.Tensor,
        channel_group: dist.ProcessGroup,
        eps: float,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
        """Batching rule for ``torch.vmap``.

        The vmapped axis of ``x`` is moved to the front; the normalization is
        independent over every leading axis.
        """
        x = x.movedim(in_dims[0], 0)  # type: ignore[arg-type]
        out = _LayerNormNoAffine.apply(x, channel_group, eps)
        return out, (0, 0)

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

        grad_x = (1.0 / std) * (
            grad_output - grad_x_norm_mean - x_norm * x_grad_x_term / n
        )

        return grad_x, None, None


def _all_reduce_sum(t: torch.Tensor, groups: Sequence[dist.ProcessGroup]) -> None:
    """Sum ``t`` in place over every group in turn, i.e. over their whole grid."""
    for g in groups:
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=g)


class _GroupNorm(torch.autograd.Function):
    """Autograd function of the GroupNorm over a spatially sharded input."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        num_groups: int,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        spatial_groups: Sequence[dist.ProcessGroup],
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize each channel group of ``x`` over all spatial shards.

        Parameters
        ----------
        x : torch.Tensor
            Local input of shape ``(N, C_local, *spatial_local)``.
        num_groups : int
            Number of local channel groups; no group straddles a channel shard.
        weight : torch.Tensor | None
            Local per-channel scale.
        bias : torch.Tensor | None
            Local per-channel shift.
        spatial_groups : Sequence[dist.ProcessGroup]
            Process groups across which the spatial dimensions are partitioned.
        eps : float
            Term added to the variance for numerical stability.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Output, normalized input, inverse standard deviation of shape
            ``(N, num_groups, 1)`` and the global number of elements per group.
            All but the output are kept for the backward pass.
        """
        x, weight, bias = promote_to_fp32_under_autocast(x, weight, bias)
        xg = x.reshape(x.shape[0], num_groups, -1)

        # Global count and sum per (sample, group) in one collective per group.
        stats = torch.stack(
            [torch.full_like(xg[..., 0], xg.shape[-1]), xg.sum(dim=-1)], dim=-1
        )
        _all_reduce_sum(stats, spatial_groups)
        n, total = stats.unbind(dim=-1)
        mean = (total / n).unsqueeze(-1)

        # Center on the global mean before squaring, as two-pass variance.
        centred = xg - mean
        var = (centred * centred).sum(dim=-1, keepdim=True)
        _all_reduce_sum(var, spatial_groups)
        n = n.unsqueeze(-1)
        rstd = torch.rsqrt(var / n + eps)

        x_norm = (centred * rstd).reshape(x.shape)
        affine_shape = (1, -1) + (1,) * (x.dim() - 2)
        # Without affine parameters the output must still be a separate tensor
        # from ``x_norm``, which is marked non-differentiable.
        y = x_norm if weight is not None or bias is not None else x_norm.clone()
        if weight is not None:
            y = y * weight.reshape(affine_shape)
        if bias is not None:
            y = y + bias.reshape(affine_shape)
        return y, x_norm, rstd, n

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple,
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Save the tensors and metadata for the backward pass."""
        _, num_groups, weight, bias, spatial_groups, _ = inputs
        _, x_norm, rstd, n = output

        ctx.mark_non_differentiable(x_norm, rstd, n)
        ctx.num_groups = num_groups  # type: ignore[attr-defined]
        ctx.spatial_groups = spatial_groups  # type: ignore[attr-defined]
        ctx.has_weight = weight is not None  # type: ignore[attr-defined]
        ctx.has_bias = bias is not None  # type: ignore[attr-defined]
        ctx.save_for_backward(x_norm, rstd, n, weight)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
        x_norm_grad: None,
        rstd_grad: None,
        n_grad: None,
    ) -> tuple[
        torch.Tensor,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
    ]:
        """Compute the gradients of the spatially sharded GroupNorm.

        The weight and bias gradients are local; the layer sums them over its
        replicas with :func:`jigsaw.parallel.sync_grad`.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context.
        grad_output : torch.Tensor
            Gradient of the output.
        x_norm_grad, rstd_grad, n_grad : None
            Unused; the corresponding outputs are non-differentiable.

        Returns
        -------
        tuple[torch.Tensor, None, torch.Tensor | None, torch.Tensor | None, None, None]
            Gradients of the input, weight (``None`` if not affine) and bias
            (``None`` without a bias), with ``None`` for the other arguments.
        """
        x_norm, rstd, n, weight = ctx.saved_tensors  # type: ignore[attr-defined]
        # Match the forward compute dtype (fp32 under autocast).
        grad_output = grad_output.to(x_norm.dtype)
        num_groups = ctx.num_groups  # type: ignore[attr-defined]

        # Reduce over batch and spatial dimensions, keep the channel dimension.
        reduce_dims = (0, *range(2, grad_output.dim()))
        grad_weight = (
            (grad_output * x_norm).sum(dim=reduce_dims)
            if ctx.has_weight  # type: ignore[attr-defined]
            else None
        )
        grad_bias = (
            grad_output.sum(dim=reduce_dims)
            if ctx.has_bias  # type: ignore[attr-defined]
            else None
        )

        grad_x_norm = grad_output
        if weight is not None:
            affine_shape = (1, -1) + (1,) * (grad_output.dim() - 2)
            grad_x_norm = grad_output * weight.reshape(affine_shape)
        shape = (x_norm.shape[0], num_groups, -1)
        gg = grad_x_norm.reshape(shape)
        xg = x_norm.reshape(shape)

        # Both per-group reductions in one collective per group.
        sums = torch.stack([gg.sum(dim=-1), (gg * xg).sum(dim=-1)], dim=-1)
        _all_reduce_sum(sums, ctx.spatial_groups)  # type: ignore[attr-defined]
        g_sum, gx_sum = (s.unsqueeze(-1) for s in sums.unbind(dim=-1))

        # GroupNorm backward formula
        grad_x = rstd * (gg - g_sum / n - xg * gx_sum / n)

        return grad_x.reshape(x_norm.shape), None, grad_weight, grad_bias, None, None
