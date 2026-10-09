"""Autograd functions of the distributed activation layers."""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from jigsaw._core.autocast import promote_to_fp32_under_autocast


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
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        dim: int,
        group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, int]:
        """Batching rule for ``torch.vmap``."""
        # The batch dimension is in front, so a non-negative ``dim`` moves by one.
        out = _DistributedSoftmax.apply(x, dim + 1 if dim >= 0 else dim, group)

        return out, 0
