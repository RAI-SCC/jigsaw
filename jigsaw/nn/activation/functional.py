"""Functional forms and autograd functions of the distributed activations."""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from jigsaw._core.autocast import promote_to_fp32_under_autocast


def _active(group: Optional[dist.ProcessGroup]) -> bool:
    """Return whether ``group`` spans more than one rank."""
    return group is not None and group.size() > 1


class _SoftmaxCtx(Protocol):
    """Attributes stored on the autograd context."""

    dim: int
    group: dist.ProcessGroup


class _Softmax(torch.autograd.Function):
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
        c = cast(_SoftmaxCtx, ctx)

        c.dim = dim
        c.group = group

    @staticmethod
    def backward(
        ctx: _SoftmaxCtx, grad_out: torch.Tensor
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
        out = _Softmax.apply(x, dim + 1 if dim >= 0 else dim, group)

        return out, 0


class _LogSoftmax(torch.autograd.Function):
    """Autograd function of ``log_softmax`` over a dimension sharded across a group."""

    @staticmethod
    def forward(
        x: torch.Tensor, dim: int, group: Optional[dist.ProcessGroup]
    ) -> torch.Tensor:
        """Compute ``log_softmax`` with all-reduces over ``group``."""
        local_max = x.amax(dim=dim, keepdim=True)
        if _active(group):
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)

        shifted = x - local_max
        sum_exp = shifted.exp().sum(dim=dim, keepdim=True)
        if _active(group):
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=group)

        return shifted - sum_exp.log()

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[torch.Tensor, int, Optional[dist.ProcessGroup]],
        output: torch.Tensor,
    ) -> None:
        """Save the tensors and metadata for the backward pass."""
        _, dim, group = inputs
        ctx.dim = dim  # type: ignore[attr-defined]
        ctx.group = group  # type: ignore[attr-defined]
        ctx.save_for_backward(output)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        """Compute the gradient of the logits."""
        (log_probs,) = ctx.saved_tensors  # type: ignore[attr-defined]
        dim = ctx.dim  # type: ignore[attr-defined]
        group = ctx.group  # type: ignore[attr-defined]

        grad_sum = grad_output.sum(dim=dim, keepdim=True)
        if _active(group):
            dist.all_reduce(grad_sum, op=dist.ReduceOp.SUM, group=group)

        grad_x = grad_output - log_probs.exp() * grad_sum
        return grad_x, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        dim: int,
        group: Optional[dist.ProcessGroup],
    ) -> tuple[torch.Tensor, int]:
        """Batching rule for ``torch.vmap``."""
        bdim = in_dims[0]
        assert bdim is not None  # ``x`` is the only tensor input
        x = x.movedim(bdim, 0)
        # ``dim`` is logical; the batch dimension now sits in front of it.
        batched_dim = dim + 1 if dim >= 0 else dim
        return _LogSoftmax.apply(x, batched_dim, group), 0


def distributed_log_softmax(
    x: torch.Tensor, dim: int = -1, group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """Compute ``log_softmax`` over a dimension sharded across ``group``.

    Parameters
    ----------
    x : torch.Tensor
        Logits with the dimension ``dim`` holding the local shard.
    dim : int
        Dimension along which ``log_softmax`` is computed.
    group : dist.ProcessGroup, optional
        Group across which ``dim`` is partitioned.

    Returns
    -------
    torch.Tensor
        Log-probabilities of the local shard.
    """
    return _LogSoftmax.apply(x, dim, group)  # type: ignore[no-any-return]


@torch.no_grad()
def distributed_argmax(
    x: torch.Tensor,
    dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Compute the arg-max along a dimension sharded across ``group``.

    Ties are resolved towards the lowest global index.

    Parameters
    ----------
    x : torch.Tensor
        Values with the dimension ``dim`` holding the local shard.
    dim : int
        Dimension to reduce.
    group : dist.ProcessGroup, optional
        Group across which ``dim`` is partitioned.

    Returns
    -------
    torch.Tensor
        Global arg-max indices with ``dim`` removed.
    """
    local_max, local_arg = x.max(dim=dim)
    rank = 0 if group is None else group.rank()
    global_arg = local_arg + rank * x.shape[dim]
    if not _active(group):
        return global_arg

    n_global = x.shape[dim] * group.size()  # type: ignore[union-attr]
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)

    candidate = torch.where(
        local_max == global_max,
        global_arg,
        global_arg.new_full((), n_global),
    )
    dist.all_reduce(candidate, op=dist.ReduceOp.MIN, group=group)
    return candidate
