"""Implementation of distributed normalization layers."""

from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist


class DistributedLayerNorm(torch.nn.Module):
    """LayerNorm with synchronized gradients across a process group."""

    def __init__(
        self,
        local_dim: int,
        local_rank: int,
        device: torch.device,
        row_group: dist.ProcessGroup,
        col_group: dist.ProcessGroup,
    ):
        """Initialize torch module for distributed layer norm."""
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.spatial_group = row_group
        self.channel_group = col_group

        # Learnable parameters
        self.weight = torch.nn.Parameter(torch.ones(local_dim, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(local_dim, device=device))
        self.normalized_dim = local_dim
        self.eps = 1e-5
        self.layernorm = DistributedLNFunction.apply

    def get_parameter_count(self) -> int:
        """Return parameter count for distributed layer norm."""
        return (self.weight.numel() + self.bias.numel()) // dist.get_world_size(
            group=self.spatial_group
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call forward function for distributed layer norm."""
        return self.layernorm(  # type: ignore[no-any-return]
            x, self.weight, self.bias, self.spatial_group, self.channel_group, self.eps
        )[0]


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


class DistributedGroupNorm(torch.nn.Module):
    """Distributed Group Norm wrapped in module."""

    def __init__(  # type: ignore[no-untyped-def]
        self,
        num_local_groups,  # type: ignore[no-untyped-def]
        num_channels,  # type: ignore[no-untyped-def]
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
        bias=True,
        channel_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()

        self.num_local_groups = num_local_groups
        self.num_channels = num_channels
        self.num_global_groups = num_local_groups * (
            channel_group.size() if channel_group is not None else 1
        )
        self.eps = eps
        self.affine = affine
        self.device = device
        self.dtype = dtype
        self.bias = bias
        self.group_norm = torch.nn.GroupNorm(
            self.num_local_groups,
            self.num_channels,
            eps=self.eps,
            affine=self.affine,
            device=self.device,
            dtype=self.dtype,
            bias=self.bias,
        )

    def reshard_parameters(
        self,
        from_process_mesh: dist.ProcessGroup,
        to_process_mesh: dict[str, dist.ProcessGroup],
        pm_row_key: str = "row",
        pm_col_key: str = "col",
    ) -> None:
        """Reshard parameters for a distributed group norm."""
        assert self.num_global_groups % to_process_mesh[pm_col_key] == 0, (
            "number of global LN groups must be divisible by to_process_mesh channel group size."
        )


class DistributedNoParamLayerNorm(torch.nn.Module):
    """LayerNorm with synchronized gradients across a process group."""

    def __init__(
        self, rank: int, device: torch.device, channel_group: dist.ProcessGroup
    ) -> None:
        """
        Initialize DistributedNoParamLayerNorm.

        inputs
        ------
        rank: int
        device: torch.device
        channel_group: dist.ProcessGroup
        """
        super().__init__()
        self.rank = rank
        self.device = device
        self.channel_group = channel_group

        self.eps = 1e-5
        self.layernorm = DistributedNoParamLNFunction().apply

    def get_parameter_count(self) -> int:
        """Return 0 parameters in this layer norm."""
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call forward pass for layer norm."""
        return self.layernorm(x, self.channel_group, self.eps)[0]  # type: ignore[no-any-return]


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


class DistributedSoftMax(torch.nn.Module):
    """Distributed Softmax operation."""

    def __init__(self, group: dist.ProcessGroup, dim: int = -1):
        super().__init__()
        self.dim = dim
        self.group = group
        self.softmax = _DistributedSoftmax.apply

    def get_parameter_count(self) -> int:
        """Return parameter count of softmax."""
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of distributed softmax."""
        return self.softmax(x, self.dim, self.group)  # type: ignore[no-any-return]


class _DistributedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
        """Overloaded Autograd function for distributed SoftMax."""
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
        """Vmap rule for softmax."""
        out = _DistributedSoftmax.apply(x, dim, group)

        return out, 0
