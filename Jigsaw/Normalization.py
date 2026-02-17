from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist


class DistributedLayerNorm(torch.nn.Module):
    """LayerNorm with synchronized gradients across a process group."""

    def __init__(self, normalized_dim, rank, device, spatial_group, channel_group):
        super().__init__()
        self.rank = rank
        self.device = device
        self.spatial_group = spatial_group
        self.channel_group = channel_group

        # Learnable parameters
        self.weight = torch.nn.Parameter(torch.ones(normalized_dim, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_dim, device=device))
        self.normalized_dim = normalized_dim
        self.eps = 1e-5
        self.layernorm = DistributedLNFunction().apply

    def forward(self, x):
        return self.layernorm(
            x, self.weight, self.bias, self.spatial_group, self.channel_group, self.eps
        )[0]


class DistributedLNFunction(torch.autograd.Function):
    """Autograd function for distributed LayerNorm."""

    @staticmethod
    def forward(x, weight, bias, spatial_group, channel_group, eps):
        # Compute mean and variance locally

        N_l = x.shape[-1]  # normalized dimension size (local)
        N_g = N_l * channel_group.size()  # normalized dimension size (global)

        local_mean = x.mean(dim=-1, keepdim=True)
        local_var = x.var(dim=-1, keepdim=True)

        mean = local_mean.clone()
        dist.all_reduce(mean, op=dist.ReduceOp.AVG, group=channel_group)

        var = ((N_l - 1) / (N_g - 1)) * local_var + (N_l / (N_g - 1)) * (mean - local_mean)**2
        dist.all_reduce(var, op=dist.ReduceOp.SUM, group=channel_group)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + eps)

        # Apply affine transformation
        y = weight * x_norm + bias
        return y, x_norm, weight, var + eps

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, spatial_group, channel_group, eps = inputs
        _, x_norm, weight, var = output

        ctx.mark_non_differentiable(x_norm, weight, var)
        ctx.spatial_group = spatial_group
        ctx.channel_group = channel_group

        # Save for backward
        ctx.save_for_backward(x_norm, weight, torch.sqrt(var))

    @staticmethod
    def vmap(info, in_dims, x, weight, bias, spatial_group, channel_group, eps):
        out = DistributedLNFunction.apply(
            x, weight, bias, spatial_group, channel_group, eps
        )
        
        return out, 0

    @staticmethod
    def backward(
        ctx, grad_output, x_norm_grad: None, weight_grad: None, var_grad: None
    ):
        
        x_norm, weight, std = ctx.saved_tensors
        spatial_group = ctx.spatial_group
        channel_group = ctx.channel_group

        N = x_norm.shape[-1] * channel_group.size()  # normalized dimension size (global)

        # Gradients for weight and bias
        # Sum over batch and sequence dimensions, keep feature dim
        
        grad_weight = (grad_output * x_norm).sum(
            dim=tuple(range(grad_output.dim() - 1))
        )
        grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        # Synchronize parameter gradients across the process group
        g_w_handle = dist.all_reduce(grad_weight, op=dist.ReduceOp.SUM, group=spatial_group, async_op=True)
        g_b_handle = dist.all_reduce(grad_bias, op=dist.ReduceOp.SUM, group=spatial_group, async_op=True)

        # Gradient for input (standard LayerNorm backward)
        grad_x_norm = grad_output * weight

        grad_x_norm_mean = grad_x_norm.mean(dim=-1, keepdim=True)
        g_gx_handle = dist.all_reduce(grad_x_norm_mean, op=dist.ReduceOp.AVG, group=channel_group, async_op=True)

        x_grad_x_term = (grad_x_norm * x_norm).sum(dim=-1, keepdim=True)
        x_g_x_handle = dist.all_reduce(x_grad_x_term, op=dist.ReduceOp.SUM, group=channel_group, async_op=True)

        g_gx_handle.wait()
        x_g_x_handle.wait()

        # LayerNorm backward formula
        grad_x = (1.0 / std) * (
            grad_x_norm
            - grad_x_norm_mean
            - x_norm * x_grad_x_term / (N - 1)
        )

        g_w_handle.wait()
        g_b_handle.wait()

        return grad_x, grad_weight, grad_bias, None, None, None


class DistributedNoParamLayerNorm(torch.nn.Module):
    """LayerNorm with synchronized gradients across a process group."""

    def __init__(self, rank, device, channel_group):
        super().__init__()
        self.rank = rank
        self.device = device
        self.channel_group = channel_group

        self.eps = 1e-5
        self.layernorm = DistributedNoParamLNFunction().apply

    def forward(self, x):
        return self.layernorm(
            x, self.channel_group, self.eps
        )[0]


class DistributedNoParamLNFunction(torch.autograd.Function):
    """Autograd function for distributed LayerNorm."""

    @staticmethod
    def forward(x, channel_group, eps):
        # Compute mean and variance locally

        N_l = x.shape[-1]  # normalized dimension size (local)
        N_g = N_l * channel_group.size()  # normalized dimension size (global)

        local_mean = x.mean(dim=-1, keepdim=True)
        local_var = x.var(dim=-1, keepdim=True)

        mean = local_mean.clone()
        dist.all_reduce(mean, op=dist.ReduceOp.AVG, group=channel_group)

        var = ((N_l - 1) / (N_g - 1)) * local_var + (N_l / (N_g - 1)) * (mean - local_mean) ** 2
        dist.all_reduce(var, op=dist.ReduceOp.SUM, group=channel_group)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + eps)

        # Apply affine transformation
        return x_norm, var + eps

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, spatial_group, channel_group, eps = inputs
        x_norm, var = output

        ctx.mark_non_differentiable(var)
        ctx.channel_group = channel_group

        # Save for backward
        ctx.save_for_backward(x_norm, torch.sqrt(var))

    @staticmethod
    def vmap(info, in_dims, x, spatial_group, channel_group, eps):
        out = DistributedNoParamLNFunction.apply(
            x, spatial_group, channel_group, eps
        )

        return out, 0

    @staticmethod
    def backward(
            ctx, grad_output, var_grad: None
    ):
        x_norm, std = ctx.saved_tensors
        channel_group = ctx.channel_group

        N = x_norm.shape[-1] * channel_group.size()  # normalized dimension size (global)

        # Gradients for weight and bias
        # Sum over batch and sequence dimensions, keep feature dim

        grad_x_norm_mean = grad_output.mean(dim=-1, keepdim=True)
        g_gx_handle = dist.all_reduce(grad_x_norm_mean, op=dist.ReduceOp.AVG, group=channel_group, async_op=True)

        # Gradient for input (standard LayerNorm backward)
        x_grad_x_term = (grad_output * x_norm).sum(dim=-1, keepdim=True)
        x_g_x_handle = dist.all_reduce(x_grad_x_term, op=dist.ReduceOp.SUM, group=channel_group, async_op=True)

        g_gx_handle.wait()
        x_g_x_handle.wait()

        # LayerNorm backward formula
        grad_x = (1.0 / std) * (
                grad_output
                - grad_x_norm_mean
                - x_norm * x_grad_x_term / (N - 1)
        )

        return grad_x, None, None

class DistributedSoftMax(torch.nn.Module):
    """Distributed Softmax operation."""

    def __init__(self, dim=-1, group=None):
        super().__init__()
        self.dim = dim
        self.group = group  
        self.func = _DistributedSoftmax.apply

    def forward(self, x):
        return self.func(x, self.dim, self.group)


class _DistributedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(x, dim, group):
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
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, dim, group)
        output: softmax
        """
        _, dim, group = inputs
        out = output
        ctx.dim = dim
        ctx.group = group
        ctx.save_for_backward(out)

    @staticmethod
    def backward(ctx, grad_out):
        (out,) = ctx.saved_tensors  # CHANGED: unpack tuple
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
    def vmap(info, in_dims: tuple[Optional[int], ...], x, dim, group):
        y = torch.exp(x - x.amax(dim=dim, keepdim=True))

        local_sum = y.sum(dim=dim, keepdim=True)
        # global sum
        global_sum = local_sum.clone()

        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=group)
        # final softmax
        out = y / global_sum

        # out has vmapped dimension at dim 0
        return out, 0
