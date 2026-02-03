from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist


class DistributedLayerNorm(torch.nn.Module):
    """LayerNorm with synchronized gradients across a process group."""

    def __init__(self, dim, rank, device, norm_group):
        super().__init__()
        self.rank = rank
        self.device = device
        self.norm_group = norm_group

        # Learnable parameters
        self.weight = torch.nn.Parameter(torch.ones(dim, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(dim, device=device))
        self.dim = dim
        self.eps = 1e-5
        self.layernorm = DistributedLNFunction().apply

    def forward(self, x):
        return self.layernorm(
            x, self.weight, self.bias, self.norm_group, self.eps
        )[0]


class DistributedLNFunction(torch.autograd.Function):
    """Autograd function for distributed LayerNorm."""

    @staticmethod
    def forward(x, weight, bias, norm_group, eps):
        # Compute mean and variance locally
        
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + eps)

        # Apply affine transformation
        y = weight * x_norm + bias
        return y, x_norm, weight, var + eps

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, norm_group, eps = inputs
        _, x_norm, weight, var = output

        ctx.mark_non_differentiable(x_norm, weight, var)
        ctx.norm_group = norm_group

        # Save for backward
        ctx.save_for_backward(x_norm, weight, torch.sqrt(var))

    @staticmethod
    def vmap(info, in_dims, x, weight, bias, norm_group, eps):
        out = DistributedLNFunction.apply(
            x, weight, bias, norm_group, eps    
        )
        
        return out, 0

    @staticmethod
    def backward(
        ctx, grad_output, x_norm_grad: None, weight_grad: None, var_grad: None
    ):
        
        x_norm, weight, std = ctx.saved_tensors
        norm_group = ctx.norm_group

        N = x_norm.shape[-1]  # normalized dimension size

        # Gradients for weight and bias
        # Sum over batch and sequence dimensions, keep feature dim
        
        grad_weight = (grad_output * x_norm).sum(
            dim=tuple(range(grad_output.dim() - 1))
        )
        # print(f'grad x {grad_output.sum()}')

        # print(f"grad out shape {grad_output.shape}, grad weight shape {grad_weight.shape}")
        grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        # Synchronize parameter gradients across the process group
        dist.all_reduce(grad_weight, op=dist.ReduceOp.SUM, group=norm_group)
        dist.all_reduce(grad_bias, op=dist.ReduceOp.SUM, group=norm_group)

        # Gradient for input (standard LayerNorm backward)
        grad_x_norm = grad_output * weight

        # LayerNorm backward formula
        grad_x = (1.0 / std) * (
            grad_x_norm
            - grad_x_norm.mean(dim=-1, keepdim=True)
            - x_norm * (grad_x_norm * x_norm).mean(dim=-1, keepdim=True)
        )

        # If input gradients also need sync (depends on your parallelism strategy)
        # dist.all_reduce(grad_x, op=dist.ReduceOp.AVG, group=norm_group)

        return grad_x, grad_weight, grad_bias, None, None


# class DistributedLayerNorm(torch.nn.Module):
#     """Batched layer norm distributed across four processes."""

#     def __init__(self, dim, rank, device, norm_group):
#         """
#         Initialize layer normalization.

#         Parameters.
#         ----------
#         dim : int
#             The dimension of the input features.
#         rank : int
#             Global rank.
#         device : torch.device
#             Device to perform computation on.
#         """
#         super().__init__()

#         self.rank = rank
#         self.device = device
#         self.norm_group = norm_group
#         self.layer_norm = torch.nn.LayerNorm(dim).to(device)

#     def forward(self, x):
#         """Forward pass of layer norm distributed across n processes."""
#         x = self.layer_norm(x)

#         return x


# class DistributedLN(torch.autograd.Function):
#     """Autograd function for layer norm function distributed across four processes."""

#     @staticmethod
#     def forward(ctx, x, norm_group):
#         """
#         Forward pass of layer norm distributed across four processes.

#         ctx : torch.autograd.Function
#             Context object to save information for backward computation.
#         """
#         ctx.norm_group = norm_group
#         return x

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass of layer norm distributed across four processes.

#         Parameters
#         ----------
#         ctx : torch.autograd.Function
#             Context object to save information for backward computation.
#         grad_output: torch.torch.Tensor
#             Gradient of the loss with respect to the input of the layer.
#         """
#         dist.all_reduce(
#             grad_output, op=dist.ReduceOp.AVG, group=ctx.norm_group, async_op=False
#         )

#         return grad_output, None, None


class DistributedSoftMax(torch.nn.Module):
    """Distributed Softmax operation."""

    # TODO: check for correctness
    def __init__(self, dim=-1, group=None):
        super().__init__()
        self.dim = dim
        self.group = group  # is not None else dist.group.WORLD
        self.func = _DistributedSoftmax.apply

    def forward(self, x):
        return self.func(x, self.dim, self.group)


class _DistributedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(x, dim, group):
        """Overloaded Autograd function for distributed SoftMax."""
        # local exponentials
        # -x.amax(dim=dim, keepdim=True)) # softmax fails with norm if only 1 variable
        local_max = x.amax(dim=-1, keepdim=True)
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)

        y = torch.exp(x - local_max)

        local_sum = y.sum(dim=dim, keepdim=True)
        # global sum
        global_sum = local_sum.clone()

        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=group)

        # final softmax
        out = y / global_sum
        # print(f'My device {x.device}, out {out}')
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

        # print(f'device {grad_out.device} grad x {grad_x.sum()}')
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
