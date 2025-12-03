import torch
import torch.distributed as dist


class DistributedLayerNorm(torch.nn.Module):
    """Batched layer norm distributed across four processes."""

    def __init__(self, dim, rank, device, norm_group):
        """
        Initialize layer normalization.

        Parameters.
        ----------
        dim : int
            The dimension of the input features.
        rank : int
            Global rank.
        device : torch.device
            Device to perform computation on.
        """
        super().__init__()

        self.rank = rank
        self.device = device
        self.norm_group = norm_group
        self.layer_norm = torch.nn.LayerNorm(dim).to(device)
        #self.layer_norm_apply = DistributedLN.apply

    def forward(self, x):
        """Forward pass of layer norm distributed across n processes."""
        #x = self.layer_norm_apply(x, self.norm_group)
        x = self.layer_norm(x)

        return x


class DistributedLN(torch.autograd.Function):
    """Autograd function for layer norm function distributed across four processes."""

    @staticmethod
    def forward(ctx, x, norm_group):
        """
        Forward pass of layer norm distributed across four processes.

        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        """
        ctx.norm_group = norm_group
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of layer norm distributed across four processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output: torch.torch.Tensor
            Gradient of the loss with respect to the input of the layer.
        """
        dist.all_reduce(
            grad_output,
            op=dist.ReduceOp.AVG,
            group=ctx.norm_group,
            async_op=False
        )

        return grad_output, None, None


class DistributedSoftMax(torch.nn.Module):
    """Distributed Softmax operation."""

    # TODO: check for correctness
    def __init__(self, dim=-1, group=None):
        super().__init__()
        self.dim = dim
        self.group = group if group is not None else dist.group.WORLD
        self.func = _DistributedSoftmax.apply

    def forward(self, x):
        return self.func(x, self.dim, self.group)


class _DistributedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, group):
        """Overloaded Autograd function for distirbuted SoftMax."""
        # store context
        ctx.dim = dim
        ctx.group = group

        # local exponentials

        # -x.amax(dim=dim, keepdim=True)) # softmax fails with norm if only 1 variable
        y = torch.exp(x)
        local_sum = y.sum(dim=dim, keepdim=True)
        # global sum
        global_sum = local_sum.clone()

        dist.all_reduce(
            global_sum, op=dist.ReduceOp.SUM, group=group)

        # final softmax
        out = y / global_sum
        ctx.save_for_backward(out, global_sum)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        out, global_sum = ctx.saved_tensors
        dim = ctx.dim
        group = ctx.group

        # local grad numerator
        local_grad = grad_out * out
        local_sum = local_grad.sum(dim=dim, keepdim=True)
        # global sum for gradient
        global_grad_sum = local_sum.clone()
        dist.all_reduce(
            global_grad_sum, op=dist.ReduceOp.SUM, group=group)

        grad_x = local_grad - out * global_grad_sum
        return grad_x, None, None
