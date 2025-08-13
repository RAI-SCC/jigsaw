import torch

class DistributedLayerNorm(torch.nn.Module): 
    """Batched layer norm distributed across four processes."""

    def __init__(self, dim, rank, device, model_group):
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
        self.model_group = model_group
        self.layer_norm = torch.nn.LayerNorm(dim).to(device)
        self.layer_norm_apply = DistributedLN.apply
    
    def forward(self, x):
        """Forward pass of layer norm distributed across n processes."""
        x = self.layer_norm_apply(x)
        x = self.layer_norm(x)

        return x
    
class DistributedLN(torch.autograd.Function):
    """Autograd function for layer norm function distributed across four processes."""

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of layer norm distributed across four processes.

        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        """
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
        return grad_output, None, None
    
class DistributedSoftMax(torch.nn.Module):
    """Distributed Softmax operation."""

    #TODO: check for correctness
    def __init__(self, dim=-1, group=None):
        super().__init__()
        self.dim = dim
        self.group = group if group is not None else torch.distributed.group.WORLD
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

        y = torch.exp(x)#-x.amax(dim=dim, keepdim=True)) # softmax fails with norm if only 1 variable
        local_sum = y.sum(dim=dim, keepdim=True)
        # global sum
        global_sum = local_sum.clone()

        torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM, group=group)
        
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
        torch.distributed.all_reduce(global_grad_sum, op=torch.distributed.ReduceOp.SUM, group=group)

        grad_x = local_grad - out * global_grad_sum
        return grad_x, None, None
