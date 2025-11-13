import torch
import torch.distributed as dist
from distributed_convolutions import conv1d_twoway, conv2d_twoway, conv2d_fourway, conv1d_twoway_channel
import torch.nn.functional as F
from torch.nn.grad import conv1d_input, conv1d_weight, conv2d_input, conv2d_weight
from typing import Optional, Tuple, Union


class DistributedConv1D(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        rank,
        groups,
        device
    ):
        """
        Distributed 1D Convolution.

        random seed is required because of initialization/synchronization between "partner processes"

        input shape of forward pass: (N, Cin, Lin) or (Cin, Lin)
        output shape of forward pass: (N, Cout, Lout)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.rank = rank
        self.model_group = groups
        self.device = device
        self.n_parallel = 2
        
        self.conv1d = torch.nn.Conv1d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride
        )

        self.conv1d_apply = Conv1d().apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
            B, out_dim are global values.
            in_dim is local value.
        """
        Cij = self.conv1d_apply(
            x, self.conv1d.weight, self.conv1d.bias, self.kernel_size, self.stride, self.padding, self.dilation, self.rank, self.device, self.n_parallel, self.model_group
        )

        return Cij


class Conv1d(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.model_group = model_group
        ctx.b = bias
        
        x_aug, L_local = conv1d_twoway(
            x,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            rank=rank,
            model_group=model_group,
        )
        y = F.conv1d(
            x_aug,
            weight,
            bias,
            stride=stride,
            padding=padding
        )
        ctx.save_for_backward(x_aug, weight, torch.tensor([L_local], device=x.device))
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = (
            stride,
            padding,
            dilation,
            model_group,
        )
        ctx.rank, ctx.group = rank, model_group
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None]:
        """
        Backward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : tensor
            The gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        x_aug, weight, L_local_t = ctx.saved_tensors
        stride, padding, dilation, groups = (
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
        )
        rank, group = ctx.rank, ctx.group
        L_local = int(L_local_t.item())

        # parameter grads from native conv1d gradient helpers
        g_w = conv1d_weight(
            x_aug,
            weight.shape,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )
        g_b = grad_output.sum(dim=(0, 2)) if ctx.has_bias else None

        # make grads identical on all ranks
        dist.all_reduce(g_w, op=dist.ReduceOp.AVG, group=group)
        if g_b is not None:
            dist.all_reduce(g_b, op=dist.ReduceOp.AVG, group=group)
        
        # input grad then crop back to local slice
        g_x_aug = conv1d_input(
            x_aug.shape,
            weight,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )
        g_x = g_x_aug[..., :L_local] if (rank % 2 == 0) else g_x_aug[..., -L_local:]

        return g_x, g_w, g_b, None, None, None, None, None, None, None, None

class DistributedConv1D_channel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        rank,
        groups,
        device,
        n_parallel=2
    ):
        """
        Distributed 1D Convolution.

        random seed is required because of initialization/synchronization between "partner processes"

        input shape of forward pass: (N, Cin, Lin) or (Cin, Lin)
        output shape of forward pass: (N, Cout, Lout)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.rank = rank
        self.model_group = groups
        self.device = device
        self.n_parallel = 2
        
        self.conv1d = torch.nn.Conv1d(
            self.in_channels, self.out_channels//2, kernel_size=self.kernel_size, stride=self.stride
        )

        self.conv1d_apply = Conv1d_channel().apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
            B, out_dim are global values.
            in_dim is local value.
        """
        Cij = self.conv1d_apply(
            x, self.conv1d.weight, self.conv1d.bias, self.in_channels, self.stride, self.padding, self.dilation, self.rank, self.device, self.n_parallel, self.model_group
        )

        return Cij


class Conv1d_channel(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        in_channels: int,
        stride: int,
        padding: int,
        dilation: int,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.model_group = model_group
        ctx.b = bias
        
        y, x0, x1 = conv1d_twoway_channel(x, weight, in_channels, rank, model_group)
        x = torch.cat([x0, x1], dim=1)
        ctx.save_for_backward(x, weight)
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = (
            stride,
            padding,
            dilation,
            model_group,
        )
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None]:
        """
        Backward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : tensor
            The gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        x, weight = ctx.saved_tensors
        stride, padding, dilation = (
            ctx.stride,
            ctx.padding,
            ctx.dilation
        )
        rank, model_group = ctx.rank, ctx.model_group
        

        # parameter grads from native conv1d gradient helpers
        g_w = conv1d_weight(
            x,
            weight.shape,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )
        g_b = grad_output.sum(dim=(0, 2)) if ctx.has_bias else None

        
        # input grad then crop back to local slice
        g_x = conv1d_input(
            x.shape,
            weight,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )
        
        dist.all_reduce(g_x, op=dist.ReduceOp.AVG, group=model_group)

        return g_x, g_w, g_b, None, None, None, None, None, None, None, None

class DistributedConv2D(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        rank,
        groups,
        device,
        n_parallel
    ):
        """
        Distributed 1D Convolution.

        random seed is required because of initialization/synchronization between "partner processes"

        input shape of forward pass: (N, Cin, Lin) or (Cin, Lin)
        output shape of forward pass: (N, Cout, Lout)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.rank = rank
        self.model_group = groups
        self.device = device
        self.n_parallel = n_parallel
        
        self.conv2d = torch.nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride
        )

        self.conv2d_apply = Conv2d().apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
            B, out_dim are global values.
            in_dim is local value.
        """
        Cij = self.conv2d_apply(
            x, self.conv2d.weight, self.conv2d.bias, self.kernel_size, self.stride, self.padding, self.dilation, self.rank, self.device, self.n_parallel, self.model_group
        )
        
        return Cij


class Conv2d(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.model_group = model_group
        ctx.b = bias
        
        if n_parallel == 2:
            x_aug, H_local, W_local = conv2d_twoway(
                x,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                rank=rank,
                model_group=model_group,
            )
        elif n_parallel == 4:
            x_aug, H_local, W_local = conv2d_fourway(
                x,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                rank=rank,
                model_group=model_group,
            )
        
        y = F.conv2d(
            x_aug,
            weight,
            bias,
            stride=stride,
            padding=padding
        )
        
        ctx.save_for_backward(x_aug, weight,torch.tensor([H_local, W_local]))
        
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = (
            stride,
            padding,
            dilation,
            model_group,
        )
        ctx.rank, ctx.group = rank, model_group
        ctx.has_bias = bias is not None
        
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None]:
        """
        Backward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : tensor
            The gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        
        x_aug, weight, L_local_t = ctx.saved_tensors
        stride, padding, dilation, groups = (
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
        )
        rank, group = ctx.rank, ctx.group
        H_local = int(L_local_t[0].item())
        W_local = int(L_local_t[1].item())

        # parameter grads from native conv1d gradient helpers
        g_w = conv2d_weight(
            x_aug,
            weight.shape,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )
        
        g_b = grad_output.sum(dim=(0, 2, 3)) if ctx.has_bias else None
        
        # make grads identical on all ranks
        dist.all_reduce(g_w, op=dist.ReduceOp.AVG, group=group)
        if g_b is not None:
            dist.all_reduce(g_b, op=dist.ReduceOp.AVG, group=group)

        # input grad then crop back to local slice
        g_x_aug = conv2d_input(
            x_aug.shape,
            weight,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )
        
        match rank:
            case 0:
                g_x = g_x_aug[..., :H_local, :W_local] 
            case 1:
                g_x = g_x_aug[..., :H_local, -W_local:] 
            case 2:
                g_x = g_x_aug[..., -H_local:, :W_local] 
            case 3:
                g_x = g_x_aug[..., -H_local:, -W_local:]
        
        return g_x, g_w, g_b, None, None, None, None, None, None, None, None
