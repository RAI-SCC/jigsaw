from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from Jigsaw.distributed_convolutions import (
    conv1d_fourway_complete,
    conv1d_twoway,
    conv1d_twoway_channel,
    conv2d_fourway,
    conv2d_twoway,
    conv2d_twoway_channel,
)
from torch.nn.grad import conv1d_input, conv1d_weight, conv2d_input, conv2d_weight


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
        device,
        bias=False
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
        self.bias = bias

        self.conv1d = torch.nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=self.bias
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
            x,
            self.conv1d.weight,
            self.conv1d.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.rank,
            self.device,
            self.n_parallel,
            self.model_group,
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
        y = F.conv1d(x_aug, weight, bias, stride=stride, padding=padding)
        local_length = y.shape[-1]
        total_length = torch.tensor(y.shape[-1]).to(device)
        dist.all_reduce(total_length, op=dist.ReduceOp.SUM, group=model_group)  # group should be spatial group
        
        ctx.save_for_backward(x_aug, weight, torch.tensor([L_local], device=x.device))
        
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = (
            stride,
            padding,
            dilation,
            model_group,
        )
        ctx.local_length, ctx.total_length = local_length, total_length
        
        ctx.rank, ctx.group = rank, model_group
        ctx.has_bias = bias is not None
        # print(f'I am rank {rank} and y is {y}')
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
        # TODO: how to acess the total shape of the convolution?
        L_local = int(L_local_t.item())
        
        # parameter grads from native conv1d gradient helpers
        g_w = conv1d_weight(
            x_aug,
            weight.shape,
            grad_output * ctx.local_length / ctx.total_length.item(),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )
        
        # g_w = g_w * ctx.local_length / ctx.total_length.item()
        # print(f'I am rank {rank} and local length is {ctx.local_length} and total length {ctx.total_length.item()}\n grad output is {grad_output}, gw {g_w}')
        # print(f'I am rank {rank} and value of xaug is {x_aug}')
        
        # TODO: same weighing factor needed for bias term
        g_b = grad_output.sum(dim=(0, 2)) if ctx.has_bias else None

        # make grads identical on all ranks
        dist.all_reduce(g_w, op=dist.ReduceOp.SUM, group=group)
        
        if g_b is not None:
            # print(f'is this accessed?')
            dist.all_reduce(g_b, op=dist.ReduceOp.SUM, group=group)

        # input grad then crop back to local slice
        # print(f'whole x shape: {x_aug.shape}')
        # lets start by concatenating the grad outputs
        
        dummy = torch.zeros(1,1,4)
        right_to_left = 1
        left_to_right = 0
        
        empty = torch.empty(0,device=grad_output.device,dtype=grad_output.dtype)
        device = ctx.device
        dtype = grad_output.dtype
        # TODO: figure out the correct message to send
        match rank % 2:
            case 0:
                send = grad_output[..., -left_to_right:] if left_to_right > 0 else empty
                recv = (
                    torch.zeros((*grad_output.shape[:-1], right_to_left), device=device, dtype=dtype)
                    if right_to_left > 0
                    else empty
                )
                send = send.contiguous()
                dist.all_to_all(
                    [empty, recv], [empty, send], group=ctx.model_group, async_op=False
                )
                grad_output_augment = torch.cat((grad_output, recv if right_to_left > 0 else empty), dim=-1)
            case 1:
                send = grad_output[..., :right_to_left] if right_to_left > 0 else empty
                recv = (
                    torch.zeros((*grad_output.shape[:-1], left_to_right), device=device, dtype=dtype)
                    if left_to_right > 0
                    else empty
                )
                send = send.contiguous()
                dist.all_to_all(
                    [recv, empty], [send, empty], group=ctx.model_group, async_op=False
                )
                grad_output_augment = torch.cat((recv if left_to_right > 0 else empty, grad_output), dim=-1)

        # print(f'rank {rank}, grad_output_augment {grad_output_augment}')
        g_x_aug = conv1d_input( 
            dummy.shape, # total x shape in # TODO: FIGURE OUT TOTAL X SHAPE 
            weight,
            grad_output_augment,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )
        # TODO: figure out correct crop
        g_x = g_x_aug[..., :2] if (rank % 2 == 0) else g_x_aug[..., 1:-1]
        print(f'local grad output: gx {g_x}')
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
        n_parallel=2,
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
            self.in_channels,
            self.out_channels // 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
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
            x,
            self.conv1d.weight,
            self.conv1d.bias,
            self.in_channels,
            self.stride,
            self.padding,
            self.dilation,
            self.rank,
            self.device,
            self.n_parallel,
            self.model_group,
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
        stride, padding, dilation = (ctx.stride, ctx.padding, ctx.dilation)
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
        # TODO: check summation for vmapped case
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

        # dist.all_reduce(g_x, op=dist.ReduceOp.AVG, group=model_group)

        return g_x, g_w, g_b, None, None, None, None, None, None, None, None


class DistributedConv1D_complete(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        rank,
        spatial_group,
        channel_group,
        device,
        n_parallel,
        bias=False
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
        self.groups = groups
        self.rank = rank
        self.spatial_group = spatial_group
        self.channel_group = channel_group
        self.device = device
        self.n_parallel = n_parallel

        self.conv1d = torch.nn.Conv1d(
            self.in_channels,
            self.out_channels // 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias
        )

        self.conv1d_apply = Conv1d_complete().apply

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
            x,
            self.conv1d.weight,
            self.conv1d.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.in_channels,
            self.rank,
            self.spatial_group,
            self.channel_group,
            self.device,
            self.n_parallel
        )

        return Cij


class Conv1d_complete(torch.autograd.Function):
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
        in_channels: int,
        rank: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        n_parallel: int,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.spatial_group = spatial_group
        ctx.channel_group = channel_group
        ctx.b = bias
        y, x0, x1, L_local = conv1d_fourway_complete(
            x,
            weight,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            in_channels=in_channels,
            rank=rank,
            spatial_group=spatial_group,
            channel_group=channel_group
        )
        
        # print(f'I am rank {rank} and data has shape {x0.shape}, {x1.shape}')
        
        x_aug = torch.cat([x0, x1], dim=1)
        
        ctx.save_for_backward(x_aug, weight, torch.tensor([L_local], device=x.device))
        ctx.stride, ctx.padding, ctx.dilation = (
            stride,
            padding,
            dilation
        )
        ctx.rank = rank
        ctx.spatial_group = spatial_group
        ctx.channel_group = channel_group
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
        
        stride, padding, dilation = (
            ctx.stride,
            ctx.padding,
            ctx.dilation
        )
        rank, spatial_group, channel_group = ctx.rank, ctx.spatial_group, ctx.channel_group
        # print(f'I am rank {rank} with weights {weight}')
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
        # make grads identical on all ranks
        dist.all_reduce(g_w, op=dist.ReduceOp.AVG, group=spatial_group)
        # print(f'rank {rank}, g_w: {g_w}')
        if g_b is not None:
            dist.all_reduce(g_b, op=dist.ReduceOp.AVG, group=spatial_group)

        c0 = x_aug.shape[1]//2
        
        match rank % 4:
            case 0:
                g_x = g_x_aug[..., :c0, :L_local]
            case 1:
                g_x = g_x_aug[..., c0:, :L_local]
            case 2:
                g_x = g_x_aug[..., :c0, -L_local:]
            case 3:
                g_x = g_x_aug[..., c0:, -L_local:]
                

        return g_x, g_w, g_b, None, None, None, None, None, None, None, None, None, None


class DistributedConv2D_channel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        rank,
        model_group,
        device,
        n_parallel=2,
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
        self.groups = groups
        self.rank = rank
        self.model_group = model_group
        self.device = device
        self.n_parallel = 2

        self.conv1d = torch.nn.Conv2d(
            self.in_channels,
            self.out_channels // 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

        self.conv2d_apply = Conv2d_channel().apply

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
            x,
            self.conv1d.weight,
            self.conv1d.bias,
            self.in_channels,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.rank,
            self.device,
            self.n_parallel,
            self.model_group,
        )

        return Cij


class Conv2d_channel(torch.autograd.Function):
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
        groups: int = 1,
        rank: int = 0,
        device: torch.device = "cuda",
        n_parallel: int = 2,
        model_group: dist.ProcessGroup = None,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.model_group = model_group
        ctx.b = bias

        y, x0, x1 = conv2d_twoway_channel(
            x, weight, in_channels, stride, padding, dilation, groups, rank, model_group
        )
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
        stride, padding, dilation = (ctx.stride, ctx.padding, ctx.dilation)
        rank, model_group = ctx.rank, ctx.model_group

        # parameter grads from native conv1d gradient helpers
        g_w = conv2d_weight(
            x,
            weight.shape,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )
        g_b = grad_output.sum(dim=(0, 2, 3)) if ctx.has_bias else None

        # input grad then crop back to local slice
        g_x = conv2d_input(
            x.shape,
            weight,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )

        dist.all_reduce(g_x, op=dist.ReduceOp.AVG, group=model_group)

        return g_x, g_w, g_b, None, None, None, None, None, None, None, None, None


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
        n_parallel,
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
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
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
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.rank,
            self.device,
            self.n_parallel,
            self.model_group,
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

        y = F.conv2d(x_aug, weight, bias, stride=stride, padding=padding)

        ctx.save_for_backward(x_aug, weight, torch.tensor([H_local, W_local]))

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
