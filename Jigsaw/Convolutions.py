from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from Jigsaw.distributed_convolutions import (
    conv1d_fourway_complete,
    convTranspose1d_fourway_complete,
    conv1d_twoway,
    conv1d_twoway_channel,
    conv2d_fourway,
    conv2d_twoway,
    conv2d_twoway_channel,
)
import Jigsaw.distributed_convolutions as dist_conv
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
        bias=False,
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
            bias=self.bias,
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

        x_aug, L_local, right_to_left, left_to_right = conv1d_twoway(
            x,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            rank=rank,
            model_group=model_group,
        )
        y = F.conv1d(
            x_aug, weight, bias, stride=stride, padding=padding, dilation=dilation
        )

        ctx.save_for_backward(x_aug, weight)

        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = (
            stride,
            padding,
            dilation,
            model_group,
        )
        ctx.x_shape_padded = x_aug.shape
        ctx.x_shape = x.shape

        ctx.right_to_left, ctx.left_to_right = right_to_left, left_to_right

        ctx.rank, ctx.model_group = rank, model_group
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
        x_aug, weight = ctx.saved_tensors
        stride, padding, dilation, groups = (
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
        )
        rank, model_group = ctx.rank, ctx.model_group
        # TODO: how to acess the total shape of the convolution?
        print(f"x aug shape {x_aug.shape}")
        # parameter grads from native conv1d gradient helpers
        g_w = conv1d_weight(
            x_aug,
            weight.shape,
            grad_output,  # * ctx.local_length / ctx.total_length.item(),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )

        # print(f'rank {rank}, gw {g_w}, local length {ctx.local_length}, total length {ctx.total_length}')
        # TODO: same weighing factor needed for bias term
        g_b = grad_output.sum(dim=(0, 2)) if ctx.has_bias else None

        # make grads identical on all ranks
        dist.all_reduce(g_w, op=dist.ReduceOp.SUM, group=model_group)
        if g_b is not None:
            dist.all_reduce(g_b, op=dist.ReduceOp.SUM, group=model_group)

        # input grad then crop back to local slice
        right_to_left = ctx.right_to_left
        left_to_right = ctx.left_to_right

        empty = torch.empty(0, device=grad_output.device, dtype=grad_output.dtype)
        device = ctx.device
        dtype = grad_output.dtype

        # if rank in [2, 3]:
        #     conv_shape = (*ctx.x_shape[:-1], ctx.x_shape[-1] + left_to_right)
        # else:
        x_shape_padded, x_shape = ctx.x_shape_padded, ctx.x_shape
        conv_shape = (*ctx.x_shape_padded[:-1], ctx.x_shape_padded[-1] + right_to_left)

        # print(f'rank:{rank} {conv_shape}')
        g_x_aug = conv1d_input(
            conv_shape,
            weight,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )

        # TODO: figure out correct crop

        left_to_right, right_to_left = (
            right_to_left,
            left_to_right,
        )  # SWAP? # g_x_aug.shape[-1] - ctx.x_shape[-1]

        # print(f'l2r {left_to_right}, r2l {right_to_left}')
        match rank % 2:
            case 0:
                send = g_x_aug[..., -left_to_right:] if left_to_right > 0 else empty
                recv = (
                    torch.zeros(
                        (*g_x_aug.shape[:-1], right_to_left), device=device, dtype=dtype
                    )
                    if right_to_left > 0
                    else empty
                )
                send = send.contiguous()
                dist.all_to_all(
                    [empty, recv], [empty, send], group=ctx.model_group, async_op=False
                )
                recv = recv if right_to_left > 0 else empty
            case 1:
                send = g_x_aug[..., :right_to_left] if right_to_left > 0 else empty
                recv = (
                    torch.zeros(
                        (*g_x_aug.shape[:-1], left_to_right), device=device, dtype=dtype
                    )
                    if left_to_right > 0
                    else empty
                )
                send = send.contiguous()
                dist.all_to_all(
                    [recv, empty], [send, empty], group=ctx.model_group, async_op=False
                )
                recv = recv if left_to_right > 0 else empty

        if rank % 2 == 0:
            g_x = g_x_aug[..., : ctx.x_shape[-1]]
            if right_to_left > 0:
                g_x[..., -right_to_left:] = g_x[..., -right_to_left:] + recv
        else:
            g_x = g_x_aug[..., -ctx.x_shape[-1] :]
            print(f"rank {rank}. gx shape {g_x.shape}, ctx shape {ctx.x_shape}")
            if left_to_right > 0:
                g_x[..., :left_to_right] = (
                    g_x[..., :left_to_right] + recv
                )  # how to only add to single number?
            # print(f'first elemnts? {recv}')

        print(
            f"Rank {rank} x shape in {ctx.x_shape}, gxaug {g_x} r2l {right_to_left} l2r {left_to_right}"
        )
        return g_x, g_w, g_b, None, None, None, None, None, None, None, None


class DistributedConvTranspose1D(torch.nn.Module):
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
        bias=False,
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

        self.convTranspose1d = torch.nn.ConvTranspose1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=self.bias,
        )

        self.conv1d_apply = ConvTranspose1d().apply

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
            self.convTranspose1d.weight,
            self.convTranspose1d.bias,
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


class ConvTranspose1d(torch.autograd.Function):
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

        y = F.conv_transpose1d(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation
        )
        # print(f'y shape {y.shape}, x shape {x.shape}')

        y_buffer, overlap_size, local_size_left, local_size_right = (
            dist_conv.conv1dTranspose_halocalc_twoway(
                x,
                y,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                rank=rank,
                model_group=model_group,
            )
        )

        right_to_left = y.shape[-1] - local_size_left
        left_to_right = y.shape[-1] - local_size_right

        print(f"r2l {right_to_left}, l2r {left_to_right}")

        if rank % 2 == 0:  # LHS
            start_overlap_index = y.shape[-1] - overlap_size
            y[..., start_overlap_index:] = y[..., start_overlap_index:] + y_buffer
            y = y[..., :local_size_left]
        elif rank % 2 == 1:
            y[..., :overlap_size] = y[..., :overlap_size] + y_buffer
            y = y[..., -local_size_right:]

        ctx.save_for_backward(x, weight)

        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = (
            stride,
            padding,
            dilation,
            model_group,
        )
        ctx.x_shape_padded = x.shape
        ctx.x_shape = x.shape

        ctx.left_to_right, ctx.right_to_left = left_to_right, right_to_left

        ctx.rank, ctx.model_group = rank, model_group
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
        stride, padding, dilation, groups = (
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
        )
        rank, model_group = ctx.rank, ctx.model_group
        # TODO: how to acess the total shape of the convolution?

        empty = torch.empty(0, device=grad_output.device, dtype=grad_output.dtype)

        halo, _ = dist_conv.communicate_halos(
            grad_output,
            ctx.right_to_left,
            ctx.left_to_right,
            rank,
            2,
            empty,
            model_group,
            False,
        )

        if rank % 2 == 0:
            grad_output = torch.cat([grad_output, halo], dim=-1)
        else:
            grad_output = torch.cat([halo, grad_output], dim=-1)

        # parameter grads from native conv1d gradient helpers
        g_x, g_w, _ = torch.ops.aten.convolution_backward(
            grad_output,
            x,
            weight,
            None,
            [stride],
            [padding],
            [dilation],
            True,
            [0],
            groups=1,
            output_mask=[True, True, False],
        )

        # TODO: same weighing factor needed for bias term
        g_b = grad_output.sum(dim=(0, 2)) if ctx.has_bias else None

        # make grads identical on all ranks
        dist.all_reduce(g_w, op=dist.ReduceOp.SUM, group=model_group)
        if g_b is not None:
            dist.all_reduce(g_b, op=dist.ReduceOp.SUM, group=model_group)

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
        bias=False,
        model_group=None,
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
        self.model_group = model_group

        self.conv1d = torch.nn.Conv1d(
            self.in_channels,
            self.out_channels // 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=bias,
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

        y, x0, x1 = conv1d_twoway_channel(
            x,
            weight,
            in_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            rank=rank,
            model_group=model_group,
        )
        x = torch.cat(
            [x0, x1], dim=1
        )  # TODO: may need to adjust concat dimension when using torch_blue
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
        print(
            f"x shape {x.shape}, weight shape {weight.shape}, grad output shape {grad_output.shape}, stride padding, {stride}, {padding}"
        )
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

        dist.all_reduce(g_x, op=dist.ReduceOp.SUM, group=model_group)
        n_channels_out = g_x.shape[1] // 2

        match rank % 2:
            case 0:
                g_x = g_x[:, :n_channels_out]
            case 1:
                g_x = g_x[:, n_channels_out:]
        return g_x, g_w, g_b, None, None, None, None, None, None, None, None


class DistributedConvTranspose1D_channel(torch.nn.Module):
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
        bias=False,
        model_group=None,
    ):
        """
        Distributed 1D Transpose Convolution.

        input shape of forward pass: (N, Cin, Lin) or (Cin, Lin)
        output shape of forward pass: (N, Cout, Lout)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.rank = rank
        self.model_group = groups
        self.device = device
        self.n_parallel = 2
        self.model_group = model_group

        print(f'I am rank {rank} with in channels {self.in_channels} and out channels {self.out_channels}')
        self.conv1d = torch.nn.ConvTranspose1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=bias,
        )

        self.convtranspose1d_apply = ConvTranspose1d_channel().apply

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
        Cij = self.convtranspose1d_apply(
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


class ConvTranspose1d_channel(torch.autograd.Function):
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

        y, x0, x1 = dist_conv.convTranspose1d_twoway_channel(
            x,
            weight,
            in_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            rank=rank,
            model_group=model_group,
        )
        
        x = torch.cat(
            [x0, x1], dim=1
        )  # TODO: may need to adjust concat dimension when using torch_blue
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
        print(
            f"x shape {x.shape}, weight shape {weight.shape}, grad output shape {grad_output.shape}, stride padding, {stride}, {padding}"
        )
        # parameter grads from native conv1d gradient helpers
        g_x, g_w, _ = torch.ops.aten.convolution_backward(
            grad_output,
            x,
            weight,
            None,
            [stride],
            [padding],
            [dilation],
            True,
            [0],
            groups=1,
            output_mask=[True, True, False],
        )
        # TODO: check summation for vmapped case
        g_b = grad_output.sum(dim=(0, 2)) if ctx.has_bias else None
        
        # input grad then crop back to local slice

        dist.all_reduce(g_x, op=dist.ReduceOp.SUM, group=model_group)
        print(f'Rank {rank} gw {g_w}')
        n_channels_out = g_x.shape[1] // 2  # in_channels or out?

        match rank % 2:
            case 0:
                g_x = g_x[:, :n_channels_out]
            case 1:
                g_x = g_x[:, n_channels_out:]
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
        bias=False,
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

        assert self.in_channels % 2 == 0, (
            "number of input channels is not divisible by 2"
        )
        assert self.out_channels % 2 == 0, (
            "number of output channels is not divisible by 2"
        )

        self.conv1d = torch.nn.Conv1d(
            self.in_channels,
            self.out_channels // 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias,
        )
        self.empty = torch.empty(0, device=device, dtype=torch.float32)

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
            self.n_parallel,
            self.empty,
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
        empty: torch.Tensor,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.spatial_group = spatial_group
        ctx.channel_group = channel_group
        ctx.in_channels_global = in_channels
        ctx.b = bias
        ctx.empty = empty  # pre-initialized empty tensor

        y, x0, x1, right_to_left, left_to_right = conv1d_fourway_complete(
            x,
            weight,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            in_channels=in_channels,
            rank=rank,
            spatial_group=spatial_group,
            channel_group=channel_group,
            empty=ctx.empty,
        )

        ctx.right_to_left, ctx.left_to_right = right_to_left, left_to_right

        x_aug = torch.cat([x0, x1], dim=1)

        ctx.x_shape_padded = x_aug.shape  # TODO: is this the correct
        ctx.x_shape = x.shape

        ctx.save_for_backward(x_aug, weight)
        ctx.stride, ctx.padding, ctx.dilation = (stride, padding, dilation)
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
        x_aug, weight = ctx.saved_tensors
        empty = ctx.empty
        stride, padding, dilation = (ctx.stride, ctx.padding, ctx.dilation)
        local_rank, spatial_group, channel_group = (
            ctx.rank % 4,  # TODO: verify with communicator group
            ctx.spatial_group,
            ctx.channel_group,
        )

        right_to_left, left_to_right = ctx.right_to_left, ctx.left_to_right
        x_shape_padded, x_shape = ctx.x_shape_padded, ctx.x_shape

        # print(f'rank:{rank} {conv_shape}')
        g_x_aug = conv1d_input(
            x_shape_padded,
            weight,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )

        left_to_right, right_to_left = (
            right_to_left,
            left_to_right,
        )  # SWAP?

        # g_x calculation
        halo, wait_handle = dist_conv.communicate_halos(
            g_x_aug,
            right_to_left,
            left_to_right,
            local_rank,
            4,
            empty,
            spatial_group,
            async_op=True,
        )

        wait_handle.wait()

        if local_rank % 4 in [0, 1]:
            g_x = g_x_aug[..., : x_shape[-1]]
            if right_to_left > 0:
                g_x[..., -right_to_left:] = g_x[..., -right_to_left:] + halo
        else:  # right half
            g_x = g_x_aug[..., -x_shape[-1] :]
            # print(f'rank {rank}. gx shape {g_x.shape}, ctx shape {ctx.x_shape}')
            if left_to_right > 0:
                g_x[..., :left_to_right] = g_x[..., :left_to_right] + halo

        # async here because we assume the size of the input data will be much larger than the convolution weights,
        # i.e., advantageous to perform the g_x communication before calculating conv1d_weight

        g_x_buffer = torch.zeros(
            (*g_x.shape[:-2], g_x.shape[-2] // 2, g_x.shape[-1]),
            device=g_x.device,
            dtype=g_x.dtype,
        )
        g_x_recv, g_x_handle = dist_conv.send_g_x_channels(
            g_x,
            g_x_buffer,
            empty,
            ctx.in_channels_global,
            channel_group,
            local_rank,
            async_op=True,
        )

        # parameter grads from native conv1d gradient helpers
        # x_aug should have the local shape
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
        g_w_handle = dist.all_reduce(
            g_w, op=dist.ReduceOp.SUM, group=spatial_group, async_op=True
        )
        if g_b is not None:
            dist.all_reduce(g_b, op=dist.ReduceOp.AVG, group=spatial_group)

        if g_x_handle is not None:
            g_x_handle.wait()
        # dist.all_reduce(g_x, op=dist.ReduceOp.SUM, group=channel_group)

        g_x = (
            g_x[..., : ctx.in_channels_global // 2, :] + g_x_recv
            if local_rank % 2 == 0
            else g_x[..., ctx.in_channels_global // 2 :, :] + g_x_recv
        )
        # print(f'Local rank {local_rank}, gx {g_x}, gx recv {g_x_buffer}')
        g_w_handle.wait()
        return (
            g_x,
            g_w,
            g_b,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DistributedConvTranspose1D_complete(torch.nn.Module):
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
        bias=False,
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

        assert self.in_channels % 2 == 0, (
            "number of input channels is not divisible by 2"
        )
        assert self.out_channels % 2 == 0, (
            "number of output channels is not divisible by 2"
        )

        self.convTranspose1d = torch.nn.ConvTranspose1d(
            self.in_channels // 2,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias,
        )
        self.empty = torch.empty(0, device=device, dtype=torch.float32)

        self.convTranspose1d_apply = ConvTranspose1d_complete().apply

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
        Cij = self.convTranspose1d_apply(
            x,
            self.convTranspose1d.weight,
            self.convTranspose1d.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.in_channels,
            self.rank,
            self.spatial_group,
            self.channel_group,
            self.device,
            self.n_parallel,
            self.empty,
        )

        return Cij


class ConvTranspose1d_complete(torch.autograd.Function):
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
        empty: torch.Tensor,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.spatial_group = spatial_group
        ctx.channel_group = channel_group
        ctx.in_channels_global = in_channels
        ctx.b = bias
        ctx.empty = empty  # pre-initialized empty tensor

        y, x0, x1, right_to_left, left_to_right = convTranspose1d_fourway_complete(
            x,
            weight,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            in_channels=in_channels,
            rank=rank,
            spatial_group=spatial_group,
            channel_group=channel_group,
            empty=ctx.empty,
        )

        ctx.right_to_left, ctx.left_to_right = right_to_left, left_to_right

        x_aug = torch.cat([x0, x1], dim=1)

        ctx.x_shape_padded = x_aug.shape  # TODO: is this the correct
        ctx.x_shape = x.shape

        ctx.save_for_backward(x_aug, weight)
        ctx.stride, ctx.padding, ctx.dilation = (stride, padding, dilation)
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
        x_aug, weight = ctx.saved_tensors
        empty = ctx.empty
        stride, padding, dilation = (ctx.stride, ctx.padding, ctx.dilation)
        local_rank, spatial_group, channel_group = (
            ctx.rank % 4,  # TODO: verify with communicator group
            ctx.spatial_group,
            ctx.channel_group,
        )

        right_to_left, left_to_right = ctx.right_to_left, ctx.left_to_right
        x_shape_padded, x_shape = ctx.x_shape_padded, ctx.x_shape

        # print(f'rank:{rank} {conv_shape}')
        print(
            f"x_shape_padded vs x_shape{grad_output.shape} {x_shape_padded}, {x_shape}"
        )
        g_x_aug = torch.ops.aten.convolution_backward(
            grad_output,
            grad_output.new_empty(1).expand(x_shape),
            weight,
            None,
            [stride],
            [padding],
            [dilation],
            True,
            [0],
            1,
            (True, False, False),
        )[0]
        # conv1d_input(
        # x_shape_padded,
        # weight,
        # grad_output,
        # stride=stride,
        # padding=padding,
        # dilation=dilation,
        # groups=1,
        # )

        left_to_right, right_to_left = (
            right_to_left,
            left_to_right,
        )  # SWAP?

        # g_x calculation
        halo, wait_handle = dist_conv.communicate_halos(
            g_x_aug,
            right_to_left,
            left_to_right,
            local_rank,
            4,
            empty,
            spatial_group,
            async_op=True,
        )

        wait_handle.wait()

        if local_rank % 4 in [0, 1]:
            g_x = g_x_aug[..., : x_shape[-1]]
            if right_to_left > 0:
                g_x[..., -right_to_left:] = g_x[..., -right_to_left:] + halo
        else:  # right half
            g_x = g_x_aug[..., -x_shape[-1] :]
            # print(f'rank {rank}. gx shape {g_x.shape}, ctx shape {ctx.x_shape}')
            if left_to_right > 0:
                g_x[..., :left_to_right] = g_x[..., :left_to_right] + halo

        # async here because we assume the size of the input data will be much larger than the convolution weights,
        # i.e., advantageous to perform the g_x communication before calculating conv1d_weight

        g_x_buffer = torch.zeros(
            (*g_x.shape[:-2], g_x.shape[-2] // 2, g_x.shape[-1]),
            device=g_x.device,
            dtype=g_x.dtype,
        )
        g_x_recv, g_x_handle = dist_conv.send_g_x_channels(
            g_x,
            g_x_buffer,
            empty,
            ctx.in_channels_global,
            channel_group,
            local_rank,
            async_op=True,
        )

        # parameter grads from native conv1d gradient helpers
        # x_aug should have the local shape
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
        g_w_handle = dist.all_reduce(
            g_w, op=dist.ReduceOp.SUM, group=spatial_group, async_op=True
        )
        if g_b is not None:
            dist.all_reduce(g_b, op=dist.ReduceOp.AVG, group=spatial_group)

        if g_x_handle is not None:
            g_x_handle.wait()
        # dist.all_reduce(g_x, op=dist.ReduceOp.SUM, group=channel_group)

        g_x = (
            g_x[..., : ctx.in_channels_global // 2, :] + g_x_recv
            if local_rank % 2 == 0
            else g_x[..., ctx.in_channels_global // 2 :, :] + g_x_recv
        )
        # print(f'Local rank {local_rank}, gx {g_x}, gx recv {g_x_buffer}')
        g_w_handle.wait()
        return (
            g_x,
            g_w,
            g_b,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
        groups,
        rank,
        spatial_group,
        channel_group,
        device,
        n_parallel,
        bias=False,
    ):
        """
        Distributed 1D Convolution.

        random seed is required because of initialization/synchronization between "partner processes"

        input shape of forward pass: (N, Cin, Lin) or (Cin, Lin)
        output shape of forward pass: (N, Cout, Lout)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels // 2

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.rank = rank
        self.spatial_group = spatial_group
        self.channel_group = channel_group
        self.device = device
        self.n_parallel = n_parallel
        self.empty = torch.empty(0, device=device, dtype=torch.float32)
        self.conv2d = torch.nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=groups,
            bias=bias,
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
            self.in_channels,
            self.rank,
            self.spatial_group,
            self.channel_group,
            self.device,
            self.n_parallel,
            self.empty,
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
        in_channels: int,
        rank: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        n_parallel: int,
        empty: torch.Tensor,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.spatial_group = spatial_group
        ctx.channel_group = channel_group
        ctx.in_channels_global = in_channels
        ctx.b = bias
        ctx.empty = empty  # pre-initialized empty tensor

        y, x0, x1, right_to_left, left_to_right = dist_conv.conv2d_fourway_complete(
            x,
            weight,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            in_channels=in_channels,
            rank=rank,
            spatial_group=spatial_group,
            channel_group=channel_group,
            empty=ctx.empty,
        )
        print(f"Rank!!! {rank}, y result!! {y}")
        ctx.right_to_left, ctx.left_to_right = right_to_left, left_to_right

        x_aug = torch.cat(
            [x0, x1], dim=1
        )  # TODO is this concatenating along the correct dimension?

        ctx.x_shape_padded = x_aug.shape  # TODO: is this the correct
        ctx.x_shape = x.shape

        ctx.save_for_backward(x_aug, weight)
        ctx.stride, ctx.padding, ctx.dilation = (stride, padding, dilation)
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
        x_aug, weight = ctx.saved_tensors
        empty = ctx.empty
        stride, padding, dilation = (ctx.stride, ctx.padding, ctx.dilation)
        local_rank, spatial_group, channel_group = (
            ctx.rank % 4,  # TODO: verify with communicator group
            ctx.spatial_group,
            ctx.channel_group,
        )

        right_to_left, left_to_right = ctx.right_to_left, ctx.left_to_right
        x_shape_padded, x_shape = ctx.x_shape_padded, ctx.x_shape

        # print(f'rank:{rank} {conv_shape}')
        g_x_aug = conv2d_input(
            x_shape_padded,
            weight,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )

        left_to_right, right_to_left = (
            right_to_left,
            left_to_right,
        )  # SWAP?

        # g_x calculation
        halo, wait_handle = dist_conv.communicate_halos(
            g_x_aug,
            right_to_left,
            left_to_right,
            local_rank,
            4,
            empty,
            spatial_group,
            async_op=True,
        )

        wait_handle.wait()

        if local_rank % 4 in [0, 1]:
            g_x = g_x_aug[..., : x_shape[-1]]
            if right_to_left > 0:
                g_x[..., -right_to_left:] = g_x[..., -right_to_left:] + halo
        else:  # right half
            g_x = g_x_aug[..., -x_shape[-1] :]
            # print(f'rank {rank}. gx shape {g_x.shape}, ctx shape {ctx.x_shape}')
            if left_to_right > 0:
                g_x[..., :left_to_right] = g_x[..., :left_to_right] + halo

        # async here because we assume the size of the input data will be much larger than the convolution weights,
        # i.e., advantageous to perform the g_x communication before calculating conv1d_weight

        g_x_buffer = torch.zeros(
            (*g_x.shape[:-3], g_x.shape[-3] // 2, *g_x.shape[-2:]),
            device=g_x.device,
            dtype=g_x.dtype,
        )
        g_x_recv, g_x_handle = dist_conv.send_g_x_channels_2d(
            g_x,
            g_x_buffer,
            empty,
            ctx.in_channels_global,
            channel_group,
            local_rank,
            async_op=True,
        )

        # parameter grads from native conv1d gradient helpers
        # x_aug should have the local shape
        g_w = conv2d_weight(
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
        g_w_handle = dist.all_reduce(
            g_w, op=dist.ReduceOp.SUM, group=spatial_group, async_op=True
        )
        if g_b is not None:
            dist.all_reduce(g_b, op=dist.ReduceOp.AVG, group=spatial_group)

        if g_x_handle is not None:
            g_x_handle.wait()
        # dist.all_reduce(g_x, op=dist.ReduceOp.SUM, group=channel_group)
        # print(f'Rankgx {g_x_recv}')
        g_x = (
            g_x[
                ...,
                : ctx.in_channels_global // 2,
                :,
                :,
            ]
            + g_x_recv
            if local_rank % 2 == 0
            else g_x[..., ctx.in_channels_global // 2 :, :, :] + g_x_recv
        )
        # print(f'Local rank {local_rank}, gx {g_x}, gx recv {g_x_buffer}')
        g_w_handle.wait()
        # if local_rank in [0, 1]:
        #     print(f'local rank {local_rank}, gw {g_w}')
        print(f"Rank {local_rank} and gw is {g_x}")
        return (
            g_x,
            g_w,
            g_b,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# class Conv2d(torch.autograd.Function):
#     """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

#     @staticmethod
#     def forward(
#         ctx: torch.autograd.Function,
#         x: torch.Tensor,
#         weight: torch.Tensor,
#         bias: torch.Tensor,
#         kernel_size: int,
#         stride: int,
#         padding: int,
#         dilation: int,
#         rank: int,
#         device: torch.device,
#         n_parallel: int,
#         model_group: dist.ProcessGroup,
#     ) -> torch.Tensor:
#         ctx.rank = rank
#         ctx.device = device
#         ctx.n_parallel = n_parallel
#         ctx.model_group = model_group
#         ctx.b = bias

#         if n_parallel == 2:
#             x_aug, H_local, W_local = conv2d_twoway(
#                 x,
#                 kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 dilation=dilation,
#                 rank=rank,
#                 model_group=model_group,
#             )
#         elif n_parallel == 4:
#             x_aug, H_local, W_local = conv2d_fourway(
#                 x,
#                 kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 dilation=dilation,
#                 rank=rank,
#                 model_group=model_group,
#             )

#         y = F.conv2d(x_aug, weight, bias, stride=stride, padding=padding)

#         ctx.save_for_backward(x_aug, weight, torch.tensor([H_local, W_local]))

#         ctx.stride, ctx.padding, ctx.dilation, ctx.groups = (
#             stride,
#             padding,
#             dilation,
#             model_group,
#         )
#         ctx.rank, ctx.group = rank, model_group
#         ctx.has_bias = bias is not None

#         return y

#     @staticmethod
#     def backward(
#         ctx: torch.autograd.Function, grad_output: torch.Tensor
#     ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None]:
#         """
#         Backward pass of linear layer (XWT) distributed across two processes.

#         Parameters
#         ----------
#         ctx : torch.autograd.Function
#             Context object to save information for backward computation.
#         grad_output : tensor
#             The gradient of the loss with respect to the output of the layer.

#         Returns
#         -------
#         grad_input: torch.Tensor
#             Gradient with respect to the input.
#         grad_weight: torch.Tensor
#             Gradient with respect to the weights.
#         grad_bias: torch.Tensor
#             Gradient with respect to the bias.
#         """

#         x_aug, weight, L_local_t = ctx.saved_tensors
#         stride, padding, dilation, groups = (
#             ctx.stride,
#             ctx.padding,
#             ctx.dilation,
#             ctx.groups,
#         )
#         rank, group = ctx.rank, ctx.group
#         H_local = int(L_local_t[0].item())
#         W_local = int(L_local_t[1].item())

#         # parameter grads from native conv1d gradient helpers
#         g_w = conv2d_weight(
#             x_aug,
#             weight.shape,
#             grad_output,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=1,
#         )

#         g_b = grad_output.sum(dim=(0, 2, 3)) if ctx.has_bias else None

#         # make grads identical on all ranks
#         dist.all_reduce(g_w, op=dist.ReduceOp.AVG, group=group)
#         if g_b is not None:
#             dist.all_reduce(g_b, op=dist.ReduceOp.AVG, group=group)

#         # input grad then crop back to local slice
#         g_x_aug = conv2d_input(
#             x_aug.shape,
#             weight,
#             grad_output,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=1,
#         )

#         match rank:
#             case 0:
#                 g_x = g_x_aug[..., :H_local, :W_local]
#             case 1:
#                 g_x = g_x_aug[..., :H_local, -W_local:]
#             case 2:
#                 g_x = g_x_aug[..., -H_local:, :W_local]
#             case 3:
#                 g_x = g_x_aug[..., -H_local:, -W_local:]

#         return g_x, g_w, g_b, None, None, None, None, None, None, None, None
