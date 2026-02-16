from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from .distributed_convolutions import (
    conv1d_fourway,
    conv1d_twoway,
    conv1d_twoway_channel,
    conv2d_fourway,
    conv2d_twoway,
    conv2d_twoway_channel,
)
import numpy as np
import Jigsaw.distributed_convolutions as dist_conv
from torch.nn.grad import conv1d_input, conv1d_weight, conv2d_input, conv2d_weight
from functools import partial

class ConvSelectionStrategy:
    def __init__(
        self,
        bayesian,  
        parallelism,
        bias
    ):
        self.conv2d_fn = self._conv2d_fn_selection(bayesian)
        self.conv2d_input_fn = self._conv2d_input_fn_selection(bayesian)
        self.conv2d_weight_fn = self._conv2d_weight_fn_selection(bayesian)
        self.weight_reshape = self._reshape_weights_selection(bayesian)
        self.grad_bias = self._grad_bias_selection(parallelism, bias)
    
    def _conv2d_fn_selection(self, bayesian):
        if bayesian:
            return torch.vmap(F.conv2d)
        else:
            return F.conv2d
    
    def _conv2d_input_fn_selection(self, bayesian):
        if bayesian:
            return torch.vmap(conv2d_input, in_dims=(None, 0, 0))
        else:
            return conv2d_input
    
    def _conv2d_weight_fn_selection(self, bayesian):
        if bayesian:
            return torch.vmap(conv2d_weight, in_dims=(0, None, 0))
        else:
            return conv2d_weight
        
    def _reshape_weights_selection(self, bayesian):
        if bayesian: 
            return self._reshape_weights
        else:
            return self._no_reshape_weights
        
    def _reshape_weights(self, w, x):
        w = w.expand(x.shape[0], *w.shape)
        return w
    
    def _no_reshape_weights(self, w, x=None):
        return w
    
    def _grad_bias_selection(self, parallelism, bias):
        if bias is None:
            return self._no_bias
        elif parallelism == 2:
            return self._no_reduce_grad_bias
        else:
            return self._reduce_bias_over_spatial_group
    
    def _no_bias(self, grad_output, spatial_group=None):
        return None
    
    def _no_reduce_grad_bias(self, grad_output, spatial_group=None):
        return grad_output.sum(dim=(-4, -2, -1))
    
    def _reduce_bias_over_spatial_group(self, grad_output, spatial_group):
        grad_b = grad_output.sum(dim=(-4, -2, -1))
        dist.all_reduce(grad_b, op=dist.ReduceOp.SUM, group=spatial_group)
        return grad_b
    

class ConvTransposeSelectionStrategy:
    def __init__(
        self,
        bayesian,  
        parallelism,
        bias
    ):
        self.convTranspose2d_fn = self._convTranspose2d_fn_selection(bayesian)
        self.convTranspose2d_grad_fn = self._convTranspose2d_grad_fn_selection(bayesian)
        self.weight_reshape = self._reshape_weights_selection(bayesian)
        self.grad_bias = self._grad_bias_selection(parallelism, bias)
    
    def _convTranspose2d_fn_selection(self, bayesian):
        if bayesian:
            return torch.vmap(F.conv_transpose2d)
        else:
            return F.conv_transpose2d
    
    def _convTranspose2d_grad_fn_selection(self, bayesian):
        if bayesian:
            return torch.vmap(
                torch.ops.aten.convolution_backward,
                in_dims=(0, 0, 0, None, None, None, None, None, None, None, None),
                out_dims=(0, 0, None),
            )
        else:
            return torch.ops.aten.convolution_backward
        
    def _reshape_weights_selection(self, bayesian):
        if bayesian: 
            return self._reshape_weights
        else:
            return self._no_reshape_weights
        
    def _reshape_weights(self, w, x):
        w = w.expand(x.shape[0], *w.shape)
        return w
    
    def _no_reshape_weights(self, w, x=None):
        return w
    
    def _grad_bias_selection(self, parallelism, bias):
        if bias is None:
            return self._no_bias
        elif parallelism == 2:
            return self._no_reduce_grad_bias
        else:
            return self._reduce_bias_over_spatial_group
    
    def _no_bias(self, grad_output, spatial_group=None):
        return None
    
    def _no_reduce_grad_bias(self, grad_output, spatial_group=None):
        return grad_output.sum(dim=(-4, -2, -1))
    
    def _reduce_bias_over_spatial_group(self, grad_output, spatial_group):
        grad_b = grad_output.sum(dim=(-4, -2, -1))
        dist.all_reduce(grad_b, op=dist.ReduceOp.SUM, group=spatial_group)
        return grad_b
    
    
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
        self.parallelism = 2
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
            self.parallelism,
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
        parallelism: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.parallelism = parallelism
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

        x_shape_padded, x_shape = ctx.x_shape_padded, ctx.x_shape
        conv_shape = (*ctx.x_shape_padded[:-1], ctx.x_shape_padded[-1] + right_to_left)

        g_x_aug = conv1d_input(
            conv_shape,
            weight,
            grad_output_augment,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )

        left_to_right, right_to_left = (
            right_to_left,
            left_to_right,
        )

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
        self.parallelism = 2
        self.bias = bias
        self.groups = groups

        self.convTranspose1d = torch.nn.ConvTranspose1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=self.bias,
            groups=self.groups
        )

        k = np.sqrt(self.groups / (self.out_channels * 2 * self.kernel_size))
        self.convTranspose1d.weight.data.uniform_(-k, k)
        if bias:
            self.convTranspose1d.bias.data.uniform_(-k, k)
            
        self.convTranspose1d_apply = ConvTranspose1d().apply

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
            self.rank,
            self.device,
            self.parallelism,
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
        parallelism: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.parallelism = parallelism
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
        parallelism=2,
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
        self.parallelism = 2
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
            self.parallelism,
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
        parallelism: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.parallelism = parallelism
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
        parallelism=2,
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
        self.parallelism = 2
        self.model_group = model_group

        print(
            f"I am rank {rank} with in channels {self.in_channels} and out channels {self.out_channels}"
        )
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
            self.parallelism,
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
        parallelism: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.parallelism = parallelism
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
        )  # TODO: will need to adjust concat dimension when using torch_blue
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
        print(f"Rank {rank} gw {g_w}")
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
        parallelism,
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
        self.parallelism = parallelism

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
            self.parallelism,
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
        parallelism: int,
        empty: torch.Tensor,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.parallelism = parallelism
        ctx.spatial_group = spatial_group
        ctx.channel_group = channel_group
        ctx.in_channels_global = in_channels
        ctx.b = bias
        ctx.empty = empty  # pre-initialized empty tensor

        y, x0, x1, right_to_left, left_to_right = conv1d_fourway(
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

        g_x = (
            g_x[..., : ctx.in_channels_global // 2, :] + g_x_recv
            if local_rank % 2 == 0
            else g_x[..., ctx.in_channels_global // 2 :, :] + g_x_recv
        )
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
        parallelism,
        bias=False,
    ):
        """
        Distributed 1D Convolution.

        random seed is required because of initialization/synchronization between "partner processes"

        input shape of forward pass: (N, Cin, Lin) or (Cin, Lin)
        output shape of forward pass: (N, Cout, Lout)
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.rank = rank
        self.spatial_group = spatial_group
        self.channel_group = channel_group
        self.device = device
        self.parallelism = parallelism

        assert in_channels % 2 == 0, "number of input channels is not divisible by 2"
        assert out_channels % 2 == 0, "number of output channels is not divisible by 2"
        self.in_channels = in_channels
        self.out_channels = out_channels // 2

        self.convTranspose1d = torch.nn.ConvTranspose1d(
            self.in_channels,
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
            self.parallelism,
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
        parallelism: int,
        empty: torch.Tensor,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.parallelism = parallelism
        ctx.spatial_group = spatial_group
        ctx.channel_group = channel_group
        ctx.in_channels_global = in_channels
        ctx.b = bias
        ctx.empty = empty  # pre-initialized empty tensor

        y, x0, x1 = dist_conv.convTranspose1d_twoway_channel(
            x,
            weight,
            in_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            rank=rank,
            model_group=channel_group,
        )

        y_buffer, overlap_size, local_size_left, local_size_right = (
            dist_conv.conv1dTranspose_halocalc_twoway(
                x,
                y,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                local_rank=rank,
                spatial_group=spatial_group,
                parallelism=4,
            )
        )

        right_to_left = y.shape[-1] - local_size_left
        left_to_right = y.shape[-1] - local_size_right

        if rank // 2 == 0:  # LHS
            start_overlap_index = y.shape[-1] - overlap_size
            y[..., start_overlap_index:] = y[..., start_overlap_index:] + y_buffer
            y = y[..., :local_size_left]
        elif rank // 2 == 1:
            y[..., :overlap_size] = y[..., :overlap_size] + y_buffer
            y = y[..., -local_size_right:]

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

        # print(f'rank:{rank} {conv_shape}')
        empty = torch.empty(0, device=grad_output.device, dtype=grad_output.dtype)

        halo, _ = dist_conv.communicate_halos(
            grad_output,
            ctx.right_to_left,
            ctx.left_to_right,
            local_rank,
            4,
            empty,
            spatial_group=spatial_group,
            async_op=False,
        )

        if local_rank // 2 == 0:  # left and right groups
            grad_output = torch.cat([grad_output, halo], dim=-1)
        else:
            grad_output = torch.cat([halo, grad_output], dim=-1)

        g_x, g_w, _ = torch.ops.aten.convolution_backward(
            grad_output,
            x_aug,
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

        g_w_handle.wait()
        # print(f'Local rank {local_rank}, gx {g_w}')
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
        parallelism=2,
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
        self.parallelism = 2

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
            self.parallelism,
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
        parallelism: int = 2,
        model_group: dist.ProcessGroup = None,
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.device = device
        ctx.parallelism = parallelism
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


def _average_weights(
    model: torch.nn.Module, spatial_group: dist.ProcessGroup, device: torch.device
):
    for name, param in model.named_parameters():
        dist.all_reduce(param, group=spatial_group, op=dist.ReduceOp.AVG)


class DistributedConv2D(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        rank=0,
        spatial_group=None,
        channel_group=None,
        device=None,
        parallelism=4,
        bayesian=False
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
        self.parallelism = parallelism
        self.empty = torch.empty(0, device=device, dtype=torch.float32)
        self.bayesian = bayesian
        # # TODO
        # self.conv2d.weights = torch.nn.Parameter(self.out_channels, self.in_channels//1, kernel_size[0], kernel_size[1])
        # self.conv2d.bias = torch.nn.Parameter(out_channels)
        
        
        self.conv2d = torch.nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=groups,
            bias=bias,
        ).to(device)
        
        self.selector = ConvSelectionStrategy(self.bayesian, self.parallelism, self.conv2d.bias)
        
        _average_weights(self, spatial_group, device)
        
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
            self.parallelism,
            self.empty,
            self.bayesian,
            self.selector
        )

        return Cij[0]


class Conv2d(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
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
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        selector
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        
        # Compute the forward pass        
        y, x0, x1, right_to_left, left_to_right = dist_conv.conv2d_fourway(
            x,
            weight,
            bias,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            in_channels=in_channels,
            rank=rank,
            spatial_group=spatial_group,
            channel_group=channel_group,
            empty=empty,
            conv2d_fn=selector.conv2d_fn
        )

        x_aug = torch.cat([x0, x1], dim=-3) # concatenate along the channel dimension
        halo_sizes = torch.tensor([right_to_left, left_to_right]) # save halo sizes in tensor
        
        # y is result, rest of the outputs are required for the backward pass
        return y, x_aug, x, halo_sizes

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Setup context for backward pass.

        Parameters
        ----------
        ctx : context object
            Context to save information for backward.
        inputs : tuple
            All inputs passed to forward (x, weight, bias, kernel_size, ...).
        output : tuple
            All outputs returned from forward (y, x_aug, x, right_to_left, left_to_right).
        """
        # Unpack inputs
        (
            x,
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            dilation,
            in_channels,
            rank,
            spatial_group,
            channel_group,
            device,
            parallelism,
            empty,
            bayesian,
            selector
        ) = inputs

        # Unpack outputs
        _, x_aug, x_orig, halo_sizes = output

        # Mark non-differentiable outputs (indices 1-4)
        ctx.mark_non_differentiable(x_aug)
        ctx.mark_non_differentiable(x_orig)
        ctx.mark_non_differentiable(halo_sizes)
        
        ctx.halo_sizes = halo_sizes

        # Save everything needed for backward
        ctx.save_for_backward(x_aug, weight)

        # Save non-tensor attributes
        ctx.rank = rank
        ctx.device = device
        ctx.parallelism = parallelism
        ctx.spatial_group = spatial_group
        ctx.channel_group = channel_group
        ctx.in_channels_global = in_channels
        ctx.b = bias
        ctx.empty = empty
        ctx.x_shape_padded = x_aug.shape
        ctx.x_shape = x_orig.shape
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.bayesian = bayesian 
        ctx.selector = selector
        
    @staticmethod
    def backward(
        ctx: torch.autograd.Function,
        grad_output: torch.Tensor,
        grad_x_aug: None,  # No gradient for x_aug
        grad_x_orig: None,  # No gradient for x_orig
        halo_sizes: None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
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
        None,
        None
    ]:
        """
        Backward pass of linear layer (XWT) distributed across two processes.
        """
        x_aug, weight = ctx.saved_tensors
        empty = ctx.empty
        stride, padding, dilation = (ctx.stride, ctx.padding, ctx.dilation)
        local_rank, spatial_group, channel_group = (
            ctx.rank % 4,
            ctx.spatial_group,
            ctx.channel_group,
        )
        selector = ctx.selector
        
        right_to_left, left_to_right = ctx.halo_sizes[0], ctx.halo_sizes[1]
        x_shape_padded, x_shape = ctx.x_shape_padded, ctx.x_shape

        g_b = selector.grad_bias(grad_output, ctx.spatial_group)
        
        g_x_aug = selector.conv2d_input_fn(
            x_shape_padded[int(ctx.bayesian):],
            weight,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        left_to_right, right_to_left = (right_to_left, left_to_right)

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
        else:
            g_x = g_x_aug[..., -x_shape[-1] :]
            if left_to_right > 0:
                g_x[..., :left_to_right] = g_x[..., :left_to_right] + halo

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

        g_w = selector.conv2d_weight_fn(
            x_aug,
            weight.shape[int(ctx.bayesian):],
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        
        g_w_handle = dist.all_reduce(
            g_w, op=dist.ReduceOp.SUM, group=spatial_group, async_op=True
        )
        
        if g_x_handle is not None:
            g_x_handle.wait()

        g_x = (
            g_x[..., : ctx.in_channels_global // 2, :, :] + g_x_recv
            if local_rank % 2 == 0
            else g_x[..., ctx.in_channels_global // 2 :, :, :] + g_x_recv
        )

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
            None,
            None,
        )

    @staticmethod
    def vmap(
        info,
        in_dims,
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
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        selector
    ) -> tuple[torch.Tensor, Optional[int]]:
        
        out = Conv2d.apply(
            x,
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            dilation,
            in_channels,
            rank,
            spatial_group,
            channel_group,
            device,
            parallelism,
            empty,
            bayesian,
            selector
        )
        
        return out, 0


class DistributedConvTranspose2D(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        rank=0,
        spatial_group=None,
        channel_group=None,
        device=0,
        parallelism=4,
        bayesian=False
    ):
        """
        Distributed 1D Convolution.

        random seed is required because of initialization/synchronization between "partner processes"

        input shape of forward pass: (N, Cin, Lin) or (Cin, Lin)
        output shape of forward pass: (N, Cout, Lout)
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert in_channels % 2 == 0, "number of input channels is not divisible by 2"
        assert out_channels % 2 == 0, "number of output channels is not divisible by 2"
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.rank = rank
        self.spatial_group = spatial_group
        self.channel_group = channel_group
        self.device = device
        self.parallelism = parallelism
        self.in_channels = in_channels
        self.out_channels = out_channels // 2
        self.bayesian = bayesian

        self.convTranspose2d = torch.nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias,
        ).to(device)
        
        # Initialize 
        k = np.sqrt(self.groups / (self.out_channels * 2 * self.kernel_size[0] * self.kernel_size[1]))
        self.convTranspose2d.weight.data.uniform_(-k, k)
        if bias:
            self.convTranspose2d.bias.data.uniform_(-k, k)
            
        self.empty = torch.empty(0, device=device, dtype=torch.float32)

        self.selector = ConvTransposeSelectionStrategy(self.bayesian, self.parallelism, self.convTranspose2d.bias)
       
        _average_weights(self, spatial_group, device)
        self.convTranspose2d_apply = ConvTranspose2d().apply

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
        Cij = self.convTranspose2d_apply(
            x,
            self.convTranspose2d.weight,
            self.convTranspose2d.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.in_channels,
            self.rank,
            self.spatial_group,
            self.channel_group,
            self.device,
            self.parallelism,
            self.empty,
            self.bayesian,
            self.selector
        )

        return Cij[0]


class ConvTranspose2d(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
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
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        selector
    ) -> torch.Tensor:
        
        y, x_aug, x, halo_sizes = dist_conv.conv_transpose2d_fourway(
            x, weight, bias, kernel_size, stride, padding, dilation, in_channels, rank, spatial_group, channel_group, empty, selector.convTranspose2d_fn
        )
        
        # Return output and any intermediate values needed for backward
        return y, x_aug, x, halo_sizes

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Unpack inputs
        (
            x,
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            dilation,
            in_channels,
            rank,
            spatial_group,
            channel_group,
            device,
            parallelism,
            empty,
            bayesian,
            selector
        ) = inputs
        _, x_aug, x_orig, halo_sizes = output

        ctx.mark_non_differentiable(x_aug)
        ctx.mark_non_differentiable(x_orig)
        ctx.mark_non_differentiable(halo_sizes)
        ctx.save_for_backward(x_aug, weight)

        ctx.halo_sizes = halo_sizes
        ctx.x_shape_padded = x_aug.shape
        ctx.x_shape = x_orig.shape

        ctx.stride, ctx.padding, ctx.dilation = (stride, padding, dilation)
        ctx.rank = rank
        ctx.spatial_group = spatial_group
        ctx.channel_group = channel_group
        ctx.rank = rank
        ctx.device = device
        ctx.parallelism = parallelism
        ctx.spatial_group = spatial_group
        ctx.channel_group = channel_group
        ctx.in_channels_global = in_channels
        ctx.b = bias
        ctx.empty = empty  # pre-initialized empty tensor
        ctx.bayesian = bayesian
        ctx.selector = selector
        

    @staticmethod
    def backward(
        ctx: torch.autograd.Function,
        grad_output: torch.Tensor,
        grad_x_aug: None,
        grad_x_orig: None,
        halo_sizes: None,
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
        rank, spatial_group, channel_group = (
            ctx.rank % 4,  # TODO: verify with communicator group
            ctx.spatial_group,
            ctx.channel_group,
        )
        selector = ctx.selector

        g_x, g_w = dist_conv.conv_transpose2d_fourway_backward(
            grad_output,
            x_aug,
            weight,
            halo_sizes,
            ctx.in_channels_global,
            rank,
            stride,
            padding,
            dilation,
            empty,
            spatial_group,
            channel_group,
            selector.convTranspose2d_grad_fn
        )

        g_b = selector.grad_bias(grad_output, spatial_group)
        
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
            None,
            None
        )

    @staticmethod
    def vmap(
        info,
        in_dims,
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
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        selector
    ) -> tuple[torch.Tensor, Optional[int]]:
        out = ConvTranspose2d.apply(
            x,
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            dilation,
            in_channels,
            rank,
            spatial_group,
            channel_group,
            device,
            parallelism,
            empty,
            bayesian,
            selector
        )

        return out, 0
