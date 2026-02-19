from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from .distributed_convolutions import (
    conv2d_fourway,
)
import numpy as np
import Jigsaw.distributed_convolutions as dist_conv
from torch.nn.grad import conv2d_input, conv2d_weight

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
        local_rank=0,
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
        self.local_rank = local_rank
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
            self.local_rank,
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
        local_rank: int,
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
            local_rank=local_rank,
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
            local_rank,
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
        ctx.local_rank = local_rank
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
            ctx.local_rank % 4,
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
        local_rank: int,
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
            local_rank,
            spatial_group,
            channel_group,
            device,
            parallelism,
            empty,
            bayesian,
            selector
        )
        
        return out, 0
    
# TODO: temporarily introduce 8-way DistributedConv2d

class DistributedConv2D_8way(torch.nn.Module):
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
        local_rank=0,
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
        self.local_rank = local_rank
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
        
        self.conv2d_apply = Conv2d_8way().apply

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
            self.local_rank,
            self.spatial_group,
            self.channel_group,
            self.device,
            self.parallelism,
            self.empty,
            self.bayesian,
            self.selector
        )

        return Cij[0]


class Conv2d_8way(torch.autograd.Function):
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
        local_rank: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        selector
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        
        # Compute the forward pass        
        y, x0, x1, right_to_left, left_to_right = dist_conv.conv2d_eightway(
            x,
            weight,
            bias,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            in_channels=in_channels,
            local_rank=local_rank,
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
            local_rank,
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
        ctx.local_rank = local_rank
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
            ctx.local_rank % 8,
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
        halo, wait_handle = dist_conv.communicate_halos_8way(
            g_x_aug,
            right_to_left,
            left_to_right,
            local_rank,
            8,
            empty,
            spatial_group,
            async_op=True,
        )

        wait_handle.wait()

        # TODO: sort out halos for backward pass
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
        local_rank: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        selector
    ) -> tuple[torch.Tensor, Optional[int]]:
        
        out = Conv2d_8way.apply(
            x,
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            dilation,
            in_channels,
            local_rank,
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
        local_rank=0,
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
        self.local_rank = local_rank
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
            self.local_rank,
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
        local_rank: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        selector
    ) -> torch.Tensor:
        
        y, x_aug, x, halo_sizes = dist_conv.conv_transpose2d_fourway(
            x, weight, bias, kernel_size, stride, padding, dilation, in_channels, local_rank, spatial_group, channel_group, empty, selector.convTranspose2d_fn
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
            local_rank,
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
        ctx.local_rank = local_rank
        ctx.spatial_group = spatial_group
        ctx.channel_group = channel_group
        ctx.local_rank = local_rank
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
        local_rank, spatial_group, channel_group = (
            ctx.local_rank % 4,  # TODO: verify with communicator group
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
            local_rank,
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
        local_rank: int,
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
            local_rank,
            spatial_group,
            channel_group,
            device,
            parallelism,
            empty,
            bayesian,
            selector
        )

        return out, 0

class DistributedConvTranspose2D_8way(torch.nn.Module):
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
        local_rank=0,
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
        self.local_rank = local_rank
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
        self.convTranspose2d_apply = ConvTranspose2d_8way().apply

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
            self.local_rank,
            self.spatial_group,
            self.channel_group,
            self.device,
            self.parallelism,
            self.empty,
            self.bayesian,
            self.selector
        )

        return Cij[0]


class ConvTranspose2d_8way(torch.autograd.Function):
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
        local_rank: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        selector
    ) -> torch.Tensor:
        
        y, x_aug, x, halo_sizes = dist_conv.conv_transpose2d_8way(
            x, weight, bias, kernel_size, stride, padding, dilation, in_channels, local_rank, spatial_group, channel_group, empty, selector.convTranspose2d_fn
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
            local_rank,
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
        ctx.local_rank = local_rank
        ctx.spatial_group = spatial_group
        ctx.channel_group = channel_group
        ctx.local_rank = local_rank
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
        local_rank, spatial_group, channel_group = (
            ctx.local_rank % 8,  # TODO: verify with communicator group
            ctx.spatial_group,
            ctx.channel_group,
        )
        selector = ctx.selector

        g_x, g_w = dist_conv.conv_transpose2d_eightway_backward(
            grad_output,
            x_aug,
            weight,
            halo_sizes,
            ctx.in_channels_global,
            local_rank,
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
        local_rank: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        selector
    ) -> tuple[torch.Tensor, Optional[int]]:
        out = ConvTranspose2d_8way.apply(
            x,
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            dilation,
            in_channels,
            local_rank,
            spatial_group,
            channel_group,
            device,
            parallelism,
            empty,
            bayesian,
            selector
        )

        return out, 0
