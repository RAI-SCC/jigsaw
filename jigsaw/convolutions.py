from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight

import jigsaw.distributed_convolutions as dist_conv
import jigsaw.hooks as hooks

from .utils_aggregate import gather_1d_results


class ConvSelectionStrategy:
    """Selection strategy for convolution kernels and halo communications."""

    def __init__(self, bayesian, parallelism, bias, kernel_size, stride):
        self.conv2d_fn = self._conv2d_fn_selection(bayesian)
        self.conv2d_input_fn = self._conv2d_input_fn_selection(bayesian)
        self.conv2d_weight_fn = self._conv2d_weight_fn_selection(bayesian)
        self.weight_reshape = self._reshape_weights_selection(bayesian)
        self.grad_bias = self._grad_bias_selection(parallelism, bias)
        self.communicate_halos_forward, self.communicate_halos_backward = (
            self._communicate_halos_selection(kernel_size, stride, parallelism)
        )

        # Select communicate halos
        self._communicate_halos_fn_selection(parallelism)

    def _communicate_halos_fn_selection(self, parallelism):
        if parallelism == 4:
            self.communicate_halos = dist_conv.communicate_halos
            self.append_halos = dist_conv.append_halos
            self.dist_conv2d_fn = dist_conv.conv2d_fourway
        elif parallelism == 8:
            self.communicate_halos = dist_conv.communicate_halos_8way
            self.append_halos = dist_conv.append_halos_8way
            self.dist_conv2d_fn = dist_conv.conv2d_eightway

    def _communicate_halos_selection(self, kernel_size, stride, parallelism):
        if isinstance(kernel_size, int) and isinstance(stride, int):
            if kernel_size == stride:
                return (
                    self._no_communicate_halos_forward,
                    self._no_communicate_halos_backward,
                )
            else:
                if parallelism == 4:
                    return (
                        self._communicate_halos_forward,
                        self._communicate_halos_backward_4way,
                    )
                else:
                    return (
                        self._communicate_halos_forward,
                        self._communicate_halos_backward_8way,
                    )
        else:
            if all(p == k for (p, k) in zip(kernel_size, stride)):
                return (
                    self._no_communicate_halos_forward,
                    self._no_communicate_halos_backward,
                )
            else:
                if parallelism == 4:
                    return (
                        self._communicate_halos_forward,
                        self._communicate_halos_backward_4way,
                    )
                else:
                    return (
                        self._communicate_halos_forward,
                        self._communicate_halos_backward_8way,
                    )

    def _communicate_halos_forward(
        self,
        x,
        right_to_left,
        left_to_right,
        local_rank,
        parallelism,
        empty,
        spatial_group,
        async_op,
    ):
        halo, _ = self.communicate_halos(
            x,
            right_to_left,
            left_to_right,
            local_rank,
            parallelism,
            empty,
            spatial_group,
            async_op=async_op,
        )
        x_with_halos = self.append_halos(
            x, halo, right_to_left, left_to_right, local_rank, 4, empty
        )

        return x_with_halos

    def _no_communicate_halos_forward(self, x, *args, **kwargs):
        x = x.contiguous()
        return x

    def _communicate_halos_backward_4way(
        self,
        g_x_aug,
        x_shape,
        right_to_left,
        left_to_right,
        local_rank,
        parallelism,
        empty,
        spatial_group,
        async_op,
    ):
        halo, wait_handle = dist_conv.communicate_halos(
            g_x_aug,
            right_to_left,
            left_to_right,
            local_rank,
            parallelism,
            empty,
            spatial_group,
            async_op=async_op,
        )

        wait_handle.wait()

        if spatial_group.rank() == 0:
            g_x = g_x_aug[..., : x_shape[-1]]
            if right_to_left > 0:
                g_x[..., -right_to_left:] = g_x[..., -right_to_left:] + halo
        elif spatial_group.rank() == 1:
            g_x = g_x_aug[..., -x_shape[-1] :]
            if left_to_right > 0:
                g_x[..., :left_to_right] = g_x[..., :left_to_right] + halo

        return g_x

    def _communicate_halos_backward_8way(
        self,
        g_x_aug,
        x_shape,
        right_to_left,
        left_to_right,
        local_rank,
        parallelism,
        empty,
        spatial_group,
        async_op,
    ):
        halo, _ = self.communicate_halos_8way(
            g_x_aug,
            right_to_left,
            left_to_right,
            local_rank,
            parallelism,
            empty,
            spatial_group,
            async_op=False,
        )

        if local_rank in [0, 1]:
            g_x = g_x_aug[..., : x_shape[-1]]
            if right_to_left[0] > 0:
                g_x[..., -right_to_left:] = g_x[..., -right_to_left[0] :] + halo[1]
        elif local_rank in [2, 3]:
            g_x = g_x_aug[..., -x_shape[-1] :]
            if left_to_right[0] > 0:
                g_x[..., : left_to_right[0]] = g_x[..., : left_to_right[0]] + halo[0]
            if right_to_left[1] > 0:
                g_x[..., -right_to_left[1]] = g_x[..., -right_to_left[1]] + halo[1]
        elif local_rank in [4, 5]:
            g_x = g_x_aug[..., -x_shape[-1] :]
            if left_to_right[1] > 0:
                g_x[..., : left_to_right[1]] = g_x[..., : left_to_right[1]] + halo[0]
            if right_to_left[2] > 0:
                g_x[..., -right_to_left[2]] = g_x[..., -right_to_left[2]] + halo[1]
        elif local_rank in [6, 7]:
            g_x = g_x_aug[..., -x_shape[-1] :]
            if left_to_right[2] > 0:
                g_x[..., : left_to_right[2]] = g_x[..., : left_to_right[2]] + halo[0]

        return g_x

    def _no_communicate_halos_backward(self, g_x_aug, *args, **kwargs):
        return g_x_aug

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
        return grad_b


class ConvTransposeSelectionStrategy:
    """Selection strategy for convolution transpose kernels and halo communications."""

    def __init__(self, bayesian, parallelism, bias, kernel_size, stride):
        """Initialize ConvTransposeSelectionStrategy."""
        self.convTranspose2d_fn = self._conv_transpose2d_fn_selection(bayesian)
        self.convTranspose2d_grad_fn = self._conv_transpose2d_grad_fn_selection(
            bayesian
        )
        self.weight_reshape = self._reshape_weights_selection(bayesian)
        self.grad_bias = self._grad_bias_selection(parallelism, bias)
        self.communicate_halos_forward, self.communicate_halos_backward = (
            self._communicate_halos_selection(kernel_size, stride, parallelism)
        )
        # Select communicate halos
        self._communicate_halos_fn_selection(parallelism)

    def _communicate_halos_fn_selection(self, parallelism):
        if parallelism == 4:
            self.communicate_halos = dist_conv.conv_transpose_halocalc_twoway
            self.append_halos = dist_conv.append_halos
            self.dummy_tensor = torch.tensor([0, 0])
            self.dist_conv_transpose2d_fn = dist_conv.conv_transpose2d_fourway
            self.dist_conv_transpose2d_backward_fn = (
                dist_conv.conv_transpose2d_fourway_backward
            )
        elif parallelism == 8:
            self.communicate_halos = dist_conv.conv_transpose_halocalc_fourway
            self.append_halos = dist_conv.append_halos_8way
            self.dummy_tensor = torch.tensor([[0, 0, 0], [0, 0, 0]])
            self.dist_conv_transpose2d_fn = dist_conv.conv_transpose2d_eightway
            self.dist_conv_transpose2d_backward_fn = (
                dist_conv.conv_transpose2d_eightway_backward
            )

    def _communicate_halos_selection(self, kernel_size, stride, parallelism):
        if isinstance(kernel_size, int) and isinstance(stride, int):
            if kernel_size == stride:
                return (
                    self._no_communicate_halos_forward,
                    self._no_communicate_halos_backward,
                )
            else:
                if parallelism == 4:
                    return (
                        self._communicate_halos_forward,
                        self._communicate_halos_backward_4way,
                    )
                else:
                    return (
                        self._communicate_halos_forward,
                        self._communicate_halos_backward_8way,
                    )
        else:
            if all(p == k for (p, k) in zip(kernel_size, stride)):
                return (
                    self._no_communicate_halos_forward,
                    self._no_communicate_halos_backward,
                )

            else:
                if parallelism == 4:
                    return (
                        self._communicate_halos_forward_4way,
                        self._communicate_halos_backward,
                    )
                else:
                    return (
                        self._communicate_halos_forward_8way,
                        self._communicate_halos_backward,
                    )

    def _communicate_halos_forward_4way(
        self,
        x,
        y,
        kernel_size,
        stride,
        padding,
        dilation,
        local_rank,
        spatial_group,
        parallelism,
    ):
        y_buffer, overlap_size, local_size_left, local_size_right = (
            self.convTranspose_halocalc_twoway(
                x,
                y,
                kernel_size[1],
                stride=stride[1],
                padding=padding[1],
                dilation=dilation[1],
                local_rank=local_rank,
                spatial_group=spatial_group,
                parallelism=parallelism,
            )
        )

        right_to_left = y.shape[-1] - local_size_left
        left_to_right = y.shape[-1] - local_size_right

        # Add overlapping bounds
        if local_rank // 2 == 0:  # LHS
            start_overlap_index = y.shape[-1] - overlap_size
            y[..., start_overlap_index:] = y[..., start_overlap_index:] + y_buffer
            y = y[..., :local_size_left]
        elif local_rank // 2 == 1:
            y[..., :overlap_size] = y[..., :overlap_size] + y_buffer
            y = y[..., -local_size_right:]

        halo_sizes = torch.tensor([right_to_left, left_to_right])

        return y, halo_sizes

    def _communicate_halos_forward_8way(
        self,
        x,
        y,
        kernel_size,
        stride,
        padding,
        dilation,
        local_rank,
        spatial_group,
        parallelism,
    ):
        y_buffer_left, y_buffer_right, overlap, local_size = (
            self.convTranspose_halocalc_fourway(
                x,
                y,
                kernel_size[1],
                stride=stride[1],
                padding=padding[1],
                dilation=dilation[1],
                local_rank=local_rank,
                spatial_group=spatial_group,
                parallelism=parallelism,
            )
        )

        right_to_left = [
            y.shape[-1] - local_size[0],
            y.shape[-1] - local_size[1],
            y.shape[-1] - local_size[2],
        ]
        # todo: proper boundaries for transpose convolutions
        left_to_right = [
            y.shape[-1] - local_size[0],
            y.shape[-1] - local_size[1],
            y.shape[-1] - local_size[2],
        ]

        # Add overlapping bounds
        match spatial_group.rank():
            case 0:
                start_overlap_index = y.shape[-1] - overlap[0]  # overlap_size in 4-way
                y[..., start_overlap_index:] = (
                    y[..., start_overlap_index:] + y_buffer_right
                )
                y = y[..., : local_size[0]]
            case 1:
                start_overlap_index = y.shape[-1] - overlap[1]  # overlap_size in 4-way
                y[..., : overlap[0]] = y[..., : overlap[0]] + y_buffer_left
                y[..., start_overlap_index:] = (
                    y[..., start_overlap_index:] + y_buffer_right
                )

            case 2:
                start_overlap_index = y.shape[-1] - overlap[1]  # overlap_size in 4-way
                y[..., : overlap[1]] = y[..., : overlap[1]] + y_buffer_left
                y[..., start_overlap_index:] = (
                    y[..., start_overlap_index:] + y_buffer_right
                )

            case 3:
                y[..., : overlap[2]] = y[..., : overlap[2]] + y_buffer_left
                y = y[..., -local_size[3] :]

        halo_sizes = torch.tensor([left_to_right, right_to_left])

        return y, halo_sizes

    def _no_communicate_halos_forward(self, x, y, *args, **kwargs):
        return y, self.dummy_tensor

    def _communicate_halos_backward(
        self,
        grad_output,
        right_to_left,
        left_to_right,
        local_rank,
        parallelism,
        empty,
        spatial_group,
        async_op,
    ):

        halo, _ = self.communicate_halos(
            grad_output,
            right_to_left,
            left_to_right,
            local_rank,
            parallelism,
            empty,
            spatial_group=spatial_group,
            async_op=async_op,
        )

        if (
            spatial_group.rank() == 0
        ):  # left and right groups # TODO get from process group?
            grad_output = torch.cat([grad_output, halo], dim=-1)
        else:
            grad_output = torch.cat([halo, grad_output], dim=-1)

        return grad_output

    def _no_communicate_halos_backward(self, grad_output, *args, **kwargs):
        return grad_output

    def _conv_transpose2d_fn_selection(self, bayesian):
        if bayesian:
            return torch.vmap(F.conv_transpose2d)
        else:
            return F.conv_transpose2d

    def _conv_transpose2d_grad_fn_selection(self, bayesian):
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
        return grad_b


class DistributedConv2D(torch.nn.Module):
    """Distributed 2D Convolution."""

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
        bayesian=False,
        dtype=torch.float32,
    ):
        """
        Distributed 2D Convolution.

        random seed is required because of initialization/synchronization between "partner processes"

        input shape of forward pass: (N, Cin, Lin) or (Cin, Lin)
        output shape of forward pass: (N, Cout, Lout)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels // 2  # TODO: //2 should be channel group size

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
        self.empty = torch.empty(0, device=device, dtype=dtype)
        self.bayesian = bayesian

        # * 2 should be channel group size
        k = np.sqrt(1 / (self.in_channels * 2 * kernel_size[0] * kernel_size[1]))
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros(
                self.out_channels, self.in_channels, kernel_size[0], kernel_size[1]
            )
            .uniform_(-k, k)
            .to(device)
        )
        self.bias = torch.nn.parameter.Parameter(
            torch.zeros(self.out_channels).uniform_(-k, k).to(device)
        )

        self.selector = ConvSelectionStrategy(
            self.bayesian,
            self.parallelism,
            self.bias,
            self.kernel_size,
            self.stride,
        )

        self.weight.register_post_accumulate_grad_hook(
            hooks.make_grad_sync_hook(self.spatial_group)
        )
        if self.bias is not None:
            self.bias.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.spatial_group)
            )
        for p in self.parameters():
            dist.broadcast(p.data, group_src=0, group=self.spatial_group)

        self.conv2d_apply = Conv2d.apply

    def get_parameter_count(self):
        """Return effective number of local parameters for Distributed Conv2D module."""
        num_param = (
            0
            if self.bias is None
            else self.bias.numel() // dist.get_world_size(group=self.spatial_group)
        )
        return num_param + self.weight.numel() // dist.get_world_size(
            group=self.spatial_group
        )

    def reshard_parameters(
        self,
        from_process_mesh: dist.ProcessGroup,
        to_process_mesh: dist.ProcessGroup,
        pm_row_key="row",
        pm_col_key="col",
    ):
        """Reshard parameters of weights according to a new process group."""
        self.parallelism = to_process_mesh[pm_col_key].size()
        self.selector = ConvSelectionStrategy(
            self.bayesian,
            self.parallelism,
            self.bias,
            self.kernel_size,
            self.stride,
        )

        from_weight_shape_local = self.weight.shape
        weight_shape_global = (
            from_weight_shape_local[0] * from_process_mesh[pm_col_key].size(),
            *from_weight_shape_local[1:],
        )
        to_weight_shape_local = (
            weight_shape_global[0] // to_process_mesh[pm_col_key].size(),
            *weight_shape_global[1:],
        )

        assert (
            weight_shape_global[0] % to_process_mesh[pm_col_key].size() == 0
        ), f"global weight shape[0] is {weight_shape_global[0]} which is not divisible by {to_process_mesh[pm_col_key].size()}"
        # gather
        global_weights = gather_1d_results(
            self.weight, from_process_mesh, dim=0, pm_key=pm_col_key
        )
        if self.bias is not None:
            from_bias_shape_local = self.bias.shape
            bias_shape_global = (
                from_bias_shape_local[0] * from_process_mesh[pm_col_key].size()
            )
            to_bias_shape_local = (
                bias_shape_global // to_process_mesh[pm_col_key].size()
            )

            assert (
                bias_shape_global % to_process_mesh[pm_col_key].size() == 0
            ), f"global bias shape is {bias_shape_global} which is not divisible by {to_process_mesh[pm_col_key].size()}"

            global_bias = gather_1d_results(
                self.bias, from_process_mesh, dim=0, pm_key=pm_col_key
            )

        col_rank, col_shard_size = (
            to_process_mesh[pm_col_key].rank(),
            to_weight_shape_local[0],
        )

        self.weight = torch.nn.Parameter(
            global_weights[
                col_rank * col_shard_size : (col_rank + 1) * col_shard_size, ...
            ].contiguous()
        )
        del global_weights

        col_rank, col_shard_size = (
            to_process_mesh[pm_col_key].rank(),
            to_bias_shape_local,
        )

        if self.bias is not None:
            self.bias = torch.nn.Parameter(
                global_bias[
                    col_rank * col_shard_size : (col_rank + 1) * col_shard_size
                ].contiguous()
            )
            del global_bias

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
        cij = self.conv2d_apply(
            x,
            self.weight,
            self.bias,
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
            self.selector,
        )

        return cij[0]


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
        selector,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Forward pass of distributed Conv2d function."""
        # Compute the forward pass
        y, x0, x1, right_to_left, left_to_right = selector.dist_conv2d_fn(
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
            selector=selector,
        )

        x_aug = torch.cat([x0, x1], dim=-3)  # concatenate along the channel dimension
        halo_sizes = torch.tensor(
            [right_to_left, left_to_right]
        )  # save halo sizes in tensor

        # y is result, rest of the outputs are required for the backward pass
        return y, x_aug, x, halo_sizes

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Set up context for backward pass.

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
            selector,
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
        None,
    ]:
        """Backward pass of Conv2d distributed across 4 or 8 processes."""
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
            x_shape_padded[int(ctx.bayesian) :],
            weight,
            grad_output.to(weight.dtype),  # handles autocast
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        left_to_right, right_to_left = (right_to_left, left_to_right)

        # g_x calculation
        # only communicates halos if they exist
        g_x = selector.communicate_halos_backward(
            g_x_aug,
            x_shape,
            right_to_left,
            left_to_right,
            local_rank,
            ctx.parallelism,
            empty,
            spatial_group,
            async_op=True,
        )

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
            weight.shape[int(ctx.bayesian) :],
            grad_output.to(weight.dtype),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        if g_x_handle is not None:
            g_x_handle.wait()

        g_x = (
            g_x[..., : ctx.in_channels_global // 2, :, :] + g_x_recv
            if local_rank % 2 == 0
            else g_x[..., ctx.in_channels_global // 2 :, :, :] + g_x_recv
        )

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
        selector,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Handle vmap for Distributed Conv2d."""
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
            selector,
        )

        return out, 0


class DistributedConvTranspose2D(torch.nn.Module):
    """Distributed ConvTranspose 2D."""

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
        bayesian=False,
        dtype=torch.float32,
    ):
        """
        Distributed ConvTranspose2D.

        random seed is required because of initialization/synchronization between "partner processes"

        input shape of forward pass: (N, Cin, Lin) or (S, N, Cin, Lin)
        output shape of forward pass: (N, Cout, Lout) or (S, N, Cout, Lout)
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
        self.device = device
        self.spatial_group = spatial_group
        self.channel_group = channel_group
        self.device = device
        self.parallelism = parallelism
        self.in_channels = in_channels
        self.out_channels = out_channels // 2
        self.bayesian = bayesian

        # Initialize
        k = np.sqrt(
            self.groups
            / (self.out_channels * 2 * self.kernel_size[0] * self.kernel_size[1])
        )
        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.in_channels,
                self.out_channels // self.groups,
                kernel_size[0],
                kernel_size[1],
            )
            .uniform_(-k, k)
            .to(device)
        )
        self.bias = (
            torch.nn.Parameter(
                torch.zeros(self.out_channels).uniform_(-k, k).to(device)
            )
            if bias
            else None
        )
        self.empty = torch.empty(0, device=self.device, dtype=dtype)

        self.selector = ConvTransposeSelectionStrategy(
            self.bayesian,
            self.parallelism,
            self.bias,
            self.kernel_size,
            self.stride,
        )

        for p in self.parameters():
            dist.broadcast(p.data, group_src=0, group=self.spatial_group)

        self.convTranspose2d_apply = ConvTranspose2d.apply

    def get_parameter_count(self):
        """Return effective number of local parameters for distributed model."""
        num_param = (
            0
            if self.bias is None
            else self.bias.numel() // dist.get_world_size(group=self.spatial_group)
        )
        return num_param + self.weight.numel() // dist.get_world_size(
            group=self.spatial_group
        )

    def reshard_parameters(
        self,
        from_process_mesh: dist.ProcessGroup,
        to_process_mesh: dist.ProcessGroup,
        pm_row_key="row",
        pm_col_key="col",
    ):
        """Reshard parameters of weights according to a new process group."""
        self.parallelism = to_process_mesh[pm_col_key].size()
        self.selector = ConvTransposeSelectionStrategy(
            self.bayesian,
            self.parallelism,
            self.bias,
            self.kernel_size,
            self.stride,
        )

        from_weight_shape_local = self.weight.shape
        weight_shape_global = (
            from_weight_shape_local[0],
            from_weight_shape_local[1] * from_process_mesh[pm_col_key].size(),
            *from_weight_shape_local[2:],
        )
        to_weight_shape_local = (
            weight_shape_global[0],
            weight_shape_global[1] // to_process_mesh[pm_col_key].size(),
            *weight_shape_global[2:],
        )

        assert (
            weight_shape_global[0] % to_process_mesh[pm_col_key].size() == 0
        ), f"global weight shape[0] is {weight_shape_global[0]} which is not divisible by {to_process_mesh[pm_col_key].size()}"

        # gather
        global_weights = gather_1d_results(
            self.weight, from_process_mesh, dim=1, pm_key=pm_col_key
        )

        if self.bias is not None:
            from_bias_shape_local = self.bias.shape
            bias_shape_global = (
                from_bias_shape_local[0] * from_process_mesh[pm_col_key].size()
            )
            to_bias_shape_local = (
                bias_shape_global // to_process_mesh[pm_col_key].size()
            )
            assert (
                bias_shape_global % to_process_mesh[pm_col_key].size() == 0
            ), f"global bias shape is {bias_shape_global} which is not divisible by {to_process_mesh[pm_col_key].size()}"

            global_bias = gather_1d_results(
                self.bias, from_process_mesh, dim=0, pm_key=pm_col_key
            )

        col_rank, col_shard_size = (
            to_process_mesh[pm_col_key].rank(),
            to_weight_shape_local[1],
        )

        self.weight = torch.nn.Parameter(
            global_weights[
                :, col_rank * col_shard_size : (col_rank + 1) * col_shard_size, ...
            ].contiguous()
        )

        del global_weights

        col_rank, col_shard_size = (
            to_process_mesh[pm_col_key].rank(),
            to_bias_shape_local,
        )

        if self.bias is not None:
            self.bias = torch.nn.Parameter(
                global_bias[
                    col_rank * col_shard_size : (col_rank + 1) * col_shard_size
                ].contiguous()
            )
            del global_bias

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
        cij = self.convTranspose2d_apply(
            x,
            self.weight,
            self.bias,
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
            self.selector,
        )

        return cij[0]


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
        selector,
    ) -> torch.Tensor:
        """Forward pass for distributed conv transpose 2d."""
        y, x_aug, x, halo_sizes = selector.dist_conv_transpose2d_fn(
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
            empty,
            selector,
        )

        # Return output and any intermediate values needed for backward
        return y, x_aug, x, halo_sizes

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Set up context."""
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
            selector,
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
            ctx.local_rank,  # TODO: verify with communicator group
            ctx.spatial_group,
            ctx.channel_group,
        )
        selector = ctx.selector

        g_x, g_w = selector.dist_conv_transpose2d_backward_fn(
            grad_output.to(weight.dtype),
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
            selector,
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
        selector,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Handle vmap for ConvTranspose2d function."""
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
            selector,
        )

        return out, 0
