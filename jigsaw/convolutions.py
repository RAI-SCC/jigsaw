"""Modules for distributed convolutions."""

from typing import Any, Callable, Optional, Protocol, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight
from torch.nn.parameter import Parameter

from . import comm, hooks
from . import distributed_convolutions as dist_conv
from .utils_aggregate import gather_1d_results


def _pair(x: int | tuple[int, ...]) -> tuple[int, int]:
    """Convert different types to 2-element tuple."""
    if isinstance(x, int):
        return (x, x)
    if len(x) == 1:
        return (x[0], x[0])
    return (x[0], x[1])


class ConvSelectionStrategy:
    """Selection strategy for convolution kernels and halo communications."""

    # type hints
    communicate_halos: Callable
    append_halos: Callable[..., torch.Tensor]
    dist_conv2d_fn: Callable
    communicate_halos_forward: Callable
    communicate_halos_backward: Callable

    def __init__(
        self,
        bayesian: bool,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bias: Parameter | None,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        device: torch.device,
    ):
        """
        Initialize selection strategy for conv.

        bayesian: bool
            Flag for whether the model is bayesian.
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.
        channel_group: dist.ProcessGroup
            Tensor-parallel process group.
        bias: torch.Tensor | None
            Bias if applicable
        kernel_size: Tuple[int, int]
            Kernel size of the convolution
        stride: Tuple[int, int]
            Size of the stride
        device: torch.device
            Device upon which communication occurs
        """
        self.conv2d_fn = self._conv2d_fn_selection(bayesian)
        self.conv2d_input_fn = self._conv2d_input_fn_selection(bayesian)
        self.conv2d_weight_fn = self._conv2d_weight_fn_selection(bayesian)
        self.weight_reshape = self._reshape_weights_selection(bayesian)
        self.grad_bias = self._grad_bias_selection(spatial_group, bias)
        self.communicate_halos_forward, self.communicate_halos_backward = (
            self._communicate_halos_selection(kernel_size, stride, spatial_group)
        )
        self.comm_fn = self._channel_comm_fn_selection(channel_group, device)
        # Select communicate halos
        self._communicate_halos_fn_selection(spatial_group)

    def _channel_comm_fn_selection(
        self, channel_group: dist.ProcessGroup, device: torch.device
    ) -> Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, dist.ProcessGroup, bool],
        tuple[torch.Tensor, list[Optional[dist.Work]]],
    ]:

        if device.type == "cuda":
            match channel_group.size():
                case 1:
                    return lambda data, buffer, empty, group, async_op: (data, [None])
                case 2:
                    return comm.comm_2way_alltoall
                case _:
                    raise NotImplementedError(
                        f"comm all to all is not implemented for channel group size {channel_group.size()}."
                    )
        elif device.type == "cpu":
            match channel_group.size():
                case 1:
                    return lambda data, buffer, empty, group, async_op: (data, [None])
                case 2:
                    return comm.comm_2way_p2p
                case _:
                    raise NotImplementedError(
                        f"comm all to all is not implemented for channel group size {channel_group.size()}."
                    )
        else:
            raise NotImplementedError(
                f"comm function not implemented for device type {device.type}"
            )

    def _communicate_halos_fn_selection(self, spatial_group: dist.ProcessGroup) -> None:
        """
        Select function for halo communication.

        Parameters
        ----------
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.
        """
        if spatial_group.size() == 2:
            self.communicate_halos = dist_conv.communicate_halos
            self.append_halos = dist_conv.append_halos
            self.dist_conv2d_fn = dist_conv.conv2d_fourway
        elif spatial_group.size() == 4:
            self.communicate_halos = dist_conv.communicate_halos_4space
            self.append_halos = dist_conv.append_halos_4space
            self.dist_conv2d_fn = dist_conv.conv2d_eightway

    def _communicate_halos_selection(
        self,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        spatial_group: dist.ProcessGroup,
    ) -> tuple[Callable, Callable]:
        if isinstance(kernel_size, int) and isinstance(stride, int):
            if kernel_size == stride:
                return (
                    self._no_communicate_halos_forward,
                    self._no_communicate_halos_backward,
                )
            else:
                if spatial_group.size() == 2:
                    return (
                        self._communicate_halos_forward,
                        self._communicate_halos_backward_2space,
                    )
                else:
                    return (
                        self._communicate_halos_forward,
                        self._communicate_halos_backward_4space,
                    )
        else:
            if all(p == k for (p, k) in zip(kernel_size, stride)):
                return (
                    self._no_communicate_halos_forward,
                    self._no_communicate_halos_backward,
                )
            else:
                if spatial_group.size() == 2:
                    return (
                        self._communicate_halos_forward,
                        self._communicate_halos_backward_2space,
                    )
                else:
                    return (
                        self._communicate_halos_forward,
                        self._communicate_halos_backward_4space,
                    )

    def _communicate_halos_forward(
        self,
        x: torch.Tensor,
        right_to_left: int,
        left_to_right: int,
        parallelism: int,
        empty: torch.Tensor,
        spatial_group: dist.ProcessGroup,
        async_op: bool,
    ) -> torch.Tensor:
        """
        Communicate halos in the forward pass.

        Starts with selecting the halos, communication, and then appending.

        Parameters
        ----------
        x: torch.Tensor
            Input data.
        right_to_left: int
            Size of the halo that is communicated from the right to left.
        left_to_right: int
            Size of the halo that is communicated from the left to right.
        parallelism: int
            Size of model group.
        empty: torch.Tensor
            Empty tensor on device.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.
        async_op: bool
            Flag for whether the communication should be asynchronous.

        Returns
        -------
        x_with_halos: torch.Tensor
        """
        halo, _ = self.communicate_halos(
            x,
            right_to_left,
            left_to_right,
            parallelism,
            empty,
            spatial_group,
            async_op=async_op,
        )
        x_with_halos = self.append_halos(
            x, halo, right_to_left, left_to_right, 4, empty
        )

        return x_with_halos

    def _no_communicate_halos_forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """No-op."""
        x = x.contiguous()
        return x

    def _communicate_halos_backward_2space(
        self,
        g_x_aug: torch.Tensor,
        x_shape: torch.Size,
        right_to_left: int,
        left_to_right: int,
        parallelism: int,
        empty: torch.Tensor,
        spatial_group: dist.ProcessGroup,
        async_op: bool,
    ) -> torch.Tensor:
        """
        Communicate halos (2 spatial x 2 channels) in the backward pass.

        Parameters
        ----------
        g_x_aug: torch.Tensor
            Gradient wrt input tensor including halos.
        x_shape: torch.Size
            Original shape of the input data.
        right_to_left: int
            Size of halo from right to left.
        left_to_right: int
            Size of halo from left to right.
        parallelism: int
            Parallelism degree.
        empty: torch.Tensor
            Empty tensor for computation reasons.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.
        async_op: bool
            Flag for whether the communication should be asynchronous.
        """
        halo, wait_handles = dist_conv.communicate_halos(
            g_x_aug,
            right_to_left,
            left_to_right,
            empty,
            spatial_group,
            async_op=async_op,
            comm_fn=self.comm_fn,
        )

        for w in wait_handles:
            if w is not None:
                w.wait()

        if spatial_group.rank() == 0:
            g_x = g_x_aug[..., : x_shape[-1]]
            if right_to_left > 0:
                g_x[..., -right_to_left:] = g_x[..., -right_to_left:] + halo
        elif spatial_group.rank() == 1:
            g_x = g_x_aug[..., -x_shape[-1] :]
            if left_to_right > 0:
                g_x[..., :left_to_right] = g_x[..., :left_to_right] + halo

        return g_x

    def _communicate_halos_backward_4space(
        self,
        g_x_aug: torch.Tensor,
        x_shape: torch.Size,
        right_to_left: list[int],
        left_to_right: list[int],
        empty: torch.Tensor,
        spatial_group: dist.ProcessGroup,
        async_op: bool,
    ) -> torch.Tensor:
        """
        Communicate halos across 4 x 2 (spatial x channels) in backwards pass.

        Parameters
        ----------
        g_x_aug: torch.Tensor
            Gradient wrt input tensor including halos.
        x_shape: torch.Size
            Original shape of the input data.
        right_to_left: list[int]
            Size of halo from right to left.
        left_to_right: list[int]
            Size of halo from left to right.
        empty: torch.Tensor
            Empty tensor for computation reasons.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.
        async_op: bool
            Flag for whether the communication should be asynchronous.
        """
        halo, _ = self.communicate_halos(
            g_x_aug,
            right_to_left,
            left_to_right,
            empty,
            spatial_group,
            async_op=False,
        )

        local_rank = spatial_group.rank()
        if local_rank == 0:
            g_x = g_x_aug[..., : x_shape[-1]]
            if right_to_left[0] > 0:
                g_x[..., -right_to_left[0] :] = g_x[..., -right_to_left[0] :] + halo[1]
        elif local_rank == 1:
            g_x = g_x_aug[..., -x_shape[-1] :]
            if left_to_right[0] > 0:
                g_x[..., : left_to_right[0]] = g_x[..., : left_to_right[0]] + halo[0]
            if right_to_left[1] > 0:
                g_x[..., -right_to_left[1]] = g_x[..., -right_to_left[1]] + halo[1]
        elif local_rank == 2:
            g_x = g_x_aug[..., -x_shape[-1] :]
            if left_to_right[1] > 0:
                g_x[..., : left_to_right[1]] = g_x[..., : left_to_right[1]] + halo[0]
            if right_to_left[2] > 0:
                g_x[..., -right_to_left[2]] = g_x[..., -right_to_left[2]] + halo[1]
        elif local_rank == 3:
            g_x = g_x_aug[..., -x_shape[-1] :]
            if left_to_right[2] > 0:
                g_x[..., : left_to_right[2]] = g_x[..., : left_to_right[2]] + halo[0]

        return g_x

    def _no_communicate_halos_backward(
        self, g_x_aug: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """No-op."""
        return g_x_aug

    def _conv2d_fn_selection(self, bayesian: bool) -> Callable:
        """
        Selector function for forward pass.

        Wraps the forward call in a vmap if necessary.

        Parameters
        ----------
        bayesian: bool
            Flag for whether model should be bayesian.

        Returns
        -------
        Callable
        """
        if bayesian:
            return torch.vmap(F.conv2d)
        else:
            return F.conv2d

    def _conv2d_input_fn_selection(self, bayesian: bool) -> Callable:
        """
        Selector function for backward pass.

        Wraps the backward grad_input calculation in a vmap if necessary.

        Parameters
        ----------
        bayesian: bool
            Flag for whether model should be bayesian.

        Returns
        -------
        Callable
        """
        if bayesian:
            return torch.vmap(conv2d_input, in_dims=(None, 0, 0))
        else:
            return conv2d_input

    def _conv2d_weight_fn_selection(self, bayesian: bool) -> Callable:
        """
        Selector function for backward pass.

        Wraps the backward grad_weight calculation in a vmap if necessary.

        Parameters
        ----------
        bayesian: bool
            Flag for whether model should be bayesian.

        Returns
        -------
        Callable
        """
        if bayesian:
            return torch.vmap(conv2d_weight, in_dims=(0, None, 0))
        else:
            return conv2d_weight

    def _reshape_weights_selection(self, bayesian: bool) -> Callable:
        """
        Selector function for reshaping the weights.

        Parameters
        ----------
        bayesian: bool
            Flag for whether the model is Bayesian.

        Returns
        -------
        reshape_weights: Callable
        """
        if bayesian:
            return self._reshape_weights
        else:
            return self._no_reshape_weights

    def _reshape_weights(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape weights as necessary for vmapped model.

        Parameters
        ----------
        w: torch.Tensor
            Convolution model weights.
        x: torch.Tensor
            data.

        Returns
        -------
        w: torch.Tensor
            Reshaped weights.
        """
        w = w.expand(x.shape[0], *w.shape)
        return w

    def _no_reshape_weights(self, w: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """No-op."""
        return w

    def _grad_bias_selection(
        self, spatial_group: dist.ProcessGroup, bias: Optional[torch.Tensor]
    ) -> Callable:
        """
        Selector function for grad bias reduction.

        Grad of bias needs to be all-reduced to synchronize across spatial process.

        Parameters
        ----------
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.
        bias: Optional[torch.Tensor]
            Bias tensor.

        Returns
        -------
        Callable
        """
        if bias is None:
            return self._no_bias
        elif spatial_group.size() == 2:
            return self._no_reduce_grad_bias
        else:
            return self._reduce_bias_over_spatial_group

    def _no_bias(self, *args: Any, **kwargs: Any) -> None:
        """No-op."""
        return None

    def _no_reduce_grad_bias(
        self,
        grad_output: torch.Tensor,
        spatial_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """
        Sum grad_output to get grad_bias.

        Parameters
        ----------
        grad_output: torch.Tensor
            Gradient wrt output.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.

        Returns
        -------
        grad_bias: torch.Tensor
            Gradient wrt bias.
        """
        return grad_output.sum(dim=(-4, -2, -1))

    def _reduce_bias_over_spatial_group(
        self,
        grad_output: torch.Tensor,
        spatial_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """
        Sum grad_output to get grad_bias.

        Parameters
        ----------
        grad_output: torch.Tensor
            Gradient wrt output.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.

        Returns
        -------
        grad_bias: torch.Tensor
            Gradient wrt bias.
        """
        grad_b = grad_output.sum(dim=(-4, -2, -1))
        return grad_b


class ConvTransposeSelectionStrategy:
    """Selection strategy for convolution transpose kernels and halo communications."""

    # type hints
    communicate_halos: Callable
    append_halos: Callable
    dist_conv_transpose2d_fn: Callable
    dist_conv_transpose2d_grad_fn: Callable
    weight_reshape: Callable
    grad_bias: Callable
    communicate_halos_forward: Callable
    communicate_halos_backward: Callable

    def __init__(
        self,
        bayesian: bool,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bias: Optional[Parameter],
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        device: torch.device,
    ):
        """
        Initialize ConvTransposeSelectionStrategy.

        Parameters
        ----------
        bayesian: bool
            Flag for whether the model is bayesian.
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.
        channel_group: dist.ProcessGroup
            Tensor-parallel process group.
        bias: Optional[Parameter]
            Bias if applicable
        kernel_size: tuple[int, int]
            Kernel size of the convolution
        stride: tuple[int, int]
            Size of the stride
        device: torch.device
            Device upon which the computations occur.
        """
        self.conv_transpose_2d_fn = self._conv_transpose2d_fn_selection(bayesian)
        self.conv_transpose2d_grad_fn = self._conv_transpose2d_grad_fn_selection(
            bayesian
        )
        self.weight_reshape = self._reshape_weights_selection(bayesian)
        self.grad_bias = self._grad_bias_selection(spatial_group, bias)
        self.communicate_halos_forward, self.communicate_halos_backward = (
            self._communicate_halos_selection(kernel_size, stride, spatial_group)
        )
        self.comm_fn = self._channel_comm_fn_selection(channel_group, device)
        # Select communicate halos
        self._communicate_halos_fn_selection(spatial_group)

    def _channel_comm_fn_selection(
        self, channel_group: dist.ProcessGroup, device: torch.device
    ) -> Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, dist.ProcessGroup, bool],
        tuple[torch.Tensor, list[Optional[dist.Work]]],
    ]:

        if device.type == "cuda":
            match channel_group.size():
                case 1:
                    return lambda data, buffer, empty, group, async_op: (data, [None])
                case 2:
                    return comm.comm_2way_alltoall
                case _:
                    raise NotImplementedError(
                        f"comm all to all is not implemented for channel parallelism {channel_group.size()}."
                    )
        elif device.type == "cpu":
            match channel_group.size():
                case 1:
                    return lambda data, buffer, empty, group, async_op: (data, [None])
                case 2:
                    return comm.comm_2way_p2p
                case _:
                    raise NotImplementedError(
                        f"comm all to all is not implemented for parallelism {channel_group.size()}."
                    )
        else:
            raise NotImplementedError(
                f"comm function not implemented for device type {device.type}"
            )

    def _communicate_halos_fn_selection(self, spatial_group: dist.ProcessGroup) -> None:
        """
        Selector function for halo communication.

        Parameters
        ----------
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.
        """
        if spatial_group.size() == 2:
            self.communicate_halos = dist_conv.conv_transpose_halocalc_twoway
            self.append_halos = dist_conv.append_halos
            self.dummy_tensor = torch.tensor([0, 0])
            self.dist_conv_transpose2d_fn = dist_conv.conv_transpose2d_twoway
            self.dist_conv_transpose2d_backward_fn = (
                dist_conv.conv_transpose2d_fourway_backward
            )
        elif spatial_group.size() == 4:
            self.communicate_halos = dist_conv.conv_transpose_halocalc_fourway
            self.append_halos = dist_conv.append_halos_4space
            self.dummy_tensor = torch.tensor([[0, 0, 0], [0, 0, 0]])
            self.dist_conv_transpose2d_fn = dist_conv.conv_transpose2d_fourway
            self.dist_conv_transpose2d_backward_fn = (
                dist_conv.conv_transpose2d_eightway_backward
            )

    def _communicate_halos_selection(
        self,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        spatial_group: dist.ProcessGroup,
    ) -> tuple[Callable, Callable]:
        """
        Selector function if no halo communication is required.

        In the case that stride==kernel size and each local domain is divisible
        by kernel size, no communication will be necessary. This detects that case
        to simplify the code.

        Parameters
        ----------
        kernel_size: tuple[int, int] | int
            Kernel size of the comvolution.
        stride: tuple[int, int] | int
            Stride of the convolution
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.

        Returns
        -------
        tuple[Callable, Callable]
            lambda functions for the halo communication functions
            in the forward and backward pass.
        """
        if all(p == k for (p, k) in zip(kernel_size, stride)):
            return (
                self._no_communicate_halos_forward,
                self._no_communicate_halos_backward,
            )

        else:
            if spatial_group.size() == 2:
                return (
                    self._communicate_halos_forward_2_space,
                    self._communicate_halos_backward,
                )
            else:
                return (
                    self._communicate_halos_forward_4_space,
                    self._communicate_halos_backward,
                )

    def _communicate_halos_forward_2_space(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        spatial_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Communicate halo in forward pass for 4-way parallelism.

        x: torch.Tensor
            Input data.
        y: torch.Tensor
            Local reesult of Conv Transpose.
        kernel_size: tuple[int, int] | int
            Kernel size.
        stride: tuple[int, int] | int
            Stride.
        padding: tuple[int, int] | int
            Padding.
        dilation: tuple[int, int] | int
            Dilation.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.
        """
        y_buffer, overlap_size, local_size_left, local_size_right = (
            dist_conv.conv_transpose_halocalc_twoway(
                x,
                y,
                kernel_size[1],
                stride=stride[1],
                padding=padding[1],
                dilation=dilation[1],
                spatial_group=spatial_group,
            )
        )

        right_to_left = y.shape[-1] - local_size_left
        left_to_right = y.shape[-1] - local_size_right

        # Add overlapping bounds
        if spatial_group.rank() == 0:  # LHS
            start_overlap_index = y.shape[-1] - overlap_size
            y[..., start_overlap_index:] = y[..., start_overlap_index:] + y_buffer
            y = y[..., :local_size_left]
        elif spatial_group.rank() == 1:
            y[..., :overlap_size] = y[..., :overlap_size] + y_buffer
            y = y[..., -local_size_right:]

        halo_sizes = torch.tensor([right_to_left, left_to_right])

        return y, halo_sizes

    def _communicate_halos_forward_4_space(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        spatial_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Communicate halo in forward pass for 8-way parallelism.

        x: torch.Tensor
            Input data.
        y: torch.Tensor
            Local reesult of Conv Transpose.
        kernel_size: tuple[int, int] | int
            Kernel size.
        stride: tuple[int, int] | int
            Stride.
        padding: tuple[int, int] | int
            Padding.
        dilation: tuple[int, int] | int
            Dilation.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.
        """
        y_buffer_left, y_buffer_right, overlap, local_size = (
            dist_conv.conv_transpose_halocalc_fourway(
                x,
                y,
                kernel_size[1],
                stride=stride[1],
                padding=padding[1],
                dilation=dilation[1],
                spatial_group=spatial_group,
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

    def _no_communicate_halos_forward(
        self, x: torch.Tensor, y: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """No-op."""
        return y, self.dummy_tensor

    def _communicate_halos_backward(
        self,
        grad_output: torch.Tensor,
        x_shape: torch.Size,
        right_to_left: int,
        left_to_right: int,
        parallelism: int,
        spatial_group: dist.ProcessGroup,
        async_op: bool,
    ) -> torch.Tensor:
        """
        Communicate halos in backward pass..

        grad_output: torch.Tensor
            Gradient wrt. output
        right_to_left: int
            Halo size communicated from right to left.
        left_to_right: int
            Halo size communicated from left to right.
        parallelism: int
            Size of parallelism.
        empty: torch.Tensor
            Empty tensor for computation reasons.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.
        async_op: bool
            Flag for whether communication should be asynchronous.
        """
        # TODO: implementation here is broken
        halo, _ = self.communicate_halos_backward(
            grad_output,
            x_shape,
            right_to_left,
            left_to_right,
            parallelism,
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

    def _no_communicate_halos_backward(
        self, grad_output: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """No-op."""
        return grad_output

    def _conv_transpose2d_fn_selection(self, bayesian: bool) -> Callable:
        """
        Selector function for forward pass.

        Wraps the forward call in a vmap if necessary.

        Parameters
        ----------
        bayesian: bool
            Flag for whether model should be bayesian.

        Returns
        -------
        Callable
        """
        if bayesian:
            return torch.vmap(F.conv_transpose2d)
        else:
            return F.conv_transpose2d

    def _conv_transpose2d_grad_fn_selection(self, bayesian: bool) -> Callable:
        """
        Selector function for conv transpose grad.

        Wraps the backward gradient computatio in a vmap if necessary.

        Parameters
        ----------
        bayesian: bool
            Flag for whether model should be bayesian.

        Returns
        -------
        Callable
        """
        if bayesian:
            return torch.vmap(
                torch.ops.aten.convolution_backward,
                in_dims=(0, 0, 0, None, None, None, None, None, None, None, None),
                out_dims=(0, 0, None),  # type: ignore[arg-type]
            )
        else:
            return torch.ops.aten.convolution_backward

    def _reshape_weights_selection(self, bayesian: bool) -> Callable:
        """
        Selector for reshaping weights.

        Parameters
        ----------
        bayesian: bool
            Flag for whether model is vmapped.

        Returns
        -------
        Callable.
        """
        if bayesian:
            return self._reshape_weights
        else:
            return self._no_reshape_weights

    def _reshape_weights(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape weights where necessary for vmap.

        Parameters
        ----------
        w: torch.Tensor
            Weights of ConvTranspose model.
        x: torch.Tensor
            data tensor.

        Returns
        -------
        w: torch.Tensor
            Expanded weights.
        """
        w = w.expand(x.shape[0], *w.shape)
        return w

    def _no_reshape_weights(
        self, w: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """No-op."""
        return w

    def _grad_bias_selection(
        self, spatial_group: dist.ProcessGroup, bias: Optional[torch.Tensor]
    ) -> Callable:
        """
        Selector function for grad bias reduction.

        Grad of bias needs to be all-reduced to synchronize across spatial process.

        Parameters
        ----------
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.
        bias: Optional[torch.Tensor]
            Bias tensor.

        Returns
        -------
        Callable
        """
        if bias is None:
            return self._no_bias
        elif spatial_group.size() == 2:
            return self._no_reduce_grad_bias
        else:
            return self._reduce_bias_over_spatial_group

    def _no_bias(self, *args: Any, **kwargs: Any) -> None:
        """No-op."""
        return None

    def _no_reduce_grad_bias(
        self,
        grad_output: torch.Tensor,
        spatial_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """
        Sum grad_output to get grad_bias.

        Parameters
        ----------
        grad_output: torch.Tensor
            Gradient wrt output.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.

        Returns
        -------
        grad_bias: torch.Tensor
            Gradient wrt bias.
        """
        return grad_output.sum(dim=(-4, -2, -1))

    def _reduce_bias_over_spatial_group(
        self, grad_output: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        Sum grad_output to get grad_bias.

        Parameters
        ----------
        grad_output: torch.Tensor
            Gradient wrt output.
        args: Any
            Additional args.
        kwargs: Any
            Additional key words.

        Returns
        -------
        grad_bias: torch.Tensor
            Gradient wrt bias.
        """
        grad_b = grad_output.sum(dim=(-4, -2, -1))
        return grad_b


class DistributedConv2D(torch.nn.Module):
    """Distributed 2D Convolution."""

    # type-hints
    bias: Optional[Parameter]
    parallelism: int
    selector: ConvSelectionStrategy
    bayesian: bool
    spatial_group: dist.ProcessGroup
    channel_group: dist.ProcessGroup
    empty: torch.Tensor
    device: torch.device
    conv2d_apply: Callable[..., torch.Tensor]
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: str | tuple[int, int]
    dilation: tuple[int, int]

    def __init__(
        self,
        # --- core conv (required) ---
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        # --- distributed (required) ---
        device: torch.device,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        # --- core conv (optional, pytorch-standard-order) ---
        stride: int | tuple[int, ...] = 1,
        padding: str | tuple[int, ...] = (0,),
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        # --- distributed (optional) ---
        parallelism: int = 4,
        bayesian: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Distributed 2D Convolution.

        input shape of forward pass: (N, Cin, Lin) or (Cin, Lin)
        output shape of forward pass: (N, Cout, Lout)

        Parameters
        ----------
        in_channels: int
            Number of in_channels globally.
        out_channels: int
            Number of in_channels globally.
        kernel_size: int | tuple[int, ...]
            Kernel size.
        device: torch.Device
            Device where the computations will be done.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.
        channel_group: dist.ProcessGroup
            Process group for channel-parallel ranks.
        stride: int | tuple[int, ...]
            Stride in the convolution.
        padding: str | tuple[int, ...]
            Padding. Not yet implemented.
        dilation: int | tuple[int, ...]
            Dilation. Not yet implemented.
        groups: int
            Number of groups. Not yet implemented.
        bias: bool
            Flag for whether bias is present.
        parallelism: int
            Size of model-parallel groups.
        bayesian: bool
            Flag for whether model is vmapped.
        dtype: torch.dtype
            Data type of the convolution.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels // 2  # TODO: //2 should be channel group size

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)

        if isinstance(padding, str):
            self.padding = padding
        else:
            self.padding = _pair(padding)

        self.groups = groups
        self.spatial_group = spatial_group
        self.channel_group = channel_group
        self.device = device
        self.parallelism = parallelism
        self.dtype = dtype
        self.empty = torch.empty(0, device=self.device, dtype=self.dtype)
        self.bayesian = bayesian

        k = np.sqrt(
            1
            / (
                self.in_channels
                * channel_group.size()
                * self.kernel_size[0]
                * self.kernel_size[1]
            )
        )
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            )
            .uniform_(-k, k)
            .to(device)
        )
        self.bias = (
            torch.nn.parameter.Parameter(
                torch.zeros(self.out_channels).uniform_(-k, k).to(device)
            )
            if bias
            else None
        )

        self.selector = ConvSelectionStrategy(
            self.bayesian,
            self.spatial_group,
            self.channel_group,
            self.bias,
            self.kernel_size,
            self.stride,
            self.device,
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

    def get_parameter_count(self) -> int:
        """Return effective number of local parameters for Distributed Conv2D module."""
        num_param = (
            0
            if self.bias is None
            else self.bias.numel() // dist.get_world_size(group=self.spatial_group)
        )
        return num_param + self.weight.numel() // dist.get_world_size(
            group=self.spatial_group
        )

    @classmethod
    def convert_to_jigsaw(
        cls,
        model: torch.nn.Conv2d,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> "DistributedConv2D":
        """
        Convert torch.nn.Conv2D to jigsaw based on given process mesh.

        Parameters
        ----------
        model: torch.nn.Conv2D
            Instance of conv2d class that should be converted to jigsaw conv2d.
        process_mesh_dict: dict
            Process mesh dict that comtains row, column, and model process groups
        device: torch.device
            Device upon which computation happens.
        row_kw: str
            Key for accessing domain process groups.
        col_kw: str
            Key for accessing channel process groups.
        model_kw: str
            Key for accessing model process groups.
            Model process group should contain the Union of the row and column groups.

        Returns
        -------
        model: jigsaw.convolutions.DistributedConv2D
            Instance of jigsaw distributed Conv2D.
            Jigsaw layer contains weights and biases given from torch.nn.Conv2D layer.
        """
        assert model.in_channels % process_mesh_dict[col_kw].size() == 0, (
            "in_features is not divisible by process mesh cols"
        )

        instance = cls(
            in_channels=model.in_channels,
            out_channels=model.out_channels,  # global ?
            kernel_size=cast(tuple[int, ...], model.kernel_size),
            stride=cast(tuple[int, ...], model.stride),
            padding=cast(tuple[int, ...], model.padding),
            dilation=cast(tuple[int, ...], model.dilation),
            groups=model.groups,
            bias=True if model.bias is not None else False,
            spatial_group=process_mesh_dict[row_kw],
            channel_group=process_mesh_dict[col_kw],
            device=device,
            parallelism=process_mesh_dict[model_kw].size(),
            bayesian=False,  # TODO
            dtype=model.weight.dtype,
        )
        # Map weights — clone so the original graph is detached
        in_rank = process_mesh_dict[col_kw].rank()
        in_shard = model.weight.shape[0] // process_mesh_dict[col_kw].size()

        instance.weight.data = (
            model.weight.data[in_rank * in_shard : (in_rank + 1) * in_shard]
            .clone()
            .to(device)
        )

        if model.bias is not None:
            assert instance.bias is not None
            bias_size = model.out_features // process_mesh_dict[col_kw].size()
            instance.bias.data = model.bias.data[
                in_rank * bias_size : (in_rank + 1) * bias_size
            ].clone()

        return instance

    def convert_to_torch(
        self,
        process_mesh_dict: dict,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> torch.nn.Conv2d:
        """
        Convert jigsaw conv2d layer to torch.

        Parameters
        ----------
        process_mesh_dict: dict
            Process mesh dict that comtains row, column, and model process groups.
        row_kw: str
            Key for accessing row process groups.
        col_kw: str
            Key for accessing col process groups.
        model_kw: str
            Key for accessing model process groups.
            Model process group should contain the Union of the row and column groups.
        """
        weights, bias = self.merge_weights(process_mesh_dict, col_kw)
        conv2d = torch.nn.Conv2d(
            in_channels=cast(int, weights.shape[0]),
            out_channels=cast(int, weights.shape[1] * self.groups),
            kernel_size=cast(tuple[int, int], self.kernel_size),
            padding=cast(tuple[int, int], self.padding),  # not supported
            dilation=cast(tuple[int, int], self.dilation),  # not supported
            groups=self.groups,  # not supported
            bias=True if bias is not None else False,
            padding_mode="zeros",  # hard-coded
        )
        assert isinstance(weights, torch.Tensor)
        conv2d.weights.data = weights
        conv2d.bias = bias

        return conv2d

    def merge_weights(
        self,
        process_mesh_dict: dict,
        col_kw: str = "col",
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Merge weights from jigsaw model.

        Parameters will be merged in a way such that they can be transplanted into a
        torch.nn.Conv2D layer natively.

        Parameters
        ----------
        process_mesh_dict: dict
            Process mesh dict that comtains row, column, and model process groups
        col_kw: str
            Key for accessing col process groups

        Returns
        -------
        global_weights: torch.Tensor
            Merged weights.
        global_bias: Union[torch.Tensor, None]
            Merged bias, if it exists.
        """
        global_weights = gather_1d_results(
            self.weight, process_mesh_dict, dim=0, pm_key=col_kw
        )

        global_bias = (
            gather_1d_results(self.bias, process_mesh_dict, dim=0, pm_key=col_kw)
            if self.bias is not None
            else None
        )
        return global_weights, global_bias

    def reshard_parameters(
        self,
        from_process_mesh_dict: dict,
        to_process_mesh_dict: dict,
        pm_row_key: str = "row",
        pm_col_key: str = "col",
        pm_model_key: str = "model",
    ) -> None:
        """Reshard parameters of weights according to a new process group."""
        self.parallelism = to_process_mesh_dict[pm_col_key].size()
        self.selector = ConvSelectionStrategy(
            self.bayesian,
            self.spatial_group,
            self.channel_group,
            self.bias,
            self.kernel_size,
            self.stride,
            self.device,
        )

        from_weight_shape_local = self.weight.shape
        weight_shape_global = (
            from_weight_shape_local[0] * from_process_mesh_dict[pm_col_key].size(),
            *from_weight_shape_local[1:],
        )
        to_weight_shape_local = (
            weight_shape_global[0] // to_process_mesh_dict[pm_col_key].size(),
            *weight_shape_global[1:],
        )

        assert weight_shape_global[0] % to_process_mesh_dict[pm_col_key].size() == 0, (
            f"global weight shape[0] is {weight_shape_global[0]} which is not divisible by {to_process_mesh_dict[pm_col_key].size()}"
        )
        # gather
        global_weights, global_bias = self.merge_weights(
            from_process_mesh_dict, pm_col_key
        )

        if self.bias is not None:
            from_bias_shape_local = self.bias.shape
            bias_shape_global = (
                from_bias_shape_local[0] * from_process_mesh_dict[pm_col_key].size()
            )
            to_bias_shape_local = (
                bias_shape_global // to_process_mesh_dict[pm_col_key].size()
            )

            assert bias_shape_global % to_process_mesh_dict[pm_col_key].size() == 0, (
                f"global bias shape is {bias_shape_global} which is not divisible by {to_process_mesh_dict[pm_col_key].size()}"
            )

            global_bias = gather_1d_results(
                self.bias, from_process_mesh_dict, dim=0, pm_key=pm_col_key
            )

        col_rank, col_shard_size = (
            to_process_mesh_dict[pm_col_key].rank(),
            to_weight_shape_local[0],
        )

        self.weight = torch.nn.Parameter(
            global_weights[
                col_rank * col_shard_size : (col_rank + 1) * col_shard_size, ...
            ].contiguous()
        )
        del global_weights

        col_rank, col_shard_size = (
            to_process_mesh_dict[pm_col_key].rank(),
            to_bias_shape_local,
        )

        if isinstance(global_bias, torch.Tensor) and isinstance(
            global_bias, torch.Tensor
        ):
            self.bias = torch.nn.Parameter(
                global_bias[
                    col_rank * col_shard_size : (col_rank + 1) * col_shard_size
                ].contiguous()
            )
            del global_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of distributed convolution.

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
            self.spatial_group,
            self.channel_group,
            self.device,
            self.parallelism,
            self.empty,
            self.bayesian,
            self.selector,
        )

        return cast(torch.Tensor, cij[0])


class Conv2d(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        kernel_size: tuple[int, int],
        stride: int,
        padding: int,
        dilation: int,
        in_channels: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        selector: ConvSelectionStrategy,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: Any,
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
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

        # Save everything needed for backward
        ctx.save_for_backward(x_aug, weight)
        c = cast(ConvCtx, ctx)
        c.halo_sizes = halo_sizes
        # Save non-tensor attributes
        c.device = device
        c.parallelism = parallelism
        c.spatial_group = spatial_group
        c.channel_group = channel_group
        c.in_channels_global = in_channels
        c.b = bias
        c.empty = empty
        c.x_shape_padded = x_aug.shape
        c.x_shape = x_orig.shape
        c.stride = stride
        c.padding = padding
        c.dilation = dilation
        c.bayesian = bayesian
        c.selector = selector

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
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
        c = cast(ConvCtx, ctx)
        x_aug, weight = c.saved_tensors
        empty = c.empty
        stride, padding, dilation = (c.stride, c.padding, c.dilation)
        spatial_group, channel_group = (
            c.spatial_group,
            c.channel_group,
        )
        selector = c.selector

        right_to_left, left_to_right = c.halo_sizes[0], c.halo_sizes[1]
        x_shape_padded, x_shape = c.x_shape_padded, c.x_shape

        g_b = selector.grad_bias(grad_output, c.spatial_group)

        g_x_aug = selector.conv2d_input_fn(
            x_shape_padded[int(c.bayesian) :],
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
            c.parallelism,
            empty,
            spatial_group,
            async_op=True,
        )

        g_x_buffer = torch.zeros(
            (*g_x.shape[:-3], g_x.shape[-3] // 2, *g_x.shape[-2:]),
            device=g_x.device,
            dtype=g_x.dtype,
        )

        g_x_recv, g_x_handles = dist_conv.send_g_x_channels_2d(
            g_x,
            g_x_buffer,
            empty,
            c.in_channels_global,
            channel_group,
            async_op=True,
            comm_fn=selector.comm_fn,
        )

        g_w = selector.conv2d_weight_fn(
            x_aug,
            weight.shape[int(c.bayesian) :],
            grad_output.to(weight.dtype),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        for g_x_handle in g_x_handles:
            if g_x_handle is not None:
                g_x_handle.wait()

        g_x = (
            g_x[..., : c.in_channels_global // 2, :, :] + g_x_recv
            if channel_group.rank() % 2 == 0
            else g_x[..., c.in_channels_global // 2 :, :, :] + g_x_recv
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
        info: Any,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        in_channels: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        selector: ConvSelectionStrategy,
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

    # type hints
    bias: Optional[Parameter]
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: str | tuple[int, int]
    dilation: tuple[int, int]

    def __init__(
        self,
        # --- core ConvTranspose (required) ---
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        # --- distributed (required) ---
        device: torch.device,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        # --- core ConvTranspose (optional, pytorch-standard order) ---
        stride: int | tuple[int, ...] = 1,
        padding: str | tuple[int, ...] = (0,),
        dilation: tuple[int, ...] = (1,),
        groups: int = 1,
        bias: bool = True,
        # --- distributed (optional) ---
        parallelism: int = 4,
        bayesian: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Distributed ConvTranspose2D.

        input shape of forward pass: (N, Cin, Lin) or (S, N, Cin, Lin)
        output shape of forward pass: (N, Cout, Lout) or (S, N, Cout, Lout)

        Distributed 2D Convolution.

        input shape of forward pass: (N, Cin, Lin) or (Cin, Lin)
        output shape of forward pass: (N, Cout, Lout)

        Parameters
        ----------
        in_channels: int
            Number of in_channels globally.
        out_channels: int
            Number of in_channels globally.
        kernel_size: int | tuple[int, ...]
            Kernel size.
        device: torch.Device
            Device where the computations will be done.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.
        channel_group: dist.ProcessGroup
            Process group for channel-parallel ranks.
        stride: int | tuple[int, ...]
            Stride in the convolution.
        padding: int | tuple[int, ...]
            Padding. Not yet implemented.
        dilation: int | tuple[int, ...]
            Dilation. Not yet implemented.
        groups: int
            Number of groups. Not yet implemented.
        bias: bool
            Flag for whether bias is present.
        parallelism: int
            Size of model-parallel groups.
        bayesian: bool
            Flag for whether model is vmapped.
        dtype: torch.dtype
            Data type of the convolution.
        """
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)

        if isinstance(padding, str):
            self.padding = padding
        else:
            self.padding = _pair(padding)

        assert in_channels % 2 == 0, "number of input channels is not divisible by 2"
        assert out_channels % 2 == 0, "number of output channels is not divisible by 2"

        self.groups = groups
        self.device = device
        self.spatial_group = spatial_group
        self.channel_group = channel_group
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
                self.kernel_size[0],
                self.kernel_size[1],
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
            self.spatial_group,
            self.channel_group,
            self.bias,
            self.kernel_size,
            self.stride,
            self.device,
        )

        for p in self.parameters():
            dist.broadcast(p.data, group_src=0, group=self.spatial_group)

        self.convTranspose2d_apply = ConvTranspose2d.apply

    def get_parameter_count(self) -> int:
        """Return effective number of local parameters for distributed model."""
        num_param = (
            0
            if self.bias is None
            else self.bias.numel() // dist.get_world_size(group=self.spatial_group)
        )
        return num_param + self.weight.numel() // dist.get_world_size(
            group=self.spatial_group
        )

    @classmethod
    def convert_to_jigsaw(
        cls,
        model: torch.nn.ConvTranspose2d,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> "DistributedConvTranspose2D":
        """
        Convert torch.nn.ConvTranspose2D to jigsaw based on given process mesh.

        Parameters
        ----------
        model: torch.nn.ConvTranspose2d
            Instance of convTranspose2d class that should be converted to jigsaw linear.
        process_mesh_dict: dict
            Process mesh dict that comtains row, column, and model process groups.
        device: torch.device
            Device upon which computations will occur.
        row_kw: str
            Key for accessing row process groups.
        col_kw: str
            Key for accessing col process groups.
        model_kw: str
            Key for accessing model process groups.
            Model process group should contain the Union of the row and column groups.

        Returns
        -------
        model: jigsaw.convolutions.DistributedConvTranspose2D
            Instance of jigsaw distributed ConvTranspose layer.
            Jigsaw layer contains weights and biases given from torch.nn.ConvTranspose2d layer.
        """
        assert model.in_channels % process_mesh_dict[col_kw].size() == 0, (
            "in_features is not divisible by process mesh cols"
        )

        instance = cls(
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            kernel_size=model.kernel_size,
            stride=model.stride,
            padding=model.padding,
            dilation=model.dilation,
            groups=model.groups,
            bias=True if model.bias is not None else False,
            spatial_group=process_mesh_dict[row_kw],
            channel_group=process_mesh_dict[col_kw],
            device=device,
            parallelism=process_mesh_dict[model_kw].size(),
            bayesian=False,  # TODO
            dtype=model.weight.dtype,
        )
        # Map weights — clone so the original graph is detached
        in_rank = process_mesh_dict[col_kw].rank()
        in_shard = model.in_channels // process_mesh_dict[col_kw].size()

        instance.weight.data = (
            model.weight.data[:, in_rank * in_shard : (in_rank + 1) * in_shard]
            .clone()
            .to(device)
        )

        if isinstance(model.bias, torch.Tensor) and isinstance(
            instance.bias, torch.Tensor
        ):
            bias_size = model.out_features // process_mesh_dict[col_kw].size()
            instance.bias.data = model.bias.data[
                in_rank * bias_size : (in_rank + 1) * bias_size
            ].clone()
        return instance

    def convert_to_torch(
        self,
        process_mesh_dict: dict,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> torch.nn.ConvTranspose2d:
        """
        Convert jigsaw transpose convolution layer to torch.

        Parameters
        ----------
        process_mesh_dict: dict
            Process mesh dict that comtains row, column, and model process groups.
        row_kw: str
            Key for accessing row process groups.
        col_kw: str
            Key for accessing col process groups.
        model_kw: str
            Key for accessing model process groups.
            Model process group should contain the Union of the row and column groups.
        """
        weights, bias = self.merge_weights(process_mesh_dict, col_kw)
        convtranspose2d = torch.nn.ConvTranspose2d(
            in_channels=weights.shape[0],
            out_channels=weights.shape[1] * self.groups,
            kernel_size=cast(tuple[int, int], self.kernel_size),
            padding=cast(tuple[int, int], self.padding),  # not supported
            dilation=cast(tuple[int, int], self.dilation),  # not supported
            groups=self.groups,  # not supported
            bias=True if bias is not None else False,
            padding_mode="zeros",  # hard-coded
        )
        convtranspose2d.weights.data = weights
        if isinstance(convtranspose2d.bias, torch.Tensor) and isinstance(
            bias, torch.Tensor
        ):
            convtranspose2d.bias.data = bias

        return convtranspose2d

    def merge_weights(
        self,
        process_mesh_dict: dict,
        col_kw: str = "col",
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Merge weights from jigsaw model.

        Parameters will be merged in a way such that they can be transplanted into a
        torch.nn.ConvTranspose2D layer natively.

        Parameters
        ----------
        process_mesh_dict: dict
            Process mesh dict that comtains row, column, and model process groups
        col_kw: str
            Key for accessing col process groups

        Returns
        -------
        global_weights: torch.Tensor
            Merged weights.
        global_bias: Union[torch.Tensor, None]
            Merged bias, if it exists.
        """
        global_weights = gather_1d_results(
            self.weight, process_mesh_dict, dim=1, pm_key=col_kw
        )

        global_bias = (
            gather_1d_results(self.bias, process_mesh_dict, dim=0, pm_key=col_kw)
            if self.bias is not None
            else None
        )
        return global_weights, global_bias

    def reshard_parameters(
        self,
        from_process_mesh_dict: dict,
        to_process_mesh_dict: dict,
        pm_row_key: str = "row",
        pm_col_key: str = "col",
    ) -> None:
        """Reshard parameters of weights according to a new process group."""
        self.parallelism = to_process_mesh_dict[pm_col_key].size()
        self.selector = ConvTransposeSelectionStrategy(
            self.bayesian,
            self.spatial_group,
            self.channel_group,
            self.bias,
            self.kernel_size,
            self.stride,
            self.device,
        )

        from_weight_shape_local = self.weight.shape
        weight_shape_global = (
            from_weight_shape_local[0],
            from_weight_shape_local[1] * from_process_mesh_dict[pm_col_key].size(),
            *from_weight_shape_local[2:],
        )
        to_weight_shape_local = (
            weight_shape_global[0],
            weight_shape_global[1] // to_process_mesh_dict[pm_col_key].size(),
            *weight_shape_global[2:],
        )

        assert weight_shape_global[0] % to_process_mesh_dict[pm_col_key].size() == 0, (
            f"global weight shape[0] is {weight_shape_global[0]} which is not divisible by {to_process_mesh_dict[pm_col_key].size()}"
        )

        # gather
        global_weights, global_bias = self.merge_weights(
            from_process_mesh_dict, pm_col_key
        )

        if self.bias is not None:
            from_bias_shape_local = self.bias.shape
            bias_shape_global = (
                from_bias_shape_local[0] * from_process_mesh_dict[pm_col_key].size()
            )
            to_bias_shape_local = (
                bias_shape_global // to_process_mesh_dict[pm_col_key].size()
            )
            assert bias_shape_global % to_process_mesh_dict[pm_col_key].size() == 0, (
                f"global bias shape is {bias_shape_global} which is not divisible by {to_process_mesh_dict[pm_col_key].size()}"
            )

            global_bias = gather_1d_results(
                self.bias, from_process_mesh_dict, dim=0, pm_key=pm_col_key
            )

        col_rank, col_shard_size = (
            to_process_mesh_dict[pm_col_key].rank(),
            to_weight_shape_local[1],
        )
        self.weight = torch.nn.Parameter(
            global_weights[
                :, col_rank * col_shard_size : (col_rank + 1) * col_shard_size, ...
            ].contiguous()
        )

        del global_weights

        if isinstance(global_bias, torch.Tensor):
            col_rank, col_shard_size = (
                to_process_mesh_dict[pm_col_key].rank(),
                to_bias_shape_local,
            )
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
            self.spatial_group,
            self.channel_group,
            self.device,
            self.parallelism,
            self.empty,
            self.bayesian,
            self.selector,
        )
        cij_ = cast(torch.Tensor, cij[0])
        return cij_


class ConvCtx(Protocol):
    """Protocol for types stored within ctx."""

    halo_sizes: torch.Tensor
    x_shape_padded: torch.Size
    x_shape: torch.Size
    kernel_size: int | tuple[int, ...]
    stride: int | tuple[int, ...]
    padding: int | tuple[int, ...]
    dilation: int | tuple[int, ...]
    spatial_group: dist.ProcessGroup
    channel_group: dist.ProcessGroup
    device: torch.device
    parallelism: int
    in_channels_global: int
    b: torch.Tensor | None
    bayesian: bool
    selector: ConvSelectionStrategy
    empty: torch.Tensor
    saved_tensors: tuple[torch.Tensor, torch.Tensor]


class ConvTransposeCtx(Protocol):
    """Protocol for types stored within ctx."""

    halo_sizes: int | tuple[int, ...]
    x_shape_padded: torch.Size
    x_shape: torch.Size
    kernel_size: int | tuple[int, ...]
    stride: int | tuple[int, ...]
    padding: int | tuple[int, ...]
    dilation: int | tuple[int, ...]
    spatial_group: dist.ProcessGroup
    channel_group: dist.ProcessGroup
    device: torch.device
    parallelism: int
    in_channels_global: int
    b: torch.Tensor | None
    bayesian: bool
    selector: ConvTransposeSelectionStrategy
    empty: torch.Tensor
    saved_tensors: tuple[torch.Tensor, torch.Tensor]


class ConvTranspose2d(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        in_channels: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        selector: ConvTransposeSelectionStrategy,
    ) -> tuple[Any, Any, torch.Tensor, Any]:
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
            spatial_group,
            channel_group,
            empty,
            selector,
        )

        # Return output and any intermediate values needed for backward
        return y, x_aug, x, halo_sizes

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: Any,
        output: tuple[Any, Any, torch.Tensor, Any],
    ) -> None:
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

        c = cast(ConvTransposeCtx, ctx)

        c.halo_sizes = halo_sizes
        c.x_shape_padded = x_aug.shape
        c.x_shape = x_orig.shape

        c.stride, c.padding, c.dilation = (stride, padding, dilation)
        c.spatial_group = spatial_group
        c.channel_group = channel_group
        c.device = device
        c.parallelism = parallelism
        c.in_channels_global = in_channels
        c.b = bias
        c.empty = empty  # pre-initialized empty tensor
        c.bayesian = bayesian
        c.selector = selector

    @staticmethod
    def backward(
        ctx: ConvTransposeCtx,
        grad_output: torch.Tensor,
        grad_x_aug: torch.Tensor,
        grad_x_orig: torch.Tensor,
        halo_sizes: torch.Tensor,
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
        """
        Backward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function.ctx
            Context object to save information for backward computation.
        grad_output : tensor
            The gradient of the loss with respect to the output of the layer.
        grad_x_aug: None
            Empty tensor for torch compatability reasons.
        grad_x_orig: None
            Empty tensor for torch compatability reasons.
        halo_sizes: None
            Empty tensor for torch compatability reasons.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        ctx = cast(ConvTransposeCtx, ctx)
        x_aug, weight = ctx.saved_tensors
        empty = ctx.empty

        stride, padding, dilation = (ctx.stride, ctx.padding, ctx.dilation)
        spatial_group, channel_group = (
            ctx.spatial_group,
            ctx.channel_group,
        )
        selector = ctx.selector
        in_channels_global = ctx.in_channels_global

        g_x, g_w = selector.dist_conv_transpose2d_backward_fn(
            grad_output.to(weight.dtype),
            x_aug,
            weight,
            halo_sizes,
            in_channels_global,
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
        info: Any,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        kernel_size: tuple[int | int],
        stride: tuple[int | int],
        padding: tuple[int | int],
        dilation: tuple[int | int],
        in_channels: int,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        device: torch.device,
        parallelism: int,
        empty: torch.Tensor,
        bayesian: bool,
        selector: ConvTransposeSelectionStrategy,
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
            spatial_group,
            channel_group,
            device,
            parallelism,
            empty,
            bayesian,
            selector,
        )

        return out, 0
