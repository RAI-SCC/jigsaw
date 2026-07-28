"""Config class for ConvTranspose module."""

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import jigsaw.comm as comm

from . import dist_conv_transpose as dist_conv


class ConvTransposeConfig:
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
        Initialize ConvTransposeConfig.

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
