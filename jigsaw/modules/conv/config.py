from typing import Callable, Optional, Any

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from . import distributed_convolutions as dist_conv
import jigsaw.comm as comm
from torch.nn.grad import conv2d_input, conv2d_weight


class ConvConfig:
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