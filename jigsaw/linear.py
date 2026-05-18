from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa

from . import distributed_matmul as dist_mm
from . import distributed_roll as dist_roll
from . import hooks
from .utils_aggregate import gather_1d_results, gather_results


class RollSelectionStrategy:
    """Selection strategy for distributed roll operation."""

    def __init__(self, parallelism: int):
        """Initialize RollSelectionStrategy."""
        self.roll = self._roll_kernel_selection(parallelism)

    def _roll_kernel_selection(self, parallelism: int) -> Callable:
        """Return roll kernel based on parallelism."""
        if parallelism == 4:
            return dist_roll.roll2d
        elif parallelism == 8:
            return dist_roll.roll2d_8way


class TransposeSelectionStrategy:
    """Selection strategy for distributed transpose operation."""

    def __init__(self, parallelism: int):
        self.x_t = self._transpose_kernel_selection(parallelism)

    def _transpose_kernel_selection(self, parallelism: int) -> Callable:
        """Return x_t kernel."""
        if parallelism == 2:
            return dist_mm.x_transpose_twoway
        elif parallelism == 4:
            return dist_mm.x_transpose_fourway
        elif parallelism == 8:
            return dist_mm.x_transpose_eightway


class LinearSelectionStrategy:
    """Selection strategy for distributed linear operation."""

    def __init__(
        self,
        parallelism: int,
        bias: torch.Tensor,
        spatial_group: dist.ProcessGroup,
        bayesian: bool,
        synchronized_layer: bool,
        linear_layer=True,
    ):
        self.mm_kernels = self._mm_kernel_selection(parallelism)
        self.grad_bias = self._grad_bias_selection(
            parallelism, bias, synchronized_layer, bayesian
        )
        self.add_bias = self._add_bias_selection(bias, bayesian)
        self.reshape_weights = self._reshape_weights_selection(bayesian, linear_layer)
        self.grad_sum = self._sum_grad_weights_selection(linear_layer)

    # reshape weights
    def _reshape_weights_selection(
        self, bayesian: bool, linear_layer: bool
    ) -> Callable:
        """
        Selector function to decide if .

        Parameters
        ----------
        bayesian: bool
            Flag to see if the weight reshape is necessary.
        linear_layer: bool
            Flag to see if the module has weights.
        """
        # If the model is Bayesian and the module is a linear layer,
        # weight reshape will be necessary.
        if linear_layer and bayesian:
            return self._reshape_weights
        # If it is a distributed matrix operation with no explicit
        # weight sampling, weight reshape is not needed.
        else:
            return self._no_reshape_weights

    def _reshape_weights(self, w: torch.Tensor) -> torch.Tensor:
        """
        Reshape weights for vmap by unsqueezing in the batch dimension.

        Parameters
        ----------
        w: torch.tensor
            model weights.
        """
        w = w.unsqueeze(1)
        return w

    def _no_reshape_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Return weights as-is. Dummy identity function."""
        return w

    def _sum_grad_weights_selection(self, linear_layer: bool) -> Callable:
        """Determine if summation over batch dimension is necessary."""
        if linear_layer:
            return self._grad_sum
        else:
            return self._no_grad_sum

    def _no_grad_sum(
        self, grad_weight: torch.Tensor, batch_dim: torch.Tensor
    ) -> torch.Tensor:
        """Identity function to return gradient weights."""
        return grad_weight

    def _grad_sum(
        self, grad_weight: torch.Tensor, batch_dim: torch.Tensor
    ) -> torch.Tensor:
        """Summation over gradient weights over the batch dimension."""
        return grad_weight.sum(dim=int(batch_dim))

    # Adding xwt + b
    def _add_bias_selection(
        self, bias: Union[torch.Tensor, None], bayesian: bool
    ) -> Callable:
        """
        Selector function to add bias to weights.

        If the bias is None, return identity.
        If the model is bayesian, perform associated bias reshape.
        Else, if bias exists, add bias.

        Parameters
        ----------
        bias: torch.Tensor
            Value of the bias term. Not the boolean flag.
        bayesian: bool
            Flag to indicate if the model is bayesian.
        """
        if bias is None:
            return self._return_identity
        elif bayesian:
            return self._add_bias_bayesian
        else:
            return self._add_bias

    def _add_bias_bayesian(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Perform reshape for bias and add to x."""
        b = b.unsqueeze(1).unsqueeze(1)
        return x.add(b)

    def _add_bias(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add bias to x."""
        return x.add(b)

    def _return_identity(self, x, b: None = None) -> torch.Tensor:
        """Return x as identity since bias does not exist."""
        return x

    # Reduce bias terms over spatial groups
    def _grad_bias_selection(
        self,
        parallelism: int,
        bias: Union[torch.Tensor, None],
        synchronized_layer: bool,
        bayesian: bool,
    ) -> Callable:
        """
        Selector function for calculating gradient of bias.

        Returns None when bias term is None.
        If parallelism is 2, all model instances hold different and independent bias parameters.
        Otherwise, reduction over the spatial groups is necessary, since spatial groups will "share" output channels.
        If we have a synchronized layer in the case of SWIN, the bias synchronization will also need to occur across spatial groups.
        """
        # return identity if bias is None
        if bias is None:
            return self._no_bias
        # all_reduce bias over space if it is a synchronized linear layer
        elif synchronized_layer and not bayesian:
            return self._reduce_bias_over_spatial_group_nonbayesian
        elif synchronized_layer:
            return self._reduce_bias_over_spatial_group
        # all_reduce is not required for 2-way parallel models
        elif parallelism == 2:
            return self._no_reduce_bias
        # sync required for 4-or-more-way distributed models
        else:
            return self._reduce_bias_over_spatial_group

    def _reduce_bias_over_spatial_group(
        self, grad_output: torch.Tensor, spatial_group: dist.ProcessGroup
    ) -> torch.Tensor:
        """
        Calculate grad wrt bias.

        grad_b is equal to the gradient output summed over space and batch dim.

        Parameters
        ----------
        grad_output: torch.Tensor
            gradient wrt loss
        spatial_group: dist.ProcessGroup
            process group of ranks distributed across space (share channels).
        """
        grad_b = grad_output.sum(dim=(-3, -2))
        return grad_b

    def _reduce_bias_over_spatial_group_nonbayesian(
        self, grad_output: torch.Tensor, spatial_group: dist.ProcessGroup
    ) -> torch.Tensor:
        """
        Calculate grad wrt bias.

        grad_b is equal to the gradient output summed over space and batch dim.

        Parameters
        ----------
        grad_output: torch.Tensor
            gradient wrt loss
        spatial_group: dist.ProcessGroup
            process group of ranks distributed across space (share channels).
        """
        if len(grad_output.shape) == 3:
            grad_b = grad_output.sum(dim=(-3, -2))
        else:
            grad_b = grad_output.sum(dim=(-2))
        return grad_b

    def _no_reduce_bias(
        self, grad_output: torch.Tensor, spatial_group: dist.ProcessGroup = None
    ) -> torch.Tensor:
        """Calculate grad wrt bias when spatial synchronization is not required."""
        return grad_output.sum(dim=(-3, -2))

    def _no_bias(self, grad_output=None, spatial_group=None) -> None:
        """Return identity when bias does not exist."""
        return None

    def _mm_kernel_selection(self, parallelism: int) -> Callable:
        """
        Select distriuted matrix multiplication kernels.

        Depends on number of model-parallel processes.


        Returns
        -------
        List[Callable, Callable, Callable]: pointers to xwt, xw, and xtw kernels.
        """
        xw_t_kernel = self._select_xw_transpose_kernel(parallelism)
        xw_kernel = self._select_xw_kernel(parallelism)
        x_t_w_kernel = self._select_x_transpose_w_kernel(parallelism)
        xw_xtw_fused_kernel = self._select_xw_xtw_fused_kernel(parallelism)

        return [xw_t_kernel, xw_kernel, x_t_w_kernel, xw_xtw_fused_kernel]

    def _select_xw_transpose_kernel(self, parallelism: int) -> Callable:
        """Return xw_t kernel based on parallelism."""
        match parallelism:
            case 2:
                return dist_mm.xw_transpose_twoway
            case 4:
                return dist_mm.xw_transpose_fourway
            case 8:
                return dist_mm.xw_transpose_eightway

    def _select_xw_kernel(self, parallelism: int) -> Callable:
        """Return xw kernel based on parallelism."""
        match parallelism:
            case 2:
                return dist_mm.xw_twoway
            case 4:
                return dist_mm.xw_fourway
            case 8:
                return dist_mm.xw_eightway

    def _select_x_transpose_w_kernel(self, parallelism: int) -> Callable:
        """Return x_tw kernel based on parallelism."""
        match parallelism:
            case 2:
                return dist_mm.x_transpose_w_twoway
            case 4:
                return dist_mm.x_transpose_w_fourway
            case 8:
                return dist_mm.x_transpose_w_eightway

    def _select_xw_xtw_fused_kernel(self, parallelism: int) -> Callable:
        match parallelism:
            case 2:
                return dist_mm.fused_xw_xtw_twoway
            case 4:
                return dist_mm.fused_xw_xtw_fourway


class DistributedXWT(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup = None,
        bayesian: bool = False,
    ) -> None:
        """
        Initialize the XWT Linear layer sharded over n processes.

        Parameters
        ----------
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        local_rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes the model is distributed over
        model_group: dist.ProcessGroup
            Process group containing all ranks within model-parallel instances
        spatial_group: dist.ProcessGroup
            Process group containing ranks that share channels.
        bayesian: bool
            Boolean flag for whether the model is bayesian
        """
        super().__init__()

        stdev = 1.0 / np.sqrt(in_dim * 2)
        self.weight = torch.nn.Parameter(
            torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev).to(device)
        )

        match parallelism:
            case 2:
                out_dim = out_dim // 2
            case 4:
                out_dim = out_dim
            case 8:
                out_dim = out_dim * 2

        self.bias = (
            torch.nn.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev).to(device))
            if bias
            else None
        )

        self.out_dim = out_dim
        self.in_dim = in_dim
        self.device = device
        self.local_rank = local_rank
        self.parallelism = parallelism
        self.model_group = model_group
        self.spatial_group = spatial_group
        self.bayesian = bayesian
        self.selector = LinearSelectionStrategy(
            parallelism,
            self.bias,
            self.spatial_group,
            self.bayesian,
            synchronized_layer=False,
            linear_layer=True,
        )

        if self.bias is not None:
            self.bias.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.spatial_group)
            )
            for p in self.parameters():
                dist.broadcast(p.data, group_src=0, group=self.spatial_group)

        self.XWT = XWT.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim) or (S, B, x_in, in_dim).
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
            B, out_dim are global values.
            in_dim is local value.
        """
        cij = self.XWT(
            x,
            self.weight,
            self.bias,
            self.local_rank,
            self.device,
            self.selector,
            self.model_group,
            self.spatial_group,
            self.bayesian,
        )
        return cij

    def get_parameter_count(self):
        """Return parameter count for linear layer."""
        num_param = (
            0
            if self.bias is None
            else self.bias.numel() // dist.get_world_size(group=self.spatial_group)
        )
        return num_param + self.weight.numel()

    def merge_weights(self, from_process_mesh: dist.ProcessGroup):
        """Gathers and returns global weights (and bias) without modifying self."""
        global_weights = gather_results(self.weight, from_process_mesh)

        global_bias = None
        if self.bias is not None:
            global_bias = gather_1d_results(
                self.bias, from_process_mesh, dim=0, pm_key="col"
            )

        return global_weights, global_bias

    def reshard_parameters(
        self,
        from_process_mesh: dist.ProcessGroup,
        to_process_mesh: dist.ProcessGroup,
        pm_row_key="row",
        pm_col_key="col",
    ):
        """from_process_mesh has process mesh with 'row' and 'col' flags."""
        # re-define self selector
        self.parallelism = to_process_mesh["row"].size() * to_process_mesh["col"].size()
        self.selector = LinearSelectionStrategy(
            self.parallelism,
            self.bias,
            self.spatial_group,
            self.bayesian,
            synchronized_layer=False,
            linear_layer=True,
        )

        # Find the local and global shapes
        from_weight_shape_local = self.weight.shape
        weight_shape_global = (
            from_weight_shape_local[0] * from_process_mesh[pm_row_key].size(),
            from_weight_shape_local[1] * from_process_mesh[pm_col_key].size(),
        )
        to_weight_shape_local = (
            weight_shape_global[0] // to_process_mesh[pm_row_key].size(),
            weight_shape_global[1] // to_process_mesh[pm_col_key].size(),
        )
        assert weight_shape_global[0] % to_process_mesh[pm_row_key].size() == 0, (
            f"global weight shape[0] is {weight_shape_global[0]} which is not divisible by {to_process_mesh[pm_row_key].size()}"
        )
        assert weight_shape_global[1] % to_process_mesh[pm_col_key].size() == 0, (
            f"global weight shape[1] is {weight_shape_global[1]} which is not divisible by {to_process_mesh[pm_col_key].size()}"
        )

        if self.bias is not None:
            from_bias_shape_local = self.bias.shape
            bias_shape_global = (
                from_bias_shape_local[0] * from_process_mesh[pm_col_key].size()
            )
            to_bias_shape_local = (
                bias_shape_global // to_process_mesh[pm_col_key].size()
            )
            assert bias_shape_global % to_process_mesh[pm_col_key].size() == 0, (
                f"global bias shape is {bias_shape_global} which is not divisible by {to_process_mesh[pm_col_key].size()}"
            )

        # Gather
        global_weights, global_bias = self.merge_weights(from_process_mesh)

        row_rank, row_shard_size = (
            to_process_mesh[pm_row_key].rank(),
            to_weight_shape_local[0],
        )
        col_rank, col_shard_size = (
            to_process_mesh[pm_col_key].rank(),
            to_weight_shape_local[1],
        )

        self.weight = torch.nn.Parameter(
            global_weights[
                row_rank * row_shard_size : (row_rank + 1) * row_shard_size,
                col_rank * col_shard_size : (col_rank + 1) * col_shard_size,
            ].contiguous()
        )
        del global_weights

        if self.bias is not None:
            col_shard_size = to_bias_shape_local
            self.bias = torch.nn.Parameter(
                global_bias[
                    col_rank * col_shard_size : (col_rank + 1) * col_shard_size
                ].contiguous()
            )
            del global_bias


class XWT(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        local_rank: int,
        device: torch.device,
        selector: LinearSelectionStrategy,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> torch.Tensor:
        """
        Forward pass of linear layer (XWT) distributed across parallelism processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim) or (S, B, x_in, in_dim).
        W : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim) or (S, in_dim, out_dim).
        b : torch.Tensor
            Local bias tensor of shape (out_dim) or (S, out_dim).
        local_rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        selector : LinearSelectionStrategy
            Instance of LinearSelectionStrategy class.
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.
        spatial_group: dist.ProcessGroup
            Process group for ranks that share channels and differ across space.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim) or (S, B, out_dim, in_dim).
        """
        w = selector.reshape_weights(w)
        xwt = selector.mm_kernels[0](x, w, local_rank, device, model_group)
        xwt = selector.add_bias(xwt, b)

        return xwt

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, w, b, local_rank, device, selector, spatial_group, bayesian)
        output: xwt
        """
        x, w, b, local_rank, device, selector, model_group, spatial_group, bayesian = (
            inputs
        )
        ctx.save_for_backward(x, w, b)
        ctx.local_rank = local_rank
        ctx.device = device
        ctx.selector = selector
        ctx.model_group = model_group
        ctx.spatial_group = spatial_group
        ctx.bayesian = bayesian

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
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
    ]:
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
        input, weight, bias = ctx.saved_tensors
        local_rank = ctx.local_rank
        device = ctx.device
        selector = ctx.selector
        model_group = ctx.model_group
        spatial_group = ctx.spatial_group

        weight = selector.reshape_weights(weight)

        grad_input, grad_weight = selector.mm_kernels[3](
            grad_output,
            weight,
            input,
            local_rank=local_rank,
            device=device,
            model_group=model_group,
        )

        grad_weight = selector.grad_sum(
            grad_weight, ctx.bayesian
        )  # sum along batch dimension

        grad_b = selector.grad_bias(grad_output, spatial_group)

        return grad_input, grad_weight, grad_b, None, None, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Union[torch.Tensor, None],
        local_rank: int,
        device: torch.device,
        selector: LinearSelectionStrategy,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XWT behavior under torch.vmap.

        Parameters
        ----------
        info :
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims :
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, rank, device, selector, model_group, spatial_group, bayesian).
        x : Tensor
            Possibly batched input.
        w : Tensor
            Possibly shared or batched weights.
        b : Tensor or None
            Possibly shared or batched bias.

        Returns
        -------
        output : Tensor
            Batched output with vmapped dimension in front.
        out_dim : Optional[int]
            Index of vmapped dimension in output (0 if batched, None if not).
        """
        x = x.contiguous()
        out = XWT.apply(
            x, w, b, local_rank, device, selector, model_group, spatial_group, bayesian
        )

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedXWTSync(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        sync_group: dist.ProcessGroup,  # Equivalent to spatial group in distributed Windowed attention
        bayesian: bool = False,
    ) -> None:
        """
        Initialize the XWT Linear layer sharded over n processes.

        Parameters
        ----------
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes it is distributed over
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.
        spatial_group: dist.ProcessGroup
            Process group for ranks that share channels and differ across space.
        bayesian : bool
            Flag for whether the model is Bayesian.
        """
        super().__init__()

        stdev = (
            1.0 / np.sqrt(in_dim * 2)
            if parallelism > 1
            else 1.0 / np.sqrt(in_dim)  # TODO CHECK
        )
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev).to(device)
        )

        match parallelism:
            case 2:
                out_dim = out_dim // 2
            case 4:
                out_dim = out_dim
            case 8:
                out_dim = out_dim * 2

        self.bias = (
            torch.nn.parameter.Parameter(
                torch.zeros(out_dim).uniform_(-stdev, stdev).to(device)
            )
            if bias
            else None
        )

        self.device = device
        self.local_rank = local_rank
        self.model_group = model_group
        self.sync_group = sync_group
        self.bayesian = bayesian
        self.selector = LinearSelectionStrategy(
            parallelism,
            self.bias,
            self.model_group,
            self.bayesian,
            synchronized_layer=True,
            linear_layer=True,
        )

        self.weight.register_post_accumulate_grad_hook(
            hooks.make_grad_sync_hook(self.sync_group)
        )
        if self.bias is not None:
            self.bias.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.sync_group)
            )

        for p in self.parameters():
            dist.broadcast(p.data, group_src=0, group=self.sync_group)

        self.XWT_sync = XWTSync.apply

    def get_parameter_count(self):
        """Return parameter count for distributed XWT Sync layer."""
        num_param = (
            0
            if self.bias is None
            else self.bias.numel() // dist.get_world_size(group=self.sync_group)
        )
        return num_param + self.weight.numel() // dist.get_world_size(
            group=self.sync_group
        )

    def reshard_parameters(
        self,
        from_process_mesh: dist.ProcessGroup,
        to_process_mesh: dist.ProcessGroup,
        pm_row_key="row",
        pm_col_key="col",
    ):
        """from_process_mesh has process mesh with 'row' and 'col' flags."""
        # re-define self selector
        self.parallelism = to_process_mesh["model"].size()
        self.selector = LinearSelectionStrategy(
            self.parallelism,
            self.bias,
            self.spatial_group,
            self.bayesian,
            synchronized_layer=False,
            linear_layer=True,
        )

        # Find the local and global shapes
        from_weight_shape_local = self.weight.shape
        weight_shape_global = (
            from_weight_shape_local[0] * from_process_mesh["row"].size(),
            from_weight_shape_local[1] * from_process_mesh["col"].size(),
        )
        to_weight_shape_local = (
            weight_shape_global[0] // to_process_mesh["row"].size(),
            weight_shape_global[1] // to_process_mesh["col"].size(),
        )

        assert weight_shape_global[0] % to_process_mesh["row"].size() == 0, (
            f"global weight shape[0] is {weight_shape_global[0]} which is not divisible by {to_process_mesh['row'].size()}"
        )
        assert weight_shape_global[1] % to_process_mesh["col"].size() == 0, (
            f"global weight shape[1] is {weight_shape_global[1]} which is not divisible by {to_process_mesh['col'].size()}"
        )

        # gather
        global_weights = gather_results(self.weight, from_process_mesh)

        if self.bias is not None:
            from_bias_shape_local = self.bias.shape
            bias_shape_global = (
                from_bias_shape_local[0] * from_process_mesh["col"].size()
            )
            to_bias_shape_local = bias_shape_global // to_process_mesh["col"].size()
            assert bias_shape_global % to_process_mesh["col"].size() == 0, (
                f"global bias shape is {bias_shape_global} which is not divisible by {to_process_mesh['col'].size()}"
            )

            global_bias = gather_1d_results(
                self.bias, from_process_mesh, dim=0, pm_key="col"
            )

        row_rank, row_shard_size = (
            to_process_mesh["row"].rank(),
            to_weight_shape_local[0],
        )
        col_rank, col_shard_size = (
            to_process_mesh["col"].rank(),
            to_weight_shape_local[1],
        )
        parameter_shard = global_weights[
            row_rank * row_shard_size : (row_rank + 1) * row_shard_size,
            col_rank * col_shard_size : (col_rank + 1) * col_shard_size,
        ].contiguous()

        self.weight = torch.nn.Parameter(parameter_shard)
        del global_weights

        if self.bias is not None:
            col_rank, col_shard_size = (
                to_process_mesh["col"].rank(),
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
            Local input tensor of shape (B, x_in, in_dim) or (S, B, x_in, in_dim).
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim) or (S, B, out_dim, in_dim).
        """
        cij = self.XWT_sync(
            x,
            self.weight,
            self.bias,
            self.local_rank,
            self.device,
            self.selector,
            self.model_group,
            self.sync_group,
            self.bayesian,
        )
        return cij


class XWTSync(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        local_rank: int,
        device: torch.device,
        selector: LinearSelectionStrategy,
        model_group: dist.ProcessGroup,
        sync_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> torch.Tensor:
        """
        Forward pass of distributed linear layer (XWT) synchronzied across domain-but-not-model-parallel processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim) or (S, B, x_in, in_dim).
        w : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim) or (S, in_dim, out_dim).
        b : torch.Tensor
            Local bias tensor of shape (out_dim) or (S, out_dim).
        rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        selector : LinearSelectionStrategy
            Instance of linear selection strategy class
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.
        spatial_group: dist.ProcessGroup
            Process group for ranks that share channels and differ across space.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
        """
        w = selector.reshape_weights(w)
        xwt = selector.mm_kernels[0](x, w, local_rank, device, model_group)
        xwt = selector.add_bias(xwt, b)

        return xwt

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, w, b, local_rank, device, selector, model_group, sync_group, bayesian)
        output: xwt
        """
        x, w, b, local_rank, device, selector, model_group, sync_group, bayesian = (
            inputs
        )
        ctx.save_for_backward(x, w, b)
        ctx.local_rank = local_rank
        ctx.device = device
        ctx.selector = selector
        ctx.model_group = model_group
        ctx.sync_group = sync_group
        ctx.bayesian = bayesian

        # infer dtype from the input tensors rather than from autocast state
        ctx._fwd_dtype = x.dtype
        ctx._fwd_used_autocast = x.is_floating_point() and x.dtype != torch.float32

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, None],
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
        input, weight, bias = ctx.saved_tensors
        local_rank = ctx.local_rank
        device = ctx.device
        mm_kernels = ctx.selector.mm_kernels
        model_group = ctx.model_group

        weight = ctx.selector.reshape_weights(weight)

        with torch.autocast("cuda", dtype=grad_output.dtype, enabled=True):
            grad_input, grad_weight = mm_kernels[3](
                grad_output,
                weight,
                input,
                local_rank=local_rank,
                device=device,
                model_group=model_group,
            )

            grad_b = ctx.selector.grad_bias(grad_output, ctx.sync_group)

            # Sum over batch dimension.
            # Batch dim = 0 if not bayesian, = 1 if bayesian

            if ctx.bayesian:
                grad_weight = grad_weight.sum(dim=int(ctx.bayesian))
            elif len(grad_weight.shape) > len(weight.shape):
                grad_weight = grad_weight.sum(dim=int(ctx.bayesian))

        return grad_input, grad_weight, grad_b, None, None, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        local_rank: int,
        device: torch.device,
        selector: LinearSelectionStrategy,
        model_group: dist.ProcessGroup,
        sync_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XWT behavior under torch.vmap.

        Parameters
        ----------
        info :
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims :
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, local_rank, device, parallelism, model_group).
        x : Tensor
            Possibly batched input.
        w : Tensor
            Possibly shared or batched weights.
        b : Tensor or None
            Possibly shared or batched bias.

        Returns
        -------
        output : Tensor
            Batched output with vmapped dimension in front.
        out_dim : Optional[int]
            Index of vmapped dimension in output (0 if batched, None if not).
        """
        # x under vmap has shape (S, B, x_in, in_dim).
        x = x.contiguous()

        out = XWTSync.apply(
            x, w, b, local_rank, device, selector, model_group, sync_group, bayesian
        )

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedXW(torch.nn.Module):
    """Batched matrix-matrix (XW) multiplication distributed across two processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool = False,
    ) -> None:
        """
        Initialize distributed matrix-matrix multiplication module.

        in_dim: int
            Global input dimension of local weight matrix.
        out_dim: int
            Local output dimension of local weight matrix.
        bias: bool
            Boolean flag for bias.
        local_rank : int
            Local model rank.
        device: torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes the operation is distributed over.
        model_group: dist.ProcessGroup
        bayesian:
        """
        super().__init__()

        stdev = 1.0 / np.sqrt(in_dim * 2) if parallelism == 4 else 1.0 / np.sqrt(in_dim)

        self.weight = torch.nn.parameter.Parameter(
            torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev).to(device)
        )
        self.bias = (
            torch.zeros(out_dim).uniform_(-stdev, stdev).to(device) if bias else None
        )

        self.local_rank = local_rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.bayesian = bayesian
        self.selector = LinearSelectionStrategy(
            parallelism,
            self.bias,
            self.model_group,
            self.bayesian,
            synchronized_layer=False,
            linear_layer=True,
        )

        if self.bias is not None:
            self.bias = torch.nn.parameter.Parameter(self.bias)
            self.bias.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.spatial_group)
            )
            for p in self.parameters():
                dist.broadcast(p.data, group_src=0, group=self.spatial_group)

        self.XW = XW.apply

    def get_parameter_count(self):
        """Return parameter count for distributed XW layer."""
        num_param = (
            0
            if self.bias is None
            else self.bias.numel() // dist.get_world_size(group=self.spatial_group)
        )
        return num_param + self.weight.numel()

    def reshard_parameters(
        self,
        from_process_mesh: dist.ProcessGroup,
        to_process_mesh: dist.ProcessGroup,
        pm_row_key="row",
        pm_col_key="col",
    ):
        """from_process_mesh has process mesh with 'row' and 'col' flags."""
        # re-define self selector
        self.parallelism = to_process_mesh["model"].size()
        self.selector = LinearSelectionStrategy(
            self.parallelism,
            self.bias,
            self.spatial_group,
            self.bayesian,
            synchronized_layer=False,
            linear_layer=True,
        )

        # Find the local and global shapes
        from_weight_shape_local = self.weight.shape
        weight_shape_global = (
            from_weight_shape_local[0] * from_process_mesh["row"].size(),
            from_weight_shape_local[1] * from_process_mesh["col"].size(),
        )
        to_weight_shape_local = (
            weight_shape_global[0] // to_process_mesh["row"].size(),
            weight_shape_global[1] // to_process_mesh["col"].size(),
        )

        assert weight_shape_global[0] % to_process_mesh["row"].size() == 0, (
            f"global weight shape[0] is {weight_shape_global[0]} which is not divisible by {to_process_mesh['row'].size()}"
        )
        assert weight_shape_global[1] % to_process_mesh["col"].size() == 0, (
            f"global weight shape[1] is {weight_shape_global[1]} which is not divisible by {to_process_mesh['col'].size()}"
        )

        # gather
        global_weights = gather_results(self.weight, from_process_mesh)

        if self.bias is not None:
            from_bias_shape_local = self.bias.shape
            bias_shape_global = (
                from_bias_shape_local[0] * from_process_mesh["col"].size()
            )
            to_bias_shape_local = bias_shape_global // to_process_mesh["col"].size()
            assert bias_shape_global % to_process_mesh["col"].size() == 0, (
                f"global bias shape is {bias_shape_global} which is not divisible by {to_process_mesh['col'].size()}"
            )

            global_bias = gather_1d_results(
                self.bias, from_process_mesh, dim=0, pm_key="col"
            )

        row_rank, row_shard_size = (
            to_process_mesh["row"].rank(),
            to_weight_shape_local[0],
        )
        col_rank, col_shard_size = (
            to_process_mesh["col"].rank(),
            to_weight_shape_local[1],
        )
        parameter_shard = global_weights[
            row_rank * row_shard_size : (row_rank + 1) * row_shard_size,
            col_rank * col_shard_size : (col_rank + 1) * col_shard_size,
        ].contiguous()

        self.weight = torch.nn.Parameter(parameter_shard)
        del global_weights

        if self.bias is not None:
            col_rank, col_shard_size = (
                to_process_mesh["col"].rank(),
                to_bias_shape_local,
            )
            self.bias = torch.nn.Parameter(
                global_bias[
                    col_rank * col_shard_size : (col_rank + 1) * col_shard_size
                ].contiguous()
            )
            del global_bias

    def forward(self, x: torch.Tensor):
        """
        Forward pass of linear layer (XW) distributed across two processes.

        x: torch.Tensor
            local input tensor of shape (B, x_in, x_out)
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        C: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim).
            B, x_in are global values.
            out_dim is local value.
        """
        cij = self.XW(
            x,
            self.weight,
            self.bias,
            self.local_rank,
            self.device,
            self.selector,
            self.model_group,
            self.bayesian,
        )
        return cij


class XW(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XW) multiplication distributed across two processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        local_rank: int,
        device: torch.device,
        selector: int,
        model_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> torch.Tensor:
        """
        Forward pass of linear layer (XW) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
            B, x_in are global dimensions.
            in_dim is local dimension.
        w : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim).
            in_dim is global dimension.
        b : torch.Tensor
            Local bias tensor of shape (out_dim)
        local_rank : int
            Local model rank
        device : torch.device
            Device to perform computation on.
        selector: LinearSelectionStrategy
            Instance of LinearSelectionStrategy class.
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        bayesian : bool
            Flag to indicate if the model is bayesian.

        Returns
        -------
        C: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim) or (S, B, x_in, out_dim).
        """
        w = selector.reshape_weights(w)
        xw = selector.mm_kernels[1](x, w, local_rank, device, model_group)
        xw = selector.add_bias(xw, b)

        return xw

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, w, b, local_rank, device, parallelism, model_group)
        output: xwt
        """
        x, w, b, local_rank, device, selector, model_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        ctx.local_rank = local_rank
        ctx.device = device
        ctx.selector = selector
        ctx.model_group = model_group
        ctx.b = b
        ctx.bayesian = bayesian

        ctx._fwd_dtype = w.dtype
        ctx._fwd_used_autocast = x.is_floating_point() and w.dtype != torch.float32

    @staticmethod
    def backward(ctx: torch.autograd.Function, grad_output: torch.Tensor):
        """
        Backward pass of linear layer (XW) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        input, weight, bias = ctx.saved_tensors
        local_rank = ctx.local_rank
        device = ctx.device
        selector = ctx.selector
        model_group = ctx.model_group

        weight = selector.reshape_weights(weight)
        with torch.autocast(
            "cuda", dtype=ctx._fwd_dtype, enabled=ctx._fwd_used_autocast
        ):
            grad_input = selector.mm_kernels[0](
                grad_output,
                weight,
                local_rank=local_rank,
                device=device,
                model_group=model_group,
            )
            grad_weight = selector.mm_kernels[2](
                input,
                grad_output,
                local_rank=local_rank,
                device=device,
                model_group=model_group,
            )

            grad_weight = selector.grad_sum(grad_weight, ctx.bayesian)
            grad_b = selector.grad_bias(grad_output)

        return grad_input, grad_weight, grad_b, None, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        local_rank: int,
        device: torch.device,
        selector: int,
        model_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XWT behavior under torch.vmap.

        Parameters
        ----------
        info :
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims :
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, local_rank, device, parallelism, model_group).
        x : Tensor
            Batched input.
        w : Tensor
            Batched weights.
        b : Tensor or None
            Batched bias.

        Returns
        -------
        output : Tensor
            Batched output with vmapped dimension in front.
        out_dim : Optional[int]
            Index of vmapped dimension in output (0 if batched, None if not).
        """
        x = x.contiguous()
        out = XW.apply(x, w, b, local_rank, device, selector, model_group, bayesian)

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedXTW(torch.nn.Module):
    """Batched matrix-matrix (XTW) linear layer distributed across n processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool = False,
    ) -> None:
        """
        Initialize distributed XTW module.

        inputs
        ------
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        bias : bool, optional
            Whether to include a bias term. Default is True.
        local_rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism : int
            Number of processes to distribute over
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        bayesian : bool
            Flag to indicate if the model is bayesian.
        """
        super().__init__()

        stdev = 1.0 / np.sqrt(in_dim * 2) if parallelism == 4 else 1.0 / np.sqrt(in_dim)
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev).to(device)
        )

        self.bias = (
            torch.zeros(out_dim).uniform_(-stdev, stdev).to(device) if bias else None
        )

        self.local_rank = local_rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.bayesian = bayesian
        self.selector = LinearSelectionStrategy(
            parallelism,
            self.bias,
            self.model_group,
            self.bayesian,
            synchronized_layer=False,
            linear_layer=True,
        )

        if self.bias is not None:
            self.bias = torch.nn.parameter.Parameter(self.bias)
            self.bias.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.spatial_group)
            )
            for p in self.parameters():
                dist.broadcast(p.data, group_src=0, group=self.spatial_group)

        self.XTW = XTW.apply

    def get_parameter_count(self):
        """Return parameter count for distributed XTW linear layer."""
        num_param = (
            0
            if self.bias is None
            else self.bias.numel() // dist.get_world_size(group=self.spatial_group)
        )
        return num_param + self.weight.numel()

    def reshard_parameters(
        self, from_process_mesh: dist.ProcessGroup, to_process_mesh: dist.ProcessGroup
    ):
        """from_process_mesh has process mesh with 'row' and 'col' flags."""
        # re-define self selector
        self.parallelism = to_process_mesh["model"].size()
        self.selector = LinearSelectionStrategy(
            self.parallelism,
            self.bias,
            self.spatial_group,
            self.bayesian,
            synchronized_layer=False,
            linear_layer=True,
        )

        # Find the local and global shapes
        from_weight_shape_local = self.weight.shape
        weight_shape_global = (
            from_weight_shape_local[0] * from_process_mesh["row"].size(),
            from_weight_shape_local[1] * from_process_mesh["row"].size(),
        )
        to_weight_shape_local = (
            weight_shape_global // to_process_mesh["row"].size(),
            weight_shape_global // to_process_mesh["col"].size(),
        )

        assert weight_shape_global[0] % to_process_mesh["row"].size() == 0, (
            f"global weight shape[0] is {weight_shape_global[0]} which is not divisible by {to_process_mesh['row'].size()}"
        )
        assert weight_shape_global[1] % to_process_mesh["col"].size() == 0, (
            f"global weight shape[1] is {weight_shape_global[1]} which is not divisible by {to_process_mesh['col'].size()}"
        )

        # gather
        global_weights = gather_results(self.weight, from_process_mesh)

        if self.bias is not None:
            from_bias_shape_local = self.bias.shape
            bias_shape_global = (
                from_bias_shape_local[0] * from_process_mesh["row"].size()
            )
            to_bias_shape_local = bias_shape_global // to_process_mesh["row"].size()

            assert bias_shape_global % to_process_mesh["row"] == 0, (
                f"global bias shape is {bias_shape_global} which is not divisible by {to_process_mesh['row'].size()}"
            )

            global_bias = gather_results(
                self.bias, from_process_mesh
            )  # keep it simple, gather all around

        row_rank, row_shard_size = (
            to_process_mesh["row"].size(),
            to_weight_shape_local[0],
        )
        col_rank, col_shard_size = (
            to_process_mesh["col"].size(),
            to_weight_shape_local[1],
        )

        self.weight = torch.nn.Parameter(
            global_weights[
                row_rank * row_shard_size : (row_rank + 1) * row_shard_size,
                col_rank * col_shard_size : (col_rank + 1) * col_shard_size,
            ].contiguous()
        )
        del global_weights

        col_rank, col_shard_size = to_process_mesh["col"].size(), to_bias_shape_local[1]

        if self.bias is not None:
            torch.nn.Parameter(
                global_bias[
                    col_rank * col_shard_size : (col_rank + 1) * col_shard_size
                ].contiguous()
            )
            del global_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear layer (XTW) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim) or (S, B, x_in, in_dim).

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim).
            B, x_in are global values.
            out_dim is local value.
        """
        cij = self.XTW(
            x,
            self.weight,
            self.bias,
            self.local_rank,
            self.device,
            self.selector,
            self.model_group,
            self.bayesian,
        )

        return cij


class XTW(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XTW) multiplication distributed across two processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        local_rank: int,
        device: torch.device,
        selector: int,
        model_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> torch.Tensor:
        """
        Forward pass of distributed linear layer (XTW).

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim) or (S, x_in, in_dim).
        w : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim) or (S, in_dim, out_dim).
        b : torch.Tensor
            Local bias tensor of shape (out_dim) or (S, in_dim, out_dim).
            Bias vector.
        local_rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.
        bayesian: bool
            Flag to indicate if the model is bayesian.

        Returns
        -------
        xtw: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim) or (S, B, x_in, out_dim).
        """
        w = selector.reshape_weights(w)
        xtw = selector.mm_kernels[2](x, w, local_rank, device, model_group)
        xtw = selector.add_bias(xtw, b)

        return xtw

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, w, b, local_rank, device, selector, model_group, bayesian)
        output: xwt
        """
        x, w, b, local_rank, device, selector, model_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        ctx.local_rank = local_rank
        ctx.device = device
        ctx.selector = selector
        ctx.model_group = model_group
        ctx.b = b
        ctx.bayesian = bayesian

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None, None
    ]:
        """
        Backward pass of linear layer (XTW) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : torch.Tensor
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
        input, weight, bias = ctx.saved_tensors
        local_rank = ctx.local_rank
        device = ctx.device
        selector = ctx.selector
        model_group = ctx.model_group

        weight = selector.reshape_weights(weight)

        grad_input = selector.mm_kernels[0](
            weight, grad_output, local_rank, device, model_group
        )
        grad_weight = selector.mm_kernels[1](
            input, grad_output, local_rank, device, model_group
        )

        grad_weight = selector.grad_sum(grad_weight, ctx.bayesian)
        grad_b = selector.grad_bias(grad_output)

        return grad_input, grad_weight, grad_b, None, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        local_rank: int,
        device: torch.device,
        selector: int,
        model_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XWT behavior under torch.vmap.

        Parameters
        ----------
        info :
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims :
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, local_rank, device, parallelism, model_group).
        x : Tensor
            Possibly batched input.
        w : Tensor
            Possibly shared or batched weights.
        b : Tensor or None
            Possibly shared or batched bias.

        Returns
        -------
        output : Tensor
            Batched output with vmapped dimension in front.
        out_dim : Optional[int]
            Index of vmapped dimension in output (0 if batched, None if not).
        """
        # x under vmap has shape (V, B, x_in, in_dim).
        x = x.contiguous()
        out = XTW.apply(x, w, b, local_rank, device, selector, model_group, bayesian)

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedXTranspose(torch.nn.Module):
    """Batched transpose (XT) operation distributed across n processes."""

    def __init__(
        self,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
    ) -> None:
        """
        Initialize the X Transpose class.

        Parameters
        ----------
        local_rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes to distribute over
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.selector = TransposeSelectionStrategy(self.parallelism)

        self.transpose = XT.apply

    def get_parameter_count(self):
        """Return 0 parameter count for transpose operation."""
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transpose (XT) distributed across two processes.

        Parameters.
        ----------

        X : torch.Tensor
            The input tensor.

        Returns
        -------
        XT: torch.Tensor
            Local transposed tensor.
        """
        x_transpose = self.transpose(
            x, self.local_rank, self.device, self.parallelism, self.model_group
        )
        return x_transpose


class XT(torch.autograd.Function):
    """Autograd function for batched transpose (XT) operation distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        local_rank: int,
        device: torch.device,
        selector: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Forward pass of transpose (XT) distributed across n processes.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        local_rank : int
            Local model rank.
        device : torch.device
            The device on which the computation is performed.
        selector: TransposeSelectionStrategy
            Instance of transpose selection strategy class
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        XT: torch.Tensor
            Local transposed tensor.
        """
        x = selector.x_t(x, local_rank, device, model_group)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, local_rank, device, selector, model_group)
        output: xwt
        """
        x, local_rank, device, selector, model_group = inputs
        ctx.device = device
        ctx.local_rank = local_rank
        ctx.selector = selector
        ctx.model_group = model_group

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        """
        Backward pass of transpose (XT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : tensor
            Gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_output:
            Transposed gradient of the loss with respect to the input of the layer.
        """
        local_rank = ctx.local_rank
        device = ctx.device
        selector = ctx.selector
        model_group = ctx.model_group

        grad_output = selector.x_t(grad_output, local_rank, device, model_group)
        return grad_output, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        local_rank: int,
        device: torch.device,
        selector: int,
        model_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Perform vmap for transpose operation."""
        x = x.contiguous()
        # x under vmap has shape (V, B, x_in, in_dim).
        out = XT.apply(x, local_rank, device, selector, model_group)

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedRoll(torch.nn.Module):
    """Batched transpose (XT) operation distributed across n processes."""

    def __init__(
        self,
        shifts: Tuple[int, int],
        dims: int,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        spatial_group: dist.ProcessGroup,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the Roll class."""
        super().__init__()

        self.local_rank = local_rank
        self.device = device
        self.spatial_group = spatial_group
        self.shifts = shifts
        self.dims = dims
        self.roll = Roll.apply
        self.empty = torch.empty(0, device=self.device, dtype=dtype)

        self.selector = RollSelectionStrategy(parallelism)

    def get_parameter_count(self):
        """Return parameter count for distributed roll operation."""
        return 0

    def reshard_parameters(
        self, from_process_mesh: dist.ProcessGroup, to_process_mesh: dist.ProcessGroup
    ):
        """from_process_mesh has process mesh with 'row' and 'col' flags."""
        # re-define self selector
        self.parallelism = to_process_mesh["model"].size()
        self.selector = RollSelectionStrategy(
            self.parallelism,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of distributed roll.

        Function currently only works when rolled across the -2 and -3 dimensions.

        Parameters.
        ----------

        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x_roll: torch.Tensor
            Rolled tensor
        """
        x_roll = self.roll(
            x,
            self.shifts,
            self.dims,
            self.local_rank,
            self.device,
            self.empty,
            self.selector,
            self.spatial_group,
        )
        return x_roll


class Roll(torch.autograd.Function):
    """Autograd function for batched transpose (XT) operation distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        shifts: Tuple[int, int],
        dims: Tuple[int, int],
        local_rank: int,
        device: torch.device,
        empty: torch.Tensor,
        selector: RollSelectionStrategy,
        spatial_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Forward pass of distribued roll across 4 processes.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        shifts: Tuple[int, int]
            Values to shift by
        dims: Tuple[int, int]
            Dimensions in which to perform the rolled operation
        device : torch.device
            The device on which the computation is performed.
        empty: torch.Tensor
            Empty torch tensor used as buffer
        selector: RollSelectionStrategy
            Member of class RollSelectionStrategy that defines function pointer
        spatial_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        x_roll: torch.Tensor
            Local rolled tensor.
        """
        x = selector.roll(x, shifts, dims, local_rank, device, empty, spatial_group)

        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Set up context for distributed roll operation."""
        x, shifts, dims, local_rank, device, empty, selector, spatial_group = inputs

        ctx.shifts = shifts
        ctx.local_rank = local_rank
        ctx.dims = dims
        ctx.device = device
        ctx.empty = empty
        ctx.selector = selector
        ctx.spatial_group = spatial_group

    @staticmethod
    def vmap(
        info,
        in_dims,
        x,
        shifts,
        dims,
        local_rank,
        device,
        empty,
        selector,
        spatial_group,
    ):
        """Perform vmap for distributed roll operation."""
        out = Roll.apply(
            x, shifts, dims, local_rank, device, empty, selector, spatial_group
        )
        return out, 0

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None]:
        """
        Backward pass of transpose (XT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : tensor
            Gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_output:
            Transposed gradient of the loss with respect to the input of the layer.
        """
        shifts = ctx.shifts
        selector = ctx.selector
        local_rank, device = ctx.local_rank, ctx.device
        spatial_group = ctx.spatial_group
        dims = ctx.dims

        shifts = [-shift for shift in shifts]
        grad_output = selector.roll(
            grad_output, shifts, dims, local_rank, device, ctx.empty, spatial_group
        )

        return grad_output, None, None, None, None, None, None, None


class DistributedMatMulXWTranspose(torch.nn.Module):
    """Batched distributed matrix-matrix (XWT) multiplication."""

    def __init__(
        self,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> None:
        """
        Initialize the Distributed XWT multiplication operation.

        Parameters
        ----------
        local_rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes the model is distributed over.
        model_group: dist.ProcessGroup
            Process group containing all ranks within model-parallel instances.
        spatial_group: dist.ProcessGroup
            Process group containing ranks that share channels.
        bayesian: bool
            Boolean flag for whether the model is Bayesian.
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.model_group = model_group
        self.spatial_group = spatial_group
        self.bayesian = bayesian
        self.selector = LinearSelectionStrategy(
            parallelism,
            None,
            self.model_group,
            self.bayesian,
            synchronized_layer=False,
            linear_layer=False,
        )

        self.XWT = XWT.apply

    def get_parameter_count(self):
        """Return parameter count for disdtributed MatMulXWT operation."""
        return 0

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        w : torch.Tensor
            Local weights of shape (out_dim, in_dim) or (S, out_dim, in_dim).
        b : torch Tensor
            Local bias vector of shape (out_dim) or (S, out_dim).

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim) or (S, B, x_in, out_dim).
        """
        cij = self.XWT(
            x,
            w,
            b,
            self.local_rank,
            self.device,
            self.selector,
            self.model_group,
            self.spatial_group,
            self.bayesian,
        )

        return cij


class DistributedMatMulXW(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across two processes."""

    def __init__(
        self,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool,
    ):
        """
        Initialize the distributed XW module.

        Parameters
        ----------
        local_rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes the model is distributed over.
        model_group: dist.ProcessGroup
            Process group containing all ranks within model-parallel instances.
        spatial_group: dist.ProcessGroup
            Process group containing ranks that share channels.
        bayesian: bool
            Boolean flag for whether the model is Bayesian.
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.bayesian = bayesian

        self.selector = LinearSelectionStrategy(
            parallelism,
            None,
            self.model_group,
            self.bayesian,
            synchronized_layer=False,
            linear_layer=False,
        )
        self.XW = XW.apply

    def get_parameter_count(self):
        """Return parameter count for matrix multiplication."""
        return 0

    def forward(self, x, w, b):
        """
        Forward pass of linear layer (XW) distributed across two processes.

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
        """
        cij = self.XW(
            x,
            w,
            b,
            self.local_rank,
            self.device,
            self.selector,
            self.model_group,
            self.bayesian,
        )

        return cij


class DistributedXWTNice(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        device: torch.device,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize the XWT Linear layer sharded over n processes.

        Parameters
        ----------
        parallelism: int
            number of processes it is distributed over
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        stdev : float, optional
            The standard deviation for weight initialization (default is None).
        """
        super().__init__()

        self.channel_group = channel_group
        self.spatial_group = spatial_group

        stdev = 1.0 / np.sqrt(in_dim)

        self.weight = torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev)
        self.bias = torch.zeros(out_dim).uniform_(-stdev, stdev) if bias else None

        self.weight = torch.nn.parameter.Parameter(self.weight)
        self.bias = torch.nn.parameter.Parameter(self.bias) if bias else None

        # TODO: Distributed NICE hook?
        # self.weight.register_post_accumulate_grad_hook(hooks.make_grad_sync_hook(self.spatial_group))

        self.XWT_Nice = XWTNice.apply

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
        cij = self.XWT_Nice(
            x,
            self.weight,
            self.bias,
            self.channel_group,
            self.spatial_group,
        )
        return cij


class XWTNice(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        channel_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        W : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim).
        b : torch.Tensor
            Local bias tensor of shape (out_dim).
        channel_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        spatial_group : dist.ProcessGroup, optional
            Process group for an individual domain-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
            B, out_dim are global values.
            in_dim is local value.
        """
        # Handle bias term here?
        if (b is not None) and (len(b.shape) > 1):
            b = b.reshape(b.shape[0], 1, 1, -1)
        xwt = torch.matmul(x, w.mT) + b

        return xwt

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, parallelism, model_group)
        output: xwt
        """
        x, w, b, channel_group, spatial_group = inputs
        ctx.save_for_backward(x, w)
        ctx.channel_group = channel_group
        ctx.spatial_group = spatial_group
        ctx.has_bias = True if b is not None else False

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

        channel_group = ctx.channel_group

        grad_input = torch.matmul(grad_output, weight)
        g_i_handle = dist.all_reduce(
            grad_input, op=dist.ReduceOp.SUM, group=channel_group, async_op=True
        )

        grad_weight = torch.matmul(grad_output.mT, x)
        if len(grad_weight.shape) > len(weight.shape):
            grad_weight = grad_weight.sum(dim=tuple(range(grad_weight.dim() - 2)))

        grad_bias = (
            grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            if ctx.has_bias
            else None
        )

        g_i_handle.wait()

        return grad_input, grad_weight, grad_bias, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        channel_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XWT behavior under torch.vmap.

        Parameters
        ----------
        info :
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims :
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, rank, device, parallelism, model_group).
        x : Tensor
            Possibly batched input.
        w : Tensor
            Possibly shared or batched weights.
        b : Tensor or None
            Possibly shared or batched bias.

        Returns
        -------
        output : Tensor
            Batched output with vmapped dimension in front.
        out_dim : Optional[int]
            Index of vmapped dimension in output (0 if batched, None if not).
        """
        # x under vmap has shape (V, B, x_in, in_dim).
        x = x.contiguous()

        out = XWTNice.apply(x, w, b, channel_group, spatial_group)

        # out has vmapped dimension at dim 0
        return out, 0
