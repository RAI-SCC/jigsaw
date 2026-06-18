"""Modules for distributed linear layers and operations."""

from typing import Any, Callable, Optional, Protocol, Union, cast

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from . import distributed_matmul as dist_mm
from . import hooks
from .utils_aggregate import gather_1d_results, gather_results


class LinearSelectionStrategy:
    """Selection strategy for distributed linear operation."""

    def __init__(
        self,
        parallelism: int,
        bias: torch.Tensor | None,
        bayesian: bool,
        linear_layer: bool = True,
    ):
        self.mm_kernels = self._mm_kernel_selection(parallelism)
        self.grad_bias = self._grad_bias_selection(parallelism, bias, bayesian)
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

    def _return_identity(
        self, x: torch.Tensor, b: Optional[Parameter] = None
    ) -> torch.Tensor:
        """Return x as identity since bias does not exist."""
        return x

    # Reduce bias terms over spatial groups
    def _grad_bias_selection(
        self,
        parallelism: int,
        bias: Union[torch.Tensor, None],
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
        self, grad_output: torch.Tensor, spatial_group: dist.ProcessGroup
    ) -> torch.Tensor:
        """Calculate grad wrt bias when spatial synchronization is not required."""
        return grad_output.sum(dim=(-3, -2))

    def _no_bias(
        self,
        grad_output: Optional[torch.Tensor],
        spatial_group: Optional[dist.ProcessGroup],
    ) -> Optional[torch.Tensor]:
        """Return identity when bias does not exist."""
        return None

    def _mm_kernel_selection(self, parallelism: int) -> list[Callable]:
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
            case 1:
                return lambda x, w: x @ w.mT
            case 2:
                return dist_mm.xw_transpose_twoway
            case 4:
                return dist_mm.xw_transpose_fourway
            case 8:
                return dist_mm.xw_transpose_eightway
            case _:
                raise ValueError(f"Unsupported parallelism: {parallelism}")
        return

    def _select_xw_kernel(self, parallelism: int) -> Callable:
        """Return xw kernel based on parallelism."""
        match parallelism:
            case 1:
                return lambda x, w: x @ w
            case 2:
                return dist_mm.xw_twoway
            case 4:
                return dist_mm.xw_fourway
            case 8:
                return dist_mm.xw_eightway
            case _:
                raise ValueError(f"Unsupported parallelism: {parallelism}")
        return

    def _select_x_transpose_w_kernel(self, parallelism: int) -> Callable:
        """Return x_tw kernel based on parallelism."""
        match parallelism:
            case 1:
                return lambda x, w: x.mT @ w
            case 2:
                return dist_mm.x_transpose_w_twoway
            case 4:
                return dist_mm.x_transpose_w_fourway
            case 8:
                return dist_mm.x_transpose_w_eightway
            case _:
                raise ValueError(f"Unsupported parallelism: {parallelism}")
        return

    def _select_xw_xtw_fused_kernel(self, parallelism: int) -> Callable:
        match parallelism:
            case 1:
                return dist_mm.fused_xw_xtw_oneway
            case 2:
                return dist_mm.fused_xw_xtw_twoway
            case 4:
                return dist_mm.fused_xw_xtw_fourway
            case _:
                raise ValueError(f"Unsupported parallelism: {parallelism}")
        return


class DistributedXWT(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        local_rank: int,
        device: torch.device,
        model_group: dist.ProcessGroup,
        tensor_group: dist.ProcessGroup,
        domain_group: dist.ProcessGroup,
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
            Process group containing all ranks within model-parallel instances.
            Is the Union of domain_group and tensor_group.
        domain_group: dist.ProcessGroup
            Process group containin domain-parallel processes. Splits the last dimension of X, and the last dimension of W.
        tensor_group: dist.ProcessGroup
            Process group containin tensor/weight-parallel processes. Splits the first dimension of the weights and bias terms.
        bayesian: bool
            Boolean flag for whether the model is bayesian
        """
        super().__init__()

        stdev = 1.0 / np.sqrt(in_dim * 2)
        assert in_dim % domain_group.size() == 0
        assert out_dim % (model_group.size() // domain_group.size()) == 0
        bias_dim = out_dim // domain_group.size()
        in_dim = in_dim // domain_group.size()
        out_dim = out_dim // (model_group.size() // domain_group.size()) 

        self.weight = torch.nn.Parameter(
            torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev).to(device)
        )
        self.parallelism = model_group.size()

        self.bias = (
            torch.nn.Parameter(torch.zeros(bias_dim).uniform_(-stdev, stdev).to(device))
            if bias
            else None
        )
        

        self.out_dim = out_dim
        self.in_dim = in_dim
        self.device = device
        self.local_rank = local_rank
        self.model_group = model_group
        self.domain_group = domain_group
        self.tensor_group = tensor_group
        self.bayesian = bayesian
        self.selector = LinearSelectionStrategy(
            self.parallelism,
            self.bias,
            self.bayesian,
            linear_layer=True,
        )

        if self.bias is not None:
            self.bias.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.tensor_group, async_op=True)
            )
            dist.broadcast(self.bias.data, group_src=0, group=self.tensor_group)

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
            self.tensor_group,
            self.bayesian,
        )
        cij = cast(torch.Tensor, cij)
        return cij

    def get_parameter_count(self) -> int:
        """Return parameter count for linear layer."""
        num_param = (
            0
            if self.bias is None
            else self.bias.numel() // dist.get_world_size(group=self.tensor_group)
        )
        return num_param + self.weight.numel()

    @classmethod
    def convert_to_jigsaw(
        cls,
        model: torch.nn.Linear,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
        **kwargs: Any,
    ) -> "DistributedXWT":
        """
        Convert torch.nn.Linear to jigsaw based on given process mesh.

        Parameters
        ----------
        model: torch.nn.Linear
            Instance of linear class that should be converted to jigsaw linear.
        process_mesh_dict: dict
            Process mesh dict that comtains row, column, and model process groups
        device: torch.device
            Device upon which computation happens.
        row_kw: str
            Key for accessing row process groups
        col_kw: str
            Key for accessing col process groups
        model_kw: str
            Key for accessing model process groups.
            Model process group should contain the Union of the row and column groups.
        kwargs: Any
            Additional inputs.

        Returns
        -------
        model: jigsaw.linear.DistributedXWT
            Instance of jigsaw distributed linear layer.
            Jigsaw layer contains weights and biases given from torch.nn.Linear layer.
        """
        assert model.in_features % process_mesh_dict[row_kw].size() == 0, (
            "in_features is not divisible by process mesh rows"
        )
        assert model.out_features % process_mesh_dict[col_kw].size() == 0, (
            "out_features is not divisible by process mesh columns"
        )

        instance = cls(
            in_dim=model.in_features,
            out_dim=model.out_features,
            bias=model.bias is not None,
            local_rank=process_mesh_dict[model_kw].rank(),
            device=device,
            model_group=process_mesh_dict[model_kw],
            tensor_group=process_mesh_dict[row_kw],
            domain_group=process_mesh_dict[col_kw],
            bayesian=False,  # TODO
        )

        # Map weights — clone so the original graph is detached
        row_rank = process_mesh_dict[row_kw].rank()
        row_shard = model.out_features // process_mesh_dict[row_kw].size()
        col_rank = process_mesh_dict[col_kw].rank()
        col_shard = model.in_features // process_mesh_dict[col_kw].size()

        instance.weight.data = (
            model.weight.data[
                row_rank * row_shard : (row_rank + 1) * row_shard,
                col_rank * col_shard : (col_rank + 1) * col_shard,
            ]
            .clone()
            .to(device)
        )
        if isinstance(model.bias, torch.Tensor) and isinstance(
            instance.bias, torch.Tensor
        ):
            bias_size = model.out_features // process_mesh_dict[col_kw].size()
            instance.bias.data = model.bias.data[
                col_rank * bias_size : (col_rank + 1) * bias_size
            ].clone()
        return instance

    def convert_to_torch(
        self,
        process_mesh_dict: dict,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> torch.nn.Linear:
        """
        Convert jigsaw linear layer to torch.

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
        weights, bias = self.merge_weights(process_mesh_dict, row_kw, col_kw, model_kw)
        linear = torch.nn.Linear(
            in_features=weights.shape[1],
            out_features=weights.shape[0],
            bias=True if bias is not None else False,
            device=self.device,
            dtype=self.dtype,
        )
        linear.weights.data = weights
        if isinstance(linear.bias, Parameter) and isinstance(bias, torch.Tensor):
            linear.bias.data = bias

        return linear

    def merge_weights(
        self,
        from_process_mesh: dict,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Gathers and returns global weights (and bias) in a manner compatible to torch.nn.Linear.

        Parameters
        ----------
        from_process_mesh: dict
            Process mesh dict that comtains row, column, and model process groups
        row_kw: str
            Key for accessing row process groups
        col_kw: str
            Key for accessing col process groups
        model_kw: str
            Key for accessing model process groups.
            Model process group should contain the Union of the row and column groups.

        Returns
        -------
        global_weights: torch.Tensor
            Merged weights.
        global_bias: Union[torch.Tensor, None]
            Merged bias, if it exists.
        """
        global_weights = gather_results(
            self.weight, from_process_mesh, row_kw, col_kw, model_kw
        )

        global_bias = None
        if self.bias is not None:
            global_bias = gather_1d_results(
                self.bias, from_process_mesh, dim=0, pm_key=col_kw
            )

        return global_weights, global_bias

    def reshard_parameters(
        self,
        from_process_mesh: dict,
        to_process_mesh: dict,
        pm_row_key: str = "row",
        pm_col_key: str = "col",
    ) -> None:
        """from_process_mesh has process mesh with 'row' and 'col' flags."""
        # re-define self selector
        self.parallelism = to_process_mesh["row"].size() * to_process_mesh["col"].size()
        self.selector = LinearSelectionStrategy(
            self.parallelism,
            self.bias,
            self.bayesian,
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

        if isinstance(global_bias, torch.Tensor) and isinstance(
            global_bias, torch.Tensor
        ):
            col_shard_size = to_bias_shape_local
            self.bias = torch.nn.Parameter(
                global_bias[
                    col_rank * col_shard_size : (col_rank + 1) * col_shard_size
                ].contiguous()
            )
            del global_bias


class LinearCtx(Protocol):
    """Protocol for types stored within ctx."""

    saved_tensors: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    local_rank: int
    device: torch.Tensor
    selector: LinearSelectionStrategy
    model_group: dist.ProcessGroup
    spatial_group: dist.ProcessGroup
    bayesian: bool
    b: Optional[Parameter]


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
        w : torch.Tensor
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
            Process group for domain-parallel ranks.
        bayesian: bool
            Flag for whether the model is bayesian.


        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim) or (S, B, out_dim, in_dim).
        """
        w = selector.reshape_weights(w)
        xwt = selector.mm_kernels[0](x, w, local_rank, device, model_group)
        xwt = selector.add_bias(xwt, b)

        xwt = cast(torch.Tensor, xwt)

        return xwt

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx, inputs: Any, output: torch.Tensor
    ) -> None:
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
        c = cast(LinearCtx, ctx)
        c.local_rank = local_rank
        c.device = device
        c.selector = selector
        c.model_group = model_group
        c.spatial_group = spatial_group
        c.bayesian = bayesian

    @staticmethod
    def backward(
        ctx: LinearCtx, grad_output: torch.Tensor
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
        info: Any,
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
        info:
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims:
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, rank, device, selector, model_group, spatial_group, bayesian).
        x: torch.Tensor
            Possibly batched input.
        w: torch.Tensor
            Possibly shared or batched weights.
        b: torch.Tensor | None
            Possibly shared or batched bias.
        local_rank: int
            Local rank.
        device: torch.device
            Device where computation occurs.
        selector: LinearSelectionStrategy
            Instance of selection class with function pointers based on specific config.
        model_group: dist.ProcessGroup
            Domain- and tensor-parallel process group.
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.
        bayesian: bool
            Flag to see whether model is bayesian.

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
            self.bayesian,
            linear_layer=False,
        )

        self.XWT = XWT.apply

    def get_parameter_count(self) -> int:
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
        cij = cast(torch.Tensor, cij)

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
            self.bayesian,
            linear_layer=False,
        )

        self.XW = XW.apply

    def get_parameter_count(self) -> int:
        """Return parameter count for matrix multiplication."""
        return 0

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass of linear layer (XW) distributed across two processes.

        Parameters
        ----------
        x: torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        w: torch.Tensor
            Local weight tensor of shape (local_in_dim, local_out_dim)
        b: torch.Tensor | None
            Local bias tensor of shape (local_out_dim)

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
        cij = cast(torch.Tensor, cij)

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
        selector: LinearSelectionStrategy,
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
        xw = cast(torch.Tensor, xw)
        return xw

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx, inputs: Any, output: torch.Tensor
    ) -> None:
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, w, b, local_rank, device, parallelism, model_group)
        output: xwt
        """
        x, w, b, local_rank, device, selector, model_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        c = cast(LinearCtx, ctx)
        c.local_rank = local_rank
        c.device = device
        c.selector = selector
        c.model_group = model_group
        c.b = b
        c.bayesian = bayesian

    @staticmethod
    def backward(
        ctx: LinearCtx, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None, None
    ]:
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
        info: Any,
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
        Define XW behavior under torch.vmap.

        Parameters
        ----------
        info:
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims:
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, local_rank, device, parallelism, model_group).
        x : Tensor
            Batched input.
        w : Tensor
            Batched weights.
        b : Tensor or None
            Batched bias.
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
        output : Tensor
            Batched output with vmapped dimension in front.
        out_dim : Optional[int]
            Index of vmapped dimension in output (0 if batched, None if not).
        """
        x = x.contiguous()
        out = XW.apply(x, w, b, local_rank, device, selector, model_group, bayesian)

        # out has vmapped dimension at dim 0
        return out, 0
