"""Modules for distributed linear layers and operations."""

from typing import Any, Optional, cast

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

import jigsaw.hooks as hooks
from jigsaw.utils_aggregate import gather_1d_results, gather_results

from .config import LinearConfig
from .functional import XTW, XW, XWT


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
        column_group: dist.ProcessGroup,
        row_group: dist.ProcessGroup,
        bayesian: bool = False,
        config: Optional[LinearConfig] = None,
        dtype: torch.dtype = torch.float32,
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
            Is the Union of row and column group.
        column_group: dist.ProcessGroup
            Process group splitting the last dimension of X and W.
            In a standard case, this would be the channel-wise process group.
        row_group: dist.ProcessGroup
            Process group splitting the first dimension of W.
            In a standard case, this would be the spatial-wise process group.
        bayesian: bool
            Boolean flag for whether the model is bayesian.
        config: Optional[LinearConfig]
            Configuration and selection strategy of linear layer.
            If None, this will be initialized here.
        dtype: torch.dtype
            Data type used for the layer parameters and computations.
        """
        super().__init__()

        self.dtype = dtype
        self.device = device

        stdev = 1.0 / np.sqrt(in_dim * row_group.size())
        assert in_dim % column_group.size() == 0
        assert out_dim % (model_group.size() // column_group.size()) == 0

        bias_dim = out_dim // column_group.size()
        in_dim = in_dim // (column_group.size())
        out_dim = out_dim // (model_group.size() // column_group.size())

        self.weight = torch.nn.Parameter(
            torch.zeros(out_dim, in_dim, dtype=self.dtype, device=self.device).uniform_(
                -stdev, stdev
            )
        )
        self.parallelism = model_group.size()

        self.bias = (
            torch.nn.Parameter(
                torch.zeros(bias_dim, dtype=self.dtype, device=self.device).uniform_(
                    -stdev, stdev
                )
            )
            if bias
            else None
        )

        self.out_dim = out_dim
        self.in_dim = in_dim
        self.device = device
        self.local_rank = local_rank
        self.model_group = model_group
        self.column_group = column_group
        self.row_group = row_group
        self.bayesian = bayesian
        self.config = (
            LinearConfig(
                self.parallelism,
                self.bias,
                self.bayesian,
                linear_layer=True,
                device=self.device,
            )
            if config is None
            else config
        )

        if self.bias is not None:
            self.bias.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.row_group, async_op=True)
            )
            dist.broadcast(self.bias.data, group_src=0, group=self.row_group)

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
            self.config,
            self.model_group,
            self.bayesian,
        )
        cij = cast(torch.Tensor, cij)
        return cij

    def get_parameter_count(self) -> int:
        """Return parameter count for linear layer."""
        num_param = (
            0
            if self.bias is None
            else self.bias.numel() // dist.get_world_size(group=self.row_group)
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
            row_group=process_mesh_dict[row_kw],
            column_group=process_mesh_dict[col_kw],
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
        linear.weight.data = weights
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
        self.config = LinearConfig(
            self.parallelism,
            self.bias,
            self.bayesian,
            linear_layer=True,
            device=self.device,
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

        if self.bias is not None and isinstance(global_bias, torch.Tensor):
            col_shard_size = to_bias_shape_local
            self.bias = torch.nn.Parameter(
                global_bias[
                    col_rank * col_shard_size : (col_rank + 1) * col_shard_size
                ].contiguous()
            )
            del global_bias


class _DistributedMatMul(torch.nn.Module):
    """
    Shared harness for the parameter-free distributed matmul modules.

    :class:`DistributedMatMulXWT`, :class:`DistributedMatMulXW` and
    :class:`DistributedMatMulXTW` differ only in which autograd ``Function`` drives
    the product; construction, the forward dispatch, parameter count and
    ``convert_to_jigsaw`` are identical and live here. Subclasses set
    :attr:`_autograd_fn`.
    """

    #: autograd ``Function`` whose ``.apply`` this module drives; set by subclasses.
    _autograd_fn: type[torch.autograd.Function]

    def __init__(
        self,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool = False,
        config: Optional[LinearConfig] = None,
    ) -> None:
        """
        Initialize the distributed matmul operation sharded over ``parallelism`` ranks.

        Parameters
        ----------
        local_rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism : int
            Number of processes the model is distributed over.
        model_group : dist.ProcessGroup
            Process group containing all ranks within model-parallel instances.
        bayesian : bool
            Boolean flag for whether the model is Bayesian.
        config : Optional[LinearConfig]
            Configuration and selection strategy. Initialized here when ``None``.
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.bayesian = bayesian
        self.config = (
            config
            if config is not None
            else LinearConfig(
                parallelism, None, bayesian, linear_layer=False, device=device
            )
        )
        self.matmul = self._autograd_fn.apply

    def get_parameter_count(self) -> int:
        """Return parameter count (always 0 -- these modules hold no parameters)."""
        return 0

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the distributed matmul.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        w : torch.Tensor
            Local weight tensor of shape (local_in_dim, local_out_dim).
        b : torch.Tensor | None
            Local bias tensor of shape (local_out_dim).

        Returns
        -------
        cij : torch.Tensor
            Local output tensor.
        """
        cij = self.matmul(
            x,
            w,
            b,
            self.local_rank,
            self.device,
            self.config,
            self.model_group,
            self.bayesian,
        )
        return cast(torch.Tensor, cij)

    @classmethod
    def convert_to_jigsaw(
        cls,
        module: torch.nn.Module,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
        **kwargs: Any,
    ) -> "_DistributedMatMul":
        """
        Build the distributed matmul module from a process mesh.

        Parameters
        ----------
        module : torch.nn.Module
            Plain (non-distributed) matmul module being converted. Unused: the
            operation is parameter-free, so only the mesh is needed.
        process_mesh_dict : dict
            Process mesh dict containing row, column and model process groups.
        device : torch.device
            Device upon which computation happens.
        row_kw : str
            Key for accessing row process groups.
        col_kw : str
            Key for accessing col process groups.
        model_kw : str
            Key for accessing model process groups. Must map to the union of the
            row and column groups.
        kwargs : Any
            Additional inputs (ignored).

        Returns
        -------
        _DistributedMatMul
            Instance of the concrete jigsaw distributed matmul subclass.
        """
        return cls(
            local_rank=process_mesh_dict[model_kw].rank(),
            device=device,
            parallelism=process_mesh_dict[model_kw].size(),
            model_group=process_mesh_dict[model_kw],
        )


class DistributedMatMulXWT(_DistributedMatMul):
    """Batched distributed matrix-matrix (X @ W.mT) multiplication."""

    _autograd_fn = XWT


class DistributedMatMulXW(_DistributedMatMul):
    """Batched distributed matrix-matrix (X @ W) multiplication."""

    _autograd_fn = XW


class DistributedMatMulXTW(_DistributedMatMul):
    """Batched distributed matrix-matrix (X.mT @ W) multiplication."""

    _autograd_fn = XTW


class _MatMul(torch.nn.Module):
    """
    Plain (non-distributed) reference matmul.

    A model is authored with these; :meth:`_DistributedMatMul.convert_to_jigsaw`
    swaps in the sharded version. Subclasses implement :meth:`forward`.
    """

    def get_parameter_count(self) -> int:
        """Return parameter count (always 0)."""
        return 0


class MatMulXWT(_MatMul):
    """Batched matrix-matrix (X @ W.mT) multiplication."""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Return ``x @ w.mT``."""
        return x @ w.mT


class MatMulXW(_MatMul):
    """Batched matrix-matrix (X @ W) multiplication."""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Return ``x @ w``."""
        return x @ w


class MatMulXTW(_MatMul):
    """Batched matrix-matrix (X.mT @ W) multiplication."""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Return ``x.mT @ w``."""
        return x.mT @ w
