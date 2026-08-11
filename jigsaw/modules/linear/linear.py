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
        """
        super().__init__()

        stdev = 1.0 / np.sqrt(in_dim * row_group.size())
        assert in_dim % column_group.size() == 0
        assert out_dim % (model_group.size() // column_group.size()) == 0

        bias_dim = out_dim // column_group.size()
        in_dim = in_dim // (column_group.size())
        out_dim = out_dim // (model_group.size() // column_group.size())

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

        # Map weights — clone so the original graph is detached.
        #
        # The constructor shards out_features by model_group.size() // column_group.size(),
        # not by the row group size, and the two only coincide when model_kw names the
        # full model group. Callers that pass a different group — Attention passes
        # col_kw, leaving out_features replicated across the row group — would otherwise
        # get a weight whose shape contradicts the one the constructor allocated. Since
        # the assignment below rebinds .data, that mismatch is silent.
        out_parallelism = (
            process_mesh_dict[model_kw].size() // process_mesh_dict[col_kw].size()
        )
        row_rank = (
            process_mesh_dict[row_kw].rank() % out_parallelism
            if out_parallelism > 1
            else 0
        )
        row_shard = model.out_features // out_parallelism
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


class DistributedMatMulXWT(torch.nn.Module):
    """Batched distributed matrix-matrix (XWT) multiplication."""

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
        bayesian: bool
            Boolean flag for whether the model is Bayesian.
        config: Optional[LinearConfig]
            Configuration and selection strategy of linear layer.
            If None, this will be initialized here.
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.model_group = model_group
        self.bayesian = bayesian
        self.config = (
            LinearConfig(
                parallelism, None, self.bayesian, linear_layer=False, device=self.device
            )
            if config is None
            else config
        )

        self.XWT = XWT.apply

    def get_parameter_count(self) -> int:
        """Return parameter count for disdtributed MatMulXWT operation."""
        return 0

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor = None
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
            self.config,
            self.model_group,
            self.bayesian,
        )
        cij = cast(torch.Tensor, cij)

        return cij

    @classmethod
    def convert_to_jigsaw(
        cls,
        xwt: "MatMulXWT",
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
        **kwargs: Any,
    ) -> "DistributedMatMulXWT":
        """
        Convert torch.nn.Linear to jigsaw based on given process mesh.

        Parameters
        ----------
        xwt: MatMulXWT
            Instance of xwt that should be converted to jigsaw distributed matmul.
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
        model: jigsaw.linear.DistributedMatMulXWT
            Instance of jigsaw distributed linear layer.
            Jigsaw layer contains weights and biases given from torch.nn.Linear layer.
        """
        instance = cls(
            local_rank=process_mesh_dict["model"].rank(),
            device=device,
            parallelism=process_mesh_dict["model"].size(),
            model_group=process_mesh_dict["model"],
        )
        return instance


class DistributedMatMulXW(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across two processes."""

    def __init__(
        self,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool = False,
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

        self.config = LinearConfig(
            parallelism, None, self.bayesian, linear_layer=False, device=self.device
        )

        self.XW = XW.apply

    def get_parameter_count(self) -> int:
        """Return parameter count for matrix multiplication."""
        return 0

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None
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
            self.config,
            self.model_group,
            self.bayesian,
        )
        cij = cast(torch.Tensor, cij)

        return cij

    @classmethod
    def convert_to_jigsaw(
        cls,
        xw: "MatMulXW",
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
        **kwargs: Any,
    ) -> "DistributedMatMulXW":
        """
        Convert torch.nn.Linear to jigsaw based on given process mesh.

        Parameters
        ----------
        xw: MatMulXW
            Instance of MatMulXW class that should be converted to jigsaw distributed mat-mul (XW).
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
        model: jigsaw.linear.DistributedMatMulXW
            Instance of jigsaw distributed linear layer.
            Jigsaw layer contains weights and biases given from torch.nn.Linear layer.
        """
        instance = cls(
            local_rank=process_mesh_dict["model"].rank(),
            device=device,
            parallelism=process_mesh_dict["model"].size(),
            model_group=process_mesh_dict["model"],
        )
        return instance


class DistributedMatMulXTW(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across two processes."""

    def __init__(
        self,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool = False,
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

        self.config = LinearConfig(
            parallelism, None, self.bayesian, linear_layer=False, device=self.device
        )

        self.XTW = XTW.apply

    def get_parameter_count(self) -> int:
        """Return parameter count for matrix multiplication."""
        return 0

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None
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
        cij = self.XTW(
            x,
            w,
            b,
            self.local_rank,
            self.device,
            self.config,
            self.model_group,
            self.bayesian,
        )
        cij = cast(torch.Tensor, cij)

        return cij

    @classmethod
    def convert_to_jigsaw(
        cls,
        xtw: "MatMulXTW",
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
        **kwargs: Any,
    ) -> "DistributedMatMulXTW":
        """
        Convert torch.nn.Linear to jigsaw based on given process mesh.

        Parameters
        ----------
        xtw: MatMulXTW
            Instance of matrix multiplication class that should be converted to jigsaw distributed layer.
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
        model: jigsaw.linear.DistributedMatMulXW
            Instance of jigsaw distributed linear layer.
            Jigsaw layer contains weights and biases given from torch.nn.Linear layer.
        """
        instance = cls(
            local_rank=process_mesh_dict["model"].rank(),
            device=device,
            parallelism=process_mesh_dict["model"].size(),
            model_group=process_mesh_dict["model"],
        )
        return instance


class MatMulXWT(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across two processes."""

    def __init__(
        self,
    ) -> None:
        """Initialize the distributed XW module."""
        super().__init__()

    def get_parameter_count(self) -> int:
        """Return parameter count for matrix multiplication."""
        return 0

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        x: torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        w: torch.Tensor
            Local weight tensor of shape (local_in_dim, local_out_dim)

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
        """
        cij = x @ w.mT

        return cij


class MatMulXW(torch.nn.Module):
    """Batched matrix-matrix (XW) multiplication distributed across two processes."""

    def __init__(
        self,
    ) -> None:
        """Initialize the distributed XW module."""
        super().__init__()

    def get_parameter_count(self) -> int:
        """Return parameter count for matrix multiplication."""
        return 0

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear layer (XW) distributed across two processes.

        Parameters
        ----------
        x: torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        w: torch.Tensor
            Local weight tensor of shape (local_in_dim, local_out_dim)

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
        """
        cij = x @ w

        return cij


class MatMulXTW(torch.nn.Module):
    """Batched matrix-matrix (XW) multiplication distributed across two processes."""

    def __init__(
        self,
    ) -> None:
        """Initialize the distributed XW module."""
        super().__init__()

    def get_parameter_count(self) -> int:
        """Return parameter count for matrix multiplication."""
        return 0

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear layer (XTW) distributed across two processes.

        Parameters
        ----------
        x: torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        w: torch.Tensor
            Local weight tensor of shape (local_in_dim, local_out_dim)

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
        """
        cij = x.mT @ w

        return cij
