"""Distributed linear layers and matmul modules."""

import math
from typing import Any, Optional, cast

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

import jigsaw.hooks as hooks
from jigsaw import init
from jigsaw.utils_aggregate import gather_1d_results, gather_results

from .config import LinearConfig
from .functional import XTW, XW, XWT

# ``(row_group.size(), column_group.size())`` layouts the XWT kernels implement.
# The kernel is picked by ``parallelism`` (= row * col) alone and each one bakes
# in a fixed 2-D arrangement of the model ranks, so an arbitrary ``row x col``
# factorisation of the same rank count is *not* interchangeable: parallelism 4
# is a 2x2 SUMMA, parallelism 8 a 4x2 one, parallelism 2 an in-split (1x2).
# ``merge_weights`` / ``reshard_parameters`` still work for any factorisation --
# only ``forward`` is constrained.
_SUPPORTED_XWT_LAYOUTS = frozenset({(1, 1), (1, 2), (2, 2), (4, 2)})


class DistributedXWT(torch.nn.Module):
    """Distributed linear layer computing ``x @ w.mT``."""

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
        """Initialize the layer.

        Parameters
        ----------
        in_dim : int
            Input dimension of the global (unsharded) weight.
        out_dim : int
            Output dimension of the global (unsharded) weight.
        bias : bool
            Whether to add a bias.
        local_rank : int
            Rank within the model group.
        device : torch.device
            Device of the module.
        model_group : dist.ProcessGroup
            Process group of one model-parallel instance, the union of the row
            and column groups.
        column_group : dist.ProcessGroup
            Process group splitting the last dimension of ``x`` and ``w``.
        row_group : dist.ProcessGroup
            Process group splitting the first dimension of ``w``.
        bayesian : bool
            Whether the model is Bayesian.
        config : LinearConfig, optional
            Kernel selection. Defaults to a configuration built from the
            arguments.
        dtype : torch.dtype
            Data type of the parameters and computations.
        """
        super().__init__()

        self.dtype = dtype
        self.device = device

        assert in_dim % column_group.size() == 0
        assert out_dim % (model_group.size() // column_group.size()) == 0

        # Global (unsharded) weight/bias shape, torch.nn.Linear layout ``(out,
        # in)``. ``reset_parameters`` keys the init distribution off these, so a
        # freshly built layer is drawn from the same distribution as the
        # equivalent ``torch.nn.Linear`` -- the row-/column-group shard sizes
        # only decide which slice of that distribution this rank samples.
        self.global_weight_shape = (out_dim, in_dim)
        self.global_bias_shape = (out_dim,) if bias else None

        bias_dim = out_dim // column_group.size()
        in_dim = in_dim // (column_group.size())
        out_dim = out_dim // (model_group.size() // column_group.size())

        self.weight = torch.nn.Parameter(
            torch.zeros(out_dim, in_dim, dtype=self.dtype, device=self.device)
        )
        self.parallelism = model_group.size()

        self.bias = (
            torch.nn.Parameter(
                torch.zeros(bias_dim, dtype=self.dtype, device=self.device)
            )
            if bias
            else None
        )
        self.reset_parameters()

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

    def reset_parameters(self) -> None:
        """Initialize the parameters like ``torch.nn.Linear``.

        The fans are computed from the global weight shape.
        """
        init.kaiming_uniform_(
            self.weight, a=math.sqrt(5), global_shape=self.global_weight_shape
        )
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.global_weight_shape)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer.

        Parameters
        ----------
        x : torch.Tensor
            Local input.

        Returns
        -------
        torch.Tensor
            Local output.
        """
        layout = (self.row_group.size(), self.column_group.size())
        if layout not in _SUPPORTED_XWT_LAYOUTS:
            raise NotImplementedError(
                f"DistributedXWT.forward has no kernel for a "
                f"row x col = {layout[0]}x{layout[1]} mesh; the XWT kernels "
                f"implement only {sorted(_SUPPORTED_XWT_LAYOUTS)} (row, col). "
                f"merge_weights / reshard_parameters still work on this mesh."
            )
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
        """Return the number of local parameters."""
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
        """Convert a ``torch.nn.Linear`` to a :class:`DistributedXWT`.

        Parameters
        ----------
        model : torch.nn.Linear
            Layer to convert.
        process_mesh_dict : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        device : torch.device
            Device of the converted module.
        row_kw : str
            Key of the row process group.
        col_kw : str
            Key of the column process group.
        model_kw : str
            Key of the model process group.
        **kwargs : Any
            Forwarded to the constructor.

        Returns
        -------
        DistributedXWT
            Distributed layer with the weights and bias of ``model``.
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
        """Convert the layer to a ``torch.nn.Linear``.

        Parameters
        ----------
        process_mesh_dict : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        row_kw : str
            Key of the row process group.
        col_kw : str
            Key of the column process group.
        model_kw : str
            Key of the model process group.

        Returns
        -------
        torch.nn.Linear
            Layer with the merged weights.
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
        """Merge the sharded weight and bias into global tensors.

        Parameters
        ----------
        from_process_mesh : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        row_kw : str
            Key of the row process group.
        col_kw : str
            Key of the column process group.
        model_kw : str
            Key of the model process group.

        Returns
        -------
        global_weights : torch.Tensor
            Merged weights.
        global_bias : torch.Tensor | None
            Merged bias, if any.
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
        pm_model_key: str = "model",
    ) -> None:
        """Reshard the weight and bias from one process mesh onto another.

        The module is rebound to the new mesh.

        Parameters
        ----------
        from_process_mesh : dict
            Process groups of the current mesh.
        to_process_mesh : dict
            Process groups of the new mesh.
        pm_row_key, pm_col_key, pm_model_key : str
            Keys of the row, column and model process groups in both meshes.
        """
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
        global_weights, global_bias = self.merge_weights(
            from_process_mesh, pm_row_key, pm_col_key, pm_model_key
        )

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

        # Rebind the module to the target mesh so a forward/backward pass on it
        # is valid: process groups, ranks, the kernel-selection config and the
        # bias grad-sync hook all key off the new group sizes.
        self.row_group = to_process_mesh[pm_row_key]
        self.column_group = to_process_mesh[pm_col_key]
        self.model_group = to_process_mesh[pm_model_key]
        self.local_rank = self.model_group.rank()
        self.parallelism = self.model_group.size()
        self.out_dim, self.in_dim = tuple(self.weight.shape)
        self.config = LinearConfig(
            self.parallelism,
            self.bias,
            self.bayesian,
            linear_layer=True,
            device=self.device,
        )

        if self.bias is not None:
            # ``self.bias`` is a fresh Parameter, so the __init__ hook is gone;
            # re-register it on the new row group and re-broadcast rank 0's slice.
            self.bias.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.row_group, async_op=True)
            )
            dist.broadcast(self.bias.data, group_src=0, group=self.row_group)


class _DistributedMatMul(torch.nn.Module):
    """Base class of the parameter-free distributed matmul modules.

    Subclasses set :attr:`_autograd_fn`.
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
        """Initialize the module.

        Parameters
        ----------
        local_rank : int
            Rank within the model group.
        device : torch.device
            Device of the module.
        parallelism : int
            Number of model-parallel processes.
        model_group : dist.ProcessGroup
            Process group of one model-parallel instance.
        bayesian : bool
            Whether the model is Bayesian.
        config : LinearConfig, optional
            Kernel selection. Defaults to a configuration built from the
            arguments.
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
        """Return the number of parameters, which is always 0."""
        return 0

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the distributed matmul.

        Parameters
        ----------
        x : torch.Tensor
            Local input.
        w : torch.Tensor
            Local weight.
        b : torch.Tensor | None
            Local bias.

        Returns
        -------
        torch.Tensor
            Local output.
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
        """Build a distributed matmul module from a process mesh.

        Parameters
        ----------
        module : torch.nn.Module
            Matmul module to convert. Unused, since the operation has no
            parameters.
        process_mesh_dict : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        device : torch.device
            Device of the converted module.
        row_kw : str
            Key of the row process group.
        col_kw : str
            Key of the column process group.
        model_kw : str
            Key of the model process group.
        **kwargs : Any
            Unused.

        Returns
        -------
        _DistributedMatMul
            Distributed matmul module.
        """
        return cls(
            local_rank=process_mesh_dict[model_kw].rank(),
            device=device,
            parallelism=process_mesh_dict[model_kw].size(),
            model_group=process_mesh_dict[model_kw],
        )


class DistributedMatMulXWT(_DistributedMatMul):
    """Distributed batched matmul ``x @ w.mT``."""

    _autograd_fn = XWT


class DistributedMatMulXW(_DistributedMatMul):
    """Distributed batched matmul ``x @ w``."""

    _autograd_fn = XW


class DistributedMatMulXTW(_DistributedMatMul):
    """Distributed batched matmul ``x.mT @ w``."""

    _autograd_fn = XTW


class _MatMul(torch.nn.Module):
    """Base class of the non-distributed matmul modules.

    :meth:`_DistributedMatMul.convert_to_jigsaw` replaces them with the
    distributed versions.
    """

    def get_parameter_count(self) -> int:
        """Return the number of parameters, which is always 0."""
        return 0


class MatMulXWT(_MatMul):
    """Batched matmul ``x @ w.mT``."""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Return ``x @ w.mT``."""
        return x @ w.mT


class MatMulXW(_MatMul):
    """Batched matmul ``x @ w``."""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Return ``x @ w``."""
        return x @ w


class MatMulXTW(_MatMul):
    """Batched matmul ``x.mT @ w``."""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Return ``x.mT @ w``."""
        return x.mT @ w
