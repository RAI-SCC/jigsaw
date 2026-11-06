"""Distributed linear layers and matmul modules."""

import math
import warnings
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from jigsaw import init, parallel
from jigsaw._core.shard_repr import shard_repr
from jigsaw.gather import gather_1d_results, gather_results

from ._config import LinearConfig
from ._functional import _XTW, _XW, _XWT

# Effective ``(model_group.size() // col_group.size(), col_group.size())``
# layouts the XWT kernels implement. The kernel is picked by
# ``model_group.size()`` alone and each bakes in a fixed layout, so other
# factorizations of the same rank count don't work. Only ``forward`` is
# constrained; ``merge_weights`` / ``reshard_parameters`` work for any layout.
_SUPPORTED_XWT_LAYOUTS = frozenset({(1, 1), (1, 2), (2, 2), (4, 2)})


def _warn_local_rank(local_rank: int | None, owner: str) -> None:
    """Warn that ``local_rank`` is ignored; the layer uses ``model_group.rank()``."""
    if local_rank is not None:
        warnings.warn(
            f"{owner}: local_rank is deprecated and ignored; the rank is taken "
            "from model_group.rank().",
            DeprecationWarning,
            stacklevel=3,
        )


def _row_split(mesh: dict[str, dist.ProcessGroup], col_kw: str, model_kw: str) -> int:
    """Return the number of ways the weight rows are split on ``mesh``.

    1 when the model group is the column group; the weight is then replicated
    over the row group.
    """
    return mesh[model_kw].size() // mesh[col_kw].size()


def _row_rank(
    mesh: dict[str, dist.ProcessGroup], row_kw: str, col_kw: str, model_kw: str
) -> int:
    """Return the index of this rank's block of weight rows on ``mesh``."""
    return 0 if _row_split(mesh, col_kw, model_kw) == 1 else mesh[row_kw].rank()


class DistributedLinear(torch.nn.Module):
    """Distributed linear layer computing ``x @ w.mT``.

    Parameters
    ----------
    in_dim : int
        Input dimension of the global (unsharded) weight.
    out_dim : int
        Output dimension of the global (unsharded) weight.
    device : torch.device
        Device of the module.
    model_group : dist.ProcessGroup
        Process group of one model-parallel instance, usually the union of
        the row and column groups. The first dimension of ``w`` is split
        ``model_group.size() // col_group.size()`` ways.
    row_group : dist.ProcessGroup
        Process group splitting the leading (row) dimensions of ``x``. The
        bias is replicated over it, and so is the weight when ``model_group``
        is ``col_group``.
    col_group : dist.ProcessGroup
        Process group splitting the last dimension of ``x`` and ``w``.
    bias : bool
        Whether to add a bias.
    config : LinearConfig, optional
        Kernel selection. Defaults to a configuration built from the
        arguments.
    dtype : torch.dtype
        Data type of the parameters and computations.
    local_rank : int, optional
        Deprecated and ignored. The rank is taken from ``model_group``.

    Raises
    ------
    ValueError
        If the group sizes do not divide ``in_dim`` and ``out_dim``.

    Warns
    -----
    DeprecationWarning
        If ``local_rank`` is given.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: torch.device,
        model_group: dist.ProcessGroup,
        row_group: dist.ProcessGroup,
        col_group: dist.ProcessGroup,
        bias: bool = True,
        config: LinearConfig | None = None,
        dtype: torch.dtype = torch.float32,
        *,
        local_rank: int | None = None,
    ) -> None:
        """Initialize the layer."""
        super().__init__()
        _warn_local_rank(local_rank, type(self).__name__)

        self.dtype = dtype
        self.device = device

        if in_dim % col_group.size() != 0:
            raise ValueError(
                f"in_dim={in_dim} is not divisible by the column group size "
                f"{col_group.size()}"
            )
        row_size = model_group.size() // col_group.size()
        if out_dim % row_size != 0:
            raise ValueError(
                f"out_dim={out_dim} is not divisible by the row split {row_size} "
                "(model group size // column group size)"
            )
        if bias and out_dim % col_group.size() != 0:
            raise ValueError(
                f"out_dim={out_dim} is not divisible by the column group size "
                f"{col_group.size()}, which shards the bias"
            )

        # Global (unsharded) weight/bias shape, torch.nn.Linear layout ``(out,
        # in)``. ``reset_parameters`` keys the init distribution off these, so a
        # freshly built layer is drawn from the same distribution as the
        # equivalent ``torch.nn.Linear``. The row-/column-group shard sizes
        # only decide which slice of that distribution this rank samples.
        self.global_weight_shape = (out_dim, in_dim)
        self.global_bias_shape = (out_dim,) if bias else None

        bias_dim = out_dim // col_group.size()
        in_dim = in_dim // (col_group.size())
        out_dim = out_dim // (model_group.size() // col_group.size())

        self.weight = torch.nn.Parameter(
            torch.zeros(out_dim, in_dim, dtype=self.dtype, device=self.device)
        )

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
        self.model_group = model_group
        self.row_group = row_group
        self.col_group = col_group
        self.config = (
            LinearConfig(
                self.model_group,
                linear_layer=True,
                device=self.device,
            )
            if config is None
            else config
        )

        self._mark_row_replicas()

        self.XWT = _XWT.apply

    def _mark_row_replicas(self) -> None:
        """Mark the parameters replicated over ``row_group`` and broadcast them.

        The bias always is. The weight is too when its rows are not split, i.e.
        ``model_group`` is ``col_group``; without the mark its gradient would
        differ between the row ranks.
        """
        params = [self.bias] if self.bias is not None else []
        if self.model_group.size() // self.col_group.size() == 1:
            params.insert(0, self.weight)
        for p in params:
            parallel.mark_replicated(p, self.row_group)
        if params:
            parallel.broadcast_replicas(self)

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
        # Check the effective layout, not ``row_group.size()``: with
        # ``model_group == col_group`` (e.g. attention's qkv projection, weights
        # replicated over ``row_group``) the row factor is 1 for any row size.
        row = self.model_group.size() // self.col_group.size()
        layout = (row, self.col_group.size())
        if layout not in _SUPPORTED_XWT_LAYOUTS:
            raise NotImplementedError(
                f"DistributedLinear.forward has no kernel for an effective "
                f"row x col = {layout[0]}x{layout[1]} mesh; the XWT kernels "
                f"implement only {sorted(_SUPPORTED_XWT_LAYOUTS)} (row, col). "
                f"merge_weights / reshard_parameters still work on this mesh."
            )
        weight, bias = parallel.sync_grad(self.weight, self.bias)
        cij = self.XWT(
            x,
            weight,
            bias,
            self.model_group.rank(),
            self.device,
            self.config,
            self.model_group,
        )
        cij = cast(torch.Tensor, cij)
        return cij

    def get_parameter_count(self) -> int:
        """Return the global number of parameters of this layer.

        Returns
        -------
        int
            Global (unsharded) count of this layer's parameters, including plain
            ``torch.nn`` children; children with ``get_parameter_count`` report
            their own. No communication.

        Notes
        -----
        This method calculates the number of parameters based on the global weight and bias shape calculated at initialization instead of counting the number of elements of the local parameters and scaling according to the respective communication group sizes.
        """
        num_bias_param = (
            0 if self.global_bias_shape is None else math.prod(self.global_bias_shape)
        )
        return num_bias_param + math.prod(self.global_weight_shape)

    def extra_repr(self) -> str:
        """Return the ``torch.nn.Linear`` fields and the shard sizes.

        ``row`` is the factor splitting the weight,
        ``model_group.size() // col_group.size()``, which is 1 when the weight
        is replicated over ``row_group``.

        Returns
        -------
        str
            Global ``in_features``, ``out_features`` and ``bias``, then ``row``
            and ``col``.
        """
        out_features, in_features = self.global_weight_shape
        return (
            f"in_features={in_features}, out_features={out_features}, "
            f"bias={self.bias is not None}, "
            + shard_repr(
                row=self.model_group.size() // self.col_group.size(),
                col=self.col_group,
            )
        )

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
    ) -> "DistributedLinear":
        """Convert a ``torch.nn.Linear`` to a :class:`DistributedLinear`.

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
            Unused. Accepts the options of the other converters, which
            :func:`jigsaw.convert.convert_to_jigsaw` passes to every layer.

        Returns
        -------
        DistributedLinear
            Distributed layer with the weights and bias of ``model``.

        Raises
        ------
        ValueError
            If the row split or the column group size does not divide the
            feature counts.
        """
        # The columns shard the input dimension of the weight and the (output-sized)
        # bias; the rows shard the output dimension of the weight, unless the model
        # group is the column group.
        row_split = _row_split(process_mesh_dict, col_kw, model_kw)
        if model.in_features % process_mesh_dict[col_kw].size() != 0:
            raise ValueError("in_features is not divisible by process mesh columns")
        if model.out_features % row_split != 0:
            raise ValueError(
                f"out_features is not divisible by the process mesh rows the "
                f"weight is split over ({row_split})"
            )
        if model.out_features % process_mesh_dict[col_kw].size() != 0:
            raise ValueError("out_features is not divisible by process mesh columns")

        instance = cls(
            in_dim=model.in_features,
            out_dim=model.out_features,
            bias=model.bias is not None,
            device=device,
            model_group=process_mesh_dict[model_kw],
            row_group=process_mesh_dict[row_kw],
            col_group=process_mesh_dict[col_kw],
        )

        # Map weights — clone so the original graph is detached
        row_rank = _row_rank(process_mesh_dict, row_kw, col_kw, model_kw)
        row_shard = model.out_features // row_split
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
            instance.bias.data = (
                model.bias.data[col_rank * bias_size : (col_rank + 1) * bias_size]
                .clone()
                .to(device)
            )
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        if _row_split(from_process_mesh, col_kw, model_kw) == 1:
            # Rows replicated over the row group: only the columns are sharded.
            global_weights = gather_1d_results(
                self.weight, from_process_mesh, dim=1, pm_key=col_kw
            )
        else:
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

        Raises
        ------
        ValueError
            If the target row and column group sizes do not divide the global
            weight and bias shapes.
        """
        # Find the local and global shapes. The weight rows are split over the
        # rows only if the model group is not the column group.
        from_row_split = _row_split(from_process_mesh, pm_col_key, pm_model_key)
        to_row_split = _row_split(to_process_mesh, pm_col_key, pm_model_key)
        from_weight_shape_local = self.weight.shape
        weight_shape_global = (
            from_weight_shape_local[0] * from_row_split,
            from_weight_shape_local[1] * from_process_mesh[pm_col_key].size(),
        )
        to_weight_shape_local = (
            weight_shape_global[0] // to_row_split,
            weight_shape_global[1] // to_process_mesh[pm_col_key].size(),
        )
        if weight_shape_global[0] % to_row_split != 0:
            raise ValueError(
                f"global weight shape[0] is {weight_shape_global[0]} which is not "
                f"divisible by {to_row_split}"
            )
        if weight_shape_global[1] % to_process_mesh[pm_col_key].size() != 0:
            raise ValueError(
                f"global weight shape[1] is {weight_shape_global[1]} which is not "
                f"divisible by {to_process_mesh[pm_col_key].size()}"
            )

        if self.bias is not None:
            from_bias_shape_local = self.bias.shape
            bias_shape_global = (
                from_bias_shape_local[0] * from_process_mesh[pm_col_key].size()
            )
            to_bias_shape_local = (
                bias_shape_global // to_process_mesh[pm_col_key].size()
            )

            if bias_shape_global % to_process_mesh[pm_col_key].size() != 0:
                raise ValueError(
                    f"global bias shape is {bias_shape_global} which is not "
                    f"divisible by {to_process_mesh[pm_col_key].size()}"
                )

        global_weights, global_bias = self.merge_weights(
            from_process_mesh, pm_row_key, pm_col_key, pm_model_key
        )

        row_rank, row_shard_size = (
            _row_rank(to_process_mesh, pm_row_key, pm_col_key, pm_model_key),
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
        # is valid: process groups, the kernel-selection config and the
        # bias replica mark all key off the new group sizes.
        self.row_group = to_process_mesh[pm_row_key]
        self.col_group = to_process_mesh[pm_col_key]
        self.model_group = to_process_mesh[pm_model_key]
        self.out_dim, self.in_dim = tuple(self.weight.shape)
        self.config = LinearConfig(
            self.model_group,
            linear_layer=True,
            device=self.device,
            comm=self.config.comm,
        )

        # The parameters are fresh, so the __init__ marks are gone; re-mark them
        # on the new row group and re-broadcast rank 0's slice.
        self._mark_row_replicas()


class _DistributedMatMul(torch.nn.Module):
    """Base class of the parameter-free distributed matmul modules.

    Subclasses set :attr:`_autograd_fn`.

    Parameters
    ----------
    device : torch.device
        Device of the module.
    model_group : dist.ProcessGroup or None
        Process group of one model-parallel instance. ``None`` means a
        single process.
    config : LinearConfig, optional
        Kernel selection. Defaults to a configuration built from the
        arguments.
    local_rank : int, optional
        Deprecated and ignored. The rank is taken from ``model_group``.

    Warns
    -----
    DeprecationWarning
        If ``local_rank`` is given.
    """

    #: autograd ``Function`` whose ``.apply`` this module drives; set by subclasses.
    _autograd_fn: type[torch.autograd.Function]

    def __init__(
        self,
        device: torch.device,
        model_group: dist.ProcessGroup | None,
        config: LinearConfig | None = None,
        *,
        local_rank: int | None = None,
    ) -> None:
        """Initialize the module."""
        super().__init__()
        _warn_local_rank(local_rank, type(self).__name__)
        self.device = device
        self.model_group = model_group
        self.config = (
            config
            if config is not None
            else LinearConfig(model_group, linear_layer=False, device=device)
        )
        self.matmul = self._autograd_fn.apply

    def get_parameter_count(self) -> int:
        """Return the number of parameters, which is always 0."""
        return 0

    def extra_repr(self) -> str:
        """Return the model-group size.

        Returns
        -------
        str
            ``model``.
        """
        return shard_repr(model=self.model_group)

    def forward(self, x: torch.Tensor, w: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply the distributed matmul.

        Parameters
        ----------
        x : torch.Tensor
            Local input.
        w : torch.Tensor
            Local weight.
        **kwargs : Any
            Unused. The operation has no bias.

        Returns
        -------
        torch.Tensor
            Local output.
        """
        cij = self.matmul(
            x,
            w,
            None,
            0 if self.model_group is None else self.model_group.rank(),
            self.device,
            self.config,
            self.model_group,
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
            device=device,
            model_group=process_mesh_dict[model_kw],
        )


class DistributedMatMulXWT(_DistributedMatMul):
    """Distributed batched matmul ``x @ w.mT``."""

    _autograd_fn = _XWT


class DistributedMatMulXW(_DistributedMatMul):
    """Distributed batched matmul ``x @ w``."""

    _autograd_fn = _XW


class DistributedMatMulXTW(_DistributedMatMul):
    """Distributed batched matmul ``x.mT @ w``."""

    _autograd_fn = _XTW


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
        """Return ``x @ w.mT``.

        Parameters
        ----------
        x : torch.Tensor
            Left operand.
        w : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            ``x @ w.mT``.
        """
        return x @ w.mT


class MatMulXW(_MatMul):
    """Batched matmul ``x @ w``."""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Return ``x @ w``.

        Parameters
        ----------
        x : torch.Tensor
            Left operand.
        w : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            ``x @ w``.
        """
        return x @ w


class MatMulXTW(_MatMul):
    """Batched matmul ``x.mT @ w``."""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Return ``x.mT @ w``.

        Parameters
        ----------
        x : torch.Tensor
            Left operand.
        w : torch.Tensor
            Right operand.

        Returns
        -------
        torch.Tensor
            ``x.mT @ w``.
        """
        return x.mT @ w
