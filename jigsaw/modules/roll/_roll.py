"""Distributed roll modules."""

from typing import Any, Optional, cast

import torch
import torch.distributed as dist

from ._config import RollConfig
from ._functional import _Roll


class Roll(torch.nn.Module):
    """Wrapper of :func:`torch.roll` as a module."""

    def __init__(
        self,
        shifts: tuple,
        dims: tuple,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the module.

        Parameters
        ----------
        shifts : tuple[int, ...]
            Shift along each dimension of ``dims``.
        dims : tuple[int, ...]
            Dimensions to roll.
        dtype : torch.dtype
            Data type of the input.
        """
        super().__init__()
        self.shifts = shifts
        self.dims = dims
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Roll the input.

        Parameters
        ----------
        x : torch.Tensor
            Input.

        Returns
        -------
        torch.Tensor
            Rolled input.
        """
        return torch.roll(x, shifts=self.shifts, dims=self.dims)


class DistributedRoll(torch.nn.Module):
    """Roll of a tensor partitioned across process groups."""

    def __init__(
        self,
        shifts: tuple[int, ...],
        dims: tuple[int, ...],
        distributed_dims: tuple[int, ...],
        spatial_groups: tuple[dist.ProcessGroup, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        config: Optional[RollConfig] = None,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        shifts : tuple[int, ...]
            Shift along each dimension of ``dims``.
        dims : tuple[int, ...]
            Dimensions to roll.
        distributed_dims : tuple[int, ...]
            Dimensions across which the input is partitioned. Each must appear in
            ``dims``, counted from either end: ``-2`` matches ``2`` on a 4-D input.
        spatial_groups : tuple[dist.ProcessGroup, ...]
            Process group of each entry of ``distributed_dims``.
        device : torch.device
            Device of the module.
        dtype : torch.dtype
            Data type of the input.
        config : RollConfig, optional
            Kernel selection. Defaults to a configuration built from ``device``.

        Raises
        ------
        ValueError
            If ``distributed_dims`` and ``spatial_groups`` differ in length, or an entry
            of ``distributed_dims`` is not in ``dims``. If the two count from different
            ends, that is only known, and raised, in the forward pass.
        """
        super().__init__()
        if len(distributed_dims) != len(spatial_groups):
            raise ValueError("distributed_dims and spatial_groups must align")
        for d in distributed_dims:
            # A dimension counted from the other end can only be matched once the
            # number of input dimensions is known.
            same_end = all((e < 0) == (d < 0) for e in dims)
            if d not in dims and same_end:
                raise ValueError(f"distributed dim {d} must appear in dims {dims}")

        self.shifts = tuple(shifts)
        self.dims = tuple(dims)
        self.distributed_dims = tuple(distributed_dims)
        self.spatial_groups = tuple(spatial_groups)
        self.dtype = dtype
        self.empty = torch.empty(0, device=device, dtype=dtype)
        self.config = RollConfig(device) if config is None else config

    def get_parameter_count(self) -> int:
        """Return the number of local parameters."""
        return 0

    @classmethod
    def convert_to_jigsaw(
        cls,
        roll: Roll,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
        dim_to_mesh_axis: Optional[dict] = None,
        **kwargs: Any,
    ) -> "DistributedRoll":
        """Convert a :class:`Roll` to a :class:`DistributedRoll`.

        Parameters
        ----------
        roll : Roll
            Module to convert.
        process_mesh_dict : dict
            Process groups keyed by mesh axis.
        device : torch.device
            Device of the converted module.
        row_kw : str
            Key of the row process group.
        col_kw : str
            Key of the column process group.
        model_kw : str
            Key of the model process group.
        dim_to_mesh_axis : dict, optional
            Mesh axis across which each distributed dimension is partitioned.
            Defaults to ``{-2: row_kw}``.
        **kwargs : Any
            Unused. Accepts the options of the other converters, which
            :func:`jigsaw.convert.convert_to_jigsaw` passes to every layer.

        Returns
        -------
        DistributedRoll
            Distributed module with the shifts and dimensions of ``roll``.
        """
        if dim_to_mesh_axis is None:
            dim_to_mesh_axis = {-2: row_kw}
        distributed_dims = tuple(dim_to_mesh_axis.keys())
        spatial_groups = tuple(
            (
                process_mesh_dict[ax].get_group()
                if hasattr(process_mesh_dict[ax], "get_group")
                else process_mesh_dict[ax]
            )
            for ax in dim_to_mesh_axis.values()
        )
        return cls(
            shifts=roll.shifts,
            dims=roll.dims,
            distributed_dims=distributed_dims,
            spatial_groups=spatial_groups,
            dtype=roll.dtype,
            device=device,
        )

    def convert_to_torch(self) -> dict:
        """Convert the module to the arguments of :func:`torch.roll`.

        Returns
        -------
        dict
            ``shifts``, ``dims`` and ``dtype`` of the module.
        """
        return {"shifts": self.shifts, "dims": self.dims, "dtype": self.dtype}

    def reshard(
        self,
        distributed_dims: list[int],
        spatial_groups: list[dist.ProcessGroup],
    ) -> None:
        """Reshard the module onto other process groups.

        Parameters
        ----------
        distributed_dims : list[int]
            Dimensions across which the input is partitioned.
        spatial_groups : list[dist.ProcessGroup]
            Process group of each entry of ``distributed_dims``.

        Raises
        ------
        ValueError
            If ``distributed_dims`` and ``spatial_groups`` differ in length.
        """
        if len(distributed_dims) != len(spatial_groups):
            raise ValueError("mismatched lengths")
        self.distributed_dims = tuple(distributed_dims)
        self.spatial_groups = tuple(spatial_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Roll the local shard and exchange the halos.

        Parameters
        ----------
        x : torch.Tensor
            Local input.

        Returns
        -------
        torch.Tensor
            Local output.
        """
        out = _Roll.apply(
            x,
            self.shifts,
            self.dims,
            self.distributed_dims,
            self.spatial_groups,
            self.empty,
            self.config,
        )
        out = cast(torch.Tensor, out)
        return out
