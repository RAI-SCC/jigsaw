"""Module definitions of roll functions."""

from typing import Any, Callable, Optional, Protocol, cast

import torch
import torch.distributed as dist

from . import comm
from . import distributed_roll as dist_roll


class Roll(torch.nn.Module):
    """Wrap torch roll function in a module."""

    def __init__(
        self,
        shifts: tuple,
        dims: tuple,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize roll function.

        Parameters
        ----------
        shifts: tuple, list
            number of indices to shift by
        dims: tuple, list
            dims to shift by
        dtype: torch.dtype
            data type expected of input data
        """
        super().__init__()
        self.shifts = shifts
        self.dims = dims
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of torch roll.

        Parameters
        ----------
        x: torch.Tensor
            data tensor to roll

        Returns
        -------
        x: torch.Tensor
            rolled data
        """
        return torch.roll(x, shifts=self.shifts, dims=self.dims)


class RollConfig:
    """Selection strategy for roll function."""

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
    ):
        self.comm_halo_1d_fn = self._comm_halo_1d_fn_selection(device)

    def _comm_halo_1d_fn_selection(self, device: torch.device) -> Callable:
        """
        Select comm call for halo 1d function.

        If cuda, comm will occur via all-to-all.
        If cpu, comm will occur via p2p.
        """
        if device.type == "cuda":
            return comm.comm_1d_alltoall
        elif device.type == "cpu":
            return comm.comm_1d_p2p
        else:
            raise ValueError(f"No halo 1d function possible for device {device.type}")


class DistributedRoll(torch.nn.Module):
    """Distributed roll across an arbitrary set of sharded tensor dims."""

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
        super().__init__()
        if len(distributed_dims) != len(spatial_groups):
            raise ValueError("distributed_dims and spatial_groups must align")
        for d in distributed_dims:
            if d not in dims:
                raise ValueError(f"distributed dim {d} must appear in dims {dims}")

        self.shifts = tuple(shifts)
        self.dims = tuple(dims)
        self.distributed_dims = tuple(distributed_dims)
        self.spatial_groups = tuple(spatial_groups)
        self.dtype = dtype
        self.empty = torch.empty(0, device=device, dtype=dtype)
        self.config = RollConfig(device) if config is None else None

    def get_parameter_count(self) -> int:
        """Return dummy parameter count of roll."""
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
        dim_to_mesh_axis: dict = {-2: "row"},
    ) -> "DistributedRoll":
        """Build a DistributedRoll from a process mesh + mapping of tensor dim -> mesh axis."""
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
        """Return an equivalent non-distributed torch.roll-style spec."""
        return {"shifts": self.shifts, "dims": self.dims, "dtype": self.dtype}

    def reshard(
        self,
        distributed_dims: list[int],
        spatial_groups: list[dist.ProcessGroup],
    ) -> None:
        """Update sharding without rebuilding the module."""
        if len(distributed_dims) != len(spatial_groups):
            raise ValueError("mismatched lengths")
        self.distributed_dims = tuple(distributed_dims)
        self.spatial_groups = tuple(spatial_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of roll function."""
        out = RollFunction.apply(
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


class RollCtx(Protocol):
    """Protocol for types stored within ctx."""

    saved_tensors: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
    local_rank: int
    shifts: tuple[int, ...]
    dims: tuple[int, ...]
    distributed_dims: tuple[int, ...]
    spatial_groups: tuple[dist.ProcessGroup, ...]
    empty: torch.Tensor
    config: RollConfig


class RollFunction(torch.autograd.Function):
    """Autograd wrapper for distributed roll over an arbitrary process mesh."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        shifts: tuple[int, ...],
        dims: tuple[int, ...],
        distributed_dims: tuple[int, ...],
        spatial_groups: tuple[dist.ProcessGroup, ...],
        empty: torch.Tensor,
        config: RollConfig,
    ) -> torch.Tensor:
        """Forward pass of distributed roll."""
        return dist_roll.distributed_roll(
            x,
            shifts,
            dims,
            list(distributed_dims),
            list(spatial_groups),
            empty,
            config.comm_halo_1d_fn,
        )

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: tuple[
            torch.Tensor,
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[dist.ProcessGroup, ...],
            torch.Tensor,
            RollConfig,
        ],
        output: torch.Tensor,
    ) -> None:
        """Context for roll function."""
        _, shifts, dims, distributed_dims, spatial_groups, empty, config = inputs
        c = cast(RollCtx, ctx)
        c.shifts = shifts
        c.dims = dims
        c.distributed_dims = distributed_dims
        c.spatial_groups = spatial_groups
        c.empty = empty
        c.config = config

    @staticmethod
    def backward(
        ctx: RollCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        """Backward pass of distributed roll."""
        neg_shifts = tuple(-s for s in ctx.shifts)
        grad_input = dist_roll.distributed_roll(
            grad_output,
            neg_shifts,
            ctx.dims,
            list(ctx.distributed_dims),
            list(ctx.spatial_groups),
            ctx.empty,
            ctx.config.comm_halo_1d_fn,
        )
        return grad_input, None, None, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        shifts: tuple[int, ...],
        dims: tuple[int, ...],
        distributed_dims: tuple[int, ...],
        spatial_groups: tuple[dist.ProcessGroup, ...],
        empty: torch.Tensor,
        config: RollConfig,
    ) -> tuple[torch.Tensor, int]:
        """Vmap for dealing with roll operations."""
        out = RollFunction.apply(
            x, shifts, dims, distributed_dims, spatial_groups, empty, config
        )
        return out, 0
