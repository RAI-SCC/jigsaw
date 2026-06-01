from typing import Callable, List, Tuple

import torch
import torch.distributed as dist

from . import distributed_roll as dist_roll


class Roll(torch.nn.Module):
    """Wrap torch roll function in a module."""

    def __init__(self, shifts, dims, dtype=torch.float32):
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

    def forward(self, x):
        """
        Forward pass of torch roll.

        Parameters
        ----------
        x: torch.tensor
            data tensor to roll

        Returns
        -------
        x: torch.tensor
            rolled data
        """
        return torch.roll(x, shifts=self.shifts, dims=self.dims)


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


class DistributedRoll(torch.nn.Module):
    """Distributed roll across an arbitrary set of sharded tensor dims."""

    def __init__(
        self,
        shifts: Tuple[int, ...],
        dims: Tuple[int, ...],
        distributed_dims: List[int],
        spatial_process_groups: List[dist.ProcessGroup],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if len(distributed_dims) != len(spatial_process_groups):
            raise ValueError("distributed_dims and spatial_process_groups must align")
        for d in distributed_dims:
            if d not in dims:
                raise ValueError(f"distributed dim {d} must appear in dims {dims}")

        self.shifts = tuple(shifts)
        self.dims = tuple(dims)
        self.distributed_dims = tuple(distributed_dims)
        self.spatial_process_groups = tuple(spatial_process_groups)
        self.device = device
        self.dtype = dtype
        self.empty = torch.empty(0, device=device, dtype=dtype)

    def get_parameter_count(self):
        """Return dummy parameter count of roll."""
        return 0

    @classmethod
    def convert_to_jigsaw(
        cls,
        roll: Roll,
        process_mesh: dist.ProcessGroup,
        device: torch.device,
        spatial_kw: str = "row",
        channel_kw: str = "col",
        model_kw: str = "model",
        dim_to_mesh_axis: dict = {-2: "col"},
    ):
        """Build a DistributedRoll from a process mesh + mapping of tensor dim -> mesh axis."""
        distributed_dims = list(dim_to_mesh_axis.keys())
        spatial_groups = [
            (
                process_mesh[ax].get_group()
                if hasattr(process_mesh[ax], "get_group")
                else process_mesh[ax]
            )
            for ax in dim_to_mesh_axis.values()
        ]
        return cls(
            shifts=roll.shifts,
            dims=roll.dims,
            distributed_dims=distributed_dims,
            spatial_process_groups=spatial_groups,
            device=device,
            dtype=roll.dtype,
        )

    def convert_to_torch(self):
        """Return an equivalent non-distributed torch.roll-style spec."""
        return {"shifts": self.shifts, "dims": self.dims, "dtype": self.dtype}

    def reshard(
        self,
        distributed_dims: List[int],
        spatial_process_groups: List[dist.ProcessGroup],
    ):
        """Update sharding without rebuilding the module."""
        if len(distributed_dims) != len(spatial_process_groups):
            raise ValueError("mismatched lengths")
        self.distributed_dims = tuple(distributed_dims)
        self.spatial_process_groups = tuple(spatial_process_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of roll function."""
        return RollFunction.apply(
            x,
            self.shifts,
            self.dims,
            self.distributed_dims,
            self.spatial_process_groups,
            self.device,
            self.empty,
        )


class RollFunction(torch.autograd.Function):
    """Autograd wrapper for distributed roll over an arbitrary process mesh."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        shifts: Tuple[int, ...],
        dims: Tuple[int, ...],
        distributed_dims: Tuple[int, ...],
        spatial_process_groups: Tuple[dist.ProcessGroup, ...],
        device: torch.device,
        empty: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of distributed roll."""
        return dist_roll.distributed_roll(
            x,
            shifts,
            dims,
            list(distributed_dims),
            list(spatial_process_groups),
            device,
            empty,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Context for roll function."""
        x, shifts, dims, distributed_dims, spatial_process_groups, device, empty = (
            inputs
        )
        ctx.shifts = shifts
        ctx.dims = dims
        ctx.distributed_dims = distributed_dims
        ctx.spatial_process_groups = spatial_process_groups
        ctx.device = device
        ctx.empty = empty

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of distributed roll."""
        neg_shifts = tuple(-s for s in ctx.shifts)
        grad_input = dist_roll.distributed_roll(
            grad_output,
            neg_shifts,
            ctx.dims,
            list(ctx.distributed_dims),
            list(ctx.spatial_process_groups),
            ctx.device,
            ctx.empty,
        )
        return grad_input, None, None, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims,
        x,
        shifts,
        dims,
        distributed_dims,
        spatial_process_groups,
        device,
        empty,
    ):
        """Vmap for dealing with roll operations."""
        out = Roll.apply(
            x, shifts, dims, distributed_dims, spatial_process_groups, device, empty
        )
        return out, 0
