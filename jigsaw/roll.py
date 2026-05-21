import torch
import torch.distributed as dist
from typing import Callable, Tuple

from . import distributed_roll as dist_roll


class Roll(torch.nn.Module):
    def __init__(self, shifts, dims, dtype=torch.float32):
        super().__init__()
        self.shifts = shifts
        self.dims = dims
        self.dtype=dtype

    def forward(self, x):
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

    @classmethod
    def convert_to_jigsaw(
        cls,
        roll: Roll,
        process_mesh: dist.ProcessGroup,
        device: torch.device,
        spatial_kw: str = "row",
        channel_kw: str = "col",
        model_kw: str = "model",
    ):
        instance = cls(
            shifts=roll.shifts,
            dims=roll.dims,
            local_rank=process_mesh["model"].rank(),
            device=device,
            parallelism=process_mesh["model"].size(),
            spatial_group=process_mesh[spatial_kw],
            dtype=roll.dtype
        )
        
        return instance
    
    def convert_to_vanilla(
        self,
        process_mesh: dist.ProcessGroup,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> Roll:
        roll = Roll(
            shifts=self.shifts,
            dims=self.dims,
            dtype=self.dtype
        )
        return roll

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
