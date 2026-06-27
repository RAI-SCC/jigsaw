"""Implementation of distributed normalization layers."""

from typing import Optional

import torch
import torch.distributed as dist

from . import functional


class DistributedLayerNorm(torch.nn.Module):
    """LayerNorm with synchronized gradients across a process group."""

    def __init__(
        self,
        local_dim: int,
        local_rank: int,
        device: torch.device,
        row_group: dist.ProcessGroup,
        col_group: dist.ProcessGroup,
    ):
        """Initialize torch module for distributed layer norm."""
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.spatial_group = row_group
        self.channel_group = col_group

        # Learnable parameters
        self.weight = torch.nn.Parameter(torch.ones(local_dim, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(local_dim, device=device))
        self.normalized_dim = local_dim
        self.eps = 1e-5
        self.layernorm = functional.DistributedLNFunction.apply

    def get_parameter_count(self) -> int:
        """Return parameter count for distributed layer norm."""
        return (self.weight.numel() + self.bias.numel()) // dist.get_world_size(
            group=self.spatial_group
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call forward function for distributed layer norm."""
        return self.layernorm(  # type: ignore[no-any-return]
            x, self.weight, self.bias, self.spatial_group, self.channel_group, self.eps
        )[0]


class DistributedGroupNorm(torch.nn.Module):
    """Distributed Group Norm wrapped in module."""

    def __init__(  # type: ignore[no-untyped-def]
        self,
        num_local_groups,  # type: ignore[no-untyped-def]
        num_channels,  # type: ignore[no-untyped-def]
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
        bias=True,
        channel_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()

        self.num_local_groups = num_local_groups
        self.num_channels = num_channels
        self.num_global_groups = num_local_groups * (
            channel_group.size() if channel_group is not None else 1
        )
        self.eps = eps
        self.affine = affine
        self.device = device
        self.dtype = dtype
        self.bias = bias
        self.group_norm = torch.nn.GroupNorm(
            self.num_local_groups,
            self.num_channels,
            eps=self.eps,
            affine=self.affine,
            device=self.device,
            dtype=self.dtype,
            bias=self.bias,
        )

    def reshard_parameters(
        self,
        from_process_mesh: dist.ProcessGroup,
        to_process_mesh: dict[str, dist.ProcessGroup],
        pm_row_key: str = "row",
        pm_col_key: str = "col",
    ) -> None:
        """Reshard parameters for a distributed group norm."""
        assert self.num_global_groups % to_process_mesh[pm_col_key] == 0, (
            "number of global LN groups must be divisible by to_process_mesh channel group size."
        )


class DistributedNoParamLayerNorm(torch.nn.Module):
    """LayerNorm with synchronized gradients across a process group."""

    def __init__(
        self, rank: int, device: torch.device, channel_group: dist.ProcessGroup
    ) -> None:
        """
        Initialize DistributedNoParamLayerNorm.

        inputs
        ------
        rank: int
        device: torch.device
        channel_group: dist.ProcessGroup
        """
        super().__init__()
        self.rank = rank
        self.device = device
        self.channel_group = channel_group

        self.eps = 1e-5
        self.layernorm = functional.DistributedNoParamLNFunction().apply

    def get_parameter_count(self) -> int:
        """Return 0 parameters in this layer norm."""
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call forward pass for layer norm."""
        return self.layernorm(x, self.channel_group, self.eps)[0]  # type: ignore[no-any-return]


class DistributedSoftMax(torch.nn.Module):
    """Distributed Softmax operation."""

    def __init__(self, group: dist.ProcessGroup, dim: int = -1):
        super().__init__()
        self.dim = dim
        self.group = group
        self.softmax = functional._DistributedSoftmax.apply

    def get_parameter_count(self) -> int:
        """Return parameter count of softmax."""
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of distributed softmax."""
        return self.softmax(x, self.dim, self.group)  # type: ignore[no-any-return]
