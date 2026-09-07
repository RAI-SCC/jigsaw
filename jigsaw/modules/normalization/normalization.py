"""Implementation of distributed normalization layers."""

from typing import Optional

import torch
import torch.distributed as dist

from jigsaw.utils_aggregate import gather_1d_results

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

    def reshard_parameters(
        self,
        from_process_mesh: dict[str, dist.ProcessGroup],
        to_process_mesh: dict[str, dist.ProcessGroup],
        pm_row_key: str = "row",
        pm_col_key: str = "col",
    ) -> None:
        """
        Re-slice the channel-sharded ``weight`` / ``bias`` onto ``to_process_mesh``.

        Both are ``(local_dim,)`` sharded over the channel (``col``) group. The
        gathered global parameters are unchanged; the module is rebound to
        ``to_process_mesh``.
        """
        from_col, to_col = from_process_mesh[pm_col_key], to_process_mesh[pm_col_key]
        normalized_global = self.normalized_dim * from_col.size()
        assert normalized_global % to_col.size() == 0, (
            "global normalized dim must be divisible by the target channel group size"
        )
        new_local_dim = normalized_global // to_col.size()
        col_rank = to_col.rank()

        global_weight = gather_1d_results(
            self.weight, from_process_mesh, dim=0, pm_key=pm_col_key
        )
        global_bias = gather_1d_results(
            self.bias, from_process_mesh, dim=0, pm_key=pm_col_key
        )
        self.weight = torch.nn.Parameter(
            global_weight[
                col_rank * new_local_dim : (col_rank + 1) * new_local_dim
            ].contiguous()
        )
        self.bias = torch.nn.Parameter(
            global_bias[
                col_rank * new_local_dim : (col_rank + 1) * new_local_dim
            ].contiguous()
        )
        self.normalized_dim = new_local_dim
        self.spatial_group = to_process_mesh[pm_row_key]
        self.channel_group = to_col
        self.local_rank = to_process_mesh.get("model", to_col).rank()

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
        """Initialize the object.

        Parameters
        ----------
        num_local_groups : Any
            Number of normalization groups local to each rank.
        num_channels : Any
            Number of channels normalized by the layer.
        eps : Any
            Numerical stability epsilon.
        affine : Any
            Whether the normalization layer has learnable affine parameters.
        device : Any
            Device on which tensors and communication are placed.
        dtype : Any
            Data type used for tensor values.
        bias : Any
            Optional bias parameter.
        channel_group : Optional[dist.ProcessGroup]
            Process group used for channel or tensor parallelism.

        """
        super().__init__()

        self.num_local_groups = num_local_groups
        self.num_channels = num_channels
        self.channel_group = channel_group
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize this rank's channel shard.

        ``num_global_groups = num_local_groups * channel_group.size()``, so every
        channel-parallel rank owns whole normalization groups and a plain local
        ``GroupNorm`` on the channel shard needs no cross-rank statistics.
        """
        return self.group_norm(x)  # type: ignore[no-any-return]

    def reshard_parameters(
        self,
        from_process_mesh: dict[str, dist.ProcessGroup],
        to_process_mesh: dict[str, dist.ProcessGroup],
        pm_row_key: str = "row",
        pm_col_key: str = "col",
    ) -> None:
        """
        Re-slice the channel-sharded affine parameters onto ``to_process_mesh``.

        ``weight`` / ``bias`` are ``(num_channels_local,)`` sharded over the
        channel (``col``) group. The global affine parameters are unchanged; the
        inner ``torch.nn.GroupNorm`` is rebuilt for the new local group / channel
        counts and the module is rebound to ``to_process_mesh``.
        """
        to_channel = to_process_mesh[pm_col_key]
        assert self.num_global_groups % to_channel.size() == 0, (
            "num_global_groups must be divisible by the target channel group size"
        )
        channels_global = self.num_channels * from_process_mesh[pm_col_key].size()
        assert channels_global % to_channel.size() == 0, (
            "global channel count must be divisible by the target channel group size"
        )

        new_num_channels = channels_global // to_channel.size()
        new_num_local_groups = self.num_global_groups // to_channel.size()
        col_rank = to_channel.rank()

        weight_slice = bias_slice = None
        if self.affine:
            global_weight = gather_1d_results(
                self.group_norm.weight, from_process_mesh, dim=0, pm_key=pm_col_key
            )
            weight_slice = global_weight[
                col_rank * new_num_channels : (col_rank + 1) * new_num_channels
            ].clone()
            if self.group_norm.bias is not None:
                global_bias = gather_1d_results(
                    self.group_norm.bias, from_process_mesh, dim=0, pm_key=pm_col_key
                )
                bias_slice = global_bias[
                    col_rank * new_num_channels : (col_rank + 1) * new_num_channels
                ].clone()

        self.num_channels = new_num_channels
        self.num_local_groups = new_num_local_groups
        self.channel_group = to_channel
        self.group_norm = torch.nn.GroupNorm(
            self.num_local_groups,
            self.num_channels,
            eps=self.eps,
            affine=self.affine,
            device=self.device,
            dtype=self.dtype,
            bias=self.bias,
        )
        if weight_slice is not None:
            self.group_norm.weight.data = weight_slice
        if bias_slice is not None and self.group_norm.bias is not None:
            self.group_norm.bias.data = bias_slice


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
        """Initialize the object.

        Parameters
        ----------
        group : dist.ProcessGroup
            Process group used for distributed communication.
        dim : int
            Dimension along which the operation is applied.

        """
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
