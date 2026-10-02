"""Distributed normalization layers."""

from typing import Any, Optional

import torch
import torch.distributed as dist

from jigsaw.utils_aggregate import gather_1d_results

from . import functional


class DistributedLayerNorm(torch.nn.Module):
    """LayerNorm over a channel-sharded last dimension."""

    def __init__(
        self,
        local_dim: int,
        local_rank: int,
        device: torch.device,
        row_group: dist.ProcessGroup,
        col_group: dist.ProcessGroup,
    ):
        """Initialize the layer.

        Parameters
        ----------
        local_dim : int
            Size of the local shard of the normalized dimension.
        local_rank : int
            Rank within the model group.
        device : torch.device
            Device of the module.
        row_group : dist.ProcessGroup
            Process group over which the parameter gradients are reduced.
        col_group : dist.ProcessGroup
            Process group across which the normalized dimension is partitioned.
        """
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
        """Return the number of local parameters."""
        return (self.weight.numel() + self.bias.numel()) // dist.get_world_size(
            group=self.spatial_group
        )

    @classmethod
    def convert_to_jigsaw(
        cls,
        model: torch.nn.LayerNorm,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
        **kwargs: Any,
    ) -> "DistributedLayerNorm":
        """Convert a ``torch.nn.LayerNorm`` to a :class:`DistributedLayerNorm`.

        Parameters
        ----------
        model : torch.nn.LayerNorm
            Layer to convert. Must normalize a single trailing dimension (no
            multi-axis ``normalized_shape``) and have both a weight and a bias
            (``elementwise_affine=True``, ``bias=True``).
        process_mesh_dict : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        device : torch.device
            Device of the converted module.
        row_kw : str
            Key of the row process group. The converted layer replicates its
            weight and bias across this group, and reduces their gradients
            over it -- correct only if the input is also replicated (not
            sharded) across the same group, as it is for the row/column split
            used by :func:`jigsaw.convert.convert_to_jigsaw`.
        col_kw : str
            Key of the column process group across which the normalized
            dimension is partitioned.
        model_kw : str
            Key of the model process group.
        **kwargs : Any
            Unused. Accepts the options of the other converters, which
            :func:`jigsaw.convert.convert_to_jigsaw` passes to every layer.

        Returns
        -------
        DistributedLayerNorm
            Distributed layer with the weight and bias of ``model``.

        Raises
        ------
        NotImplementedError
            If ``model`` normalizes more than one dimension, or has no weight
            or no bias -- convert those by hand (e.g. onto
            :class:`DistributedNoParamLayerNorm`).
        """
        if len(model.normalized_shape) != 1:
            raise NotImplementedError(
                "DistributedLayerNorm only normalizes a single trailing "
                f"dimension; got normalized_shape={model.normalized_shape}"
            )
        if model.weight is None or model.bias is None:
            raise NotImplementedError(
                "DistributedLayerNorm always has both a weight and a bias; "
                "convert a LayerNorm without one of them by hand"
            )

        normalized_dim = model.normalized_shape[0]
        col_group = process_mesh_dict[col_kw]
        assert normalized_dim % col_group.size() == 0, (
            "normalized_shape is not divisible by process mesh columns"
        )
        local_dim = normalized_dim // col_group.size()
        col_rank = col_group.rank()

        instance = cls(
            local_dim=local_dim,
            local_rank=process_mesh_dict[model_kw].rank(),
            device=device,
            row_group=process_mesh_dict[row_kw],
            col_group=col_group,
        )
        instance.eps = model.eps
        instance.weight.data = (
            model.weight.data[col_rank * local_dim : (col_rank + 1) * local_dim]
            .clone()
            .to(device)
        )
        instance.bias.data = (
            model.bias.data[col_rank * local_dim : (col_rank + 1) * local_dim]
            .clone()
            .to(device)
        )
        return instance

    def merge_weights(
        self,
        from_process_mesh: dict,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge the sharded weight and bias into global tensors.

        Parameters
        ----------
        from_process_mesh : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        row_kw : str
            Key of the row process group. Unused: the weight and bias are
            already replicated (not sharded) across it.
        col_kw : str
            Key of the column process group.
        model_kw : str
            Key of the model process group. Unused.

        Returns
        -------
        global_weight : torch.Tensor
            Merged weight.
        global_bias : torch.Tensor
            Merged bias.
        """
        global_weight = gather_1d_results(
            self.weight, from_process_mesh, dim=0, pm_key=col_kw
        )
        global_bias = gather_1d_results(
            self.bias, from_process_mesh, dim=0, pm_key=col_kw
        )
        return global_weight, global_bias

    def convert_to_torch(
        self,
        process_mesh_dict: dict,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> torch.nn.LayerNorm:
        """Convert the layer to a ``torch.nn.LayerNorm``.

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
        torch.nn.LayerNorm
            Layer with the merged weight and bias.
        """
        weight, bias = self.merge_weights(process_mesh_dict, row_kw, col_kw, model_kw)
        layer_norm = torch.nn.LayerNorm(
            weight.shape[0], eps=self.eps, device=self.device, dtype=weight.dtype
        )
        layer_norm.weight.data = weight
        layer_norm.bias.data = bias
        return layer_norm

    def reshard_parameters(
        self,
        from_process_mesh: dict[str, dist.ProcessGroup],
        to_process_mesh: dict[str, dist.ProcessGroup],
        pm_row_key: str = "row",
        pm_col_key: str = "col",
    ) -> None:
        """Reshard the weight and bias from one process mesh onto another.

        The module is rebound to the new mesh.

        Parameters
        ----------
        from_process_mesh : dict[str, dist.ProcessGroup]
            Process groups of the current mesh.
        to_process_mesh : dict[str, dist.ProcessGroup]
            Process groups of the new mesh.
        pm_row_key, pm_col_key : str
            Keys of the row and column process groups in both meshes.
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
        return self.layernorm(  # type: ignore[no-any-return]
            x, self.weight, self.bias, self.spatial_group, self.channel_group, self.eps
        )[0]


class DistributedGroupNorm(torch.nn.Module):
    """GroupNorm over a channel-sharded input."""

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
        """Initialize the layer.

        Parameters
        ----------
        num_local_groups : int
            Number of normalization groups of the local shard.
        num_channels : int
            Number of channels of the local shard.
        eps : float
            Term added to the variance for numerical stability.
        affine : bool
            Whether the layer has learnable affine parameters.
        device : torch.device, optional
            Device of the module.
        dtype : torch.dtype, optional
            Data type of the parameters.
        bias : bool
            Whether the affine transformation has a bias.
        channel_group : dist.ProcessGroup, optional
            Process group across which the channels are partitioned.
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
        self.group_norm = self._build_group_norm()

    def _build_group_norm(self) -> torch.nn.GroupNorm:
        """Return a ``GroupNorm`` of the current local shape.

        Older ``torch.nn.GroupNorm`` (e.g. 2.9) has no ``bias=`` argument, so a
        disabled bias is dropped here instead, as newer torch does for ``bias=False``.
        """
        group_norm = torch.nn.GroupNorm(
            self.num_local_groups,
            self.num_channels,
            eps=self.eps,
            affine=self.affine,
            device=self.device,
            dtype=self.dtype,
        )
        if self.affine and not self.bias:
            group_norm.register_parameter("bias", None)
        return group_norm

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
        return self.group_norm(x)  # type: ignore[no-any-return]

    def reshard_parameters(
        self,
        from_process_mesh: dict[str, dist.ProcessGroup],
        to_process_mesh: dict[str, dist.ProcessGroup],
        pm_row_key: str = "row",
        pm_col_key: str = "col",
    ) -> None:
        """Reshard the affine parameters from one process mesh onto another.

        The module is rebound to the new mesh.

        Parameters
        ----------
        from_process_mesh : dict[str, dist.ProcessGroup]
            Process groups of the current mesh.
        to_process_mesh : dict[str, dist.ProcessGroup]
            Process groups of the new mesh.
        pm_row_key, pm_col_key : str
            Keys of the row and column process groups in both meshes.
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
        self.group_norm = self._build_group_norm()
        if weight_slice is not None:
            self.group_norm.weight.data = weight_slice
        if bias_slice is not None and self.group_norm.bias is not None:
            self.group_norm.bias.data = bias_slice


class DistributedNoParamLayerNorm(torch.nn.Module):
    """LayerNorm without parameters over a channel-sharded last dimension."""

    def __init__(
        self, rank: int, device: torch.device, channel_group: dist.ProcessGroup
    ) -> None:
        """Initialize the layer.

        Parameters
        ----------
        rank : int
            Rank within the model group.
        device : torch.device
            Device of the module.
        channel_group : dist.ProcessGroup
            Process group across which the normalized dimension is partitioned.
        """
        super().__init__()
        self.rank = rank
        self.device = device
        self.channel_group = channel_group

        self.eps = 1e-5
        self.layernorm = functional.DistributedNoParamLNFunction.apply

    def get_parameter_count(self) -> int:
        """Return the number of local parameters."""
        return 0

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
        return self.layernorm(x, self.channel_group, self.eps)[0]  # type: ignore[no-any-return]


class DistributedSoftMax(torch.nn.Module):
    """Softmax over a sharded dimension."""

    def __init__(self, group: dist.ProcessGroup, dim: int = -1):
        """Initialize the layer.

        Parameters
        ----------
        group : dist.ProcessGroup
            Process group across which ``dim`` is partitioned.
        dim : int
            Dimension along which the softmax is applied.
        """
        super().__init__()
        self.dim = dim
        self.group = group
        self.softmax = functional._DistributedSoftmax.apply

    def get_parameter_count(self) -> int:
        """Return the number of local parameters."""
        return 0

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
        return self.softmax(x, self.dim, self.group)  # type: ignore[no-any-return]
