"""Distributed GroupNorm."""

from typing import Any, Optional

import torch
import torch.distributed as dist

from jigsaw.utils_aggregate import gather_1d_results


class DistributedGroupNorm(torch.nn.Module):
    """GroupNorm over a channel-sharded input."""

    def __init__(
        self,
        num_local_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        bias: bool = True,
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

    @classmethod
    def convert_to_jigsaw(
        cls,
        model: torch.nn.GroupNorm,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
        **kwargs: Any,
    ) -> "DistributedGroupNorm":
        """Convert a ``torch.nn.GroupNorm`` to a :class:`DistributedGroupNorm`.

        Channels are partitioned across ``col_kw`` so that no normalization
        group straddles two shards: this requires ``num_groups`` itself (not
        just ``num_channels``) to divide the column group size.

        Parameters
        ----------
        model : torch.nn.GroupNorm
            Layer to convert.
        process_mesh_dict : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        device : torch.device
            Device of the converted module.
        row_kw : str
            Key of the row process group. Unused: the layer has no collective
            to reduce over it, since each shard normalizes whole groups
            independently.
        col_kw : str
            Key of the column process group across which the channels (and
            thus groups) are partitioned.
        model_kw : str
            Key of the model process group. Unused.
        **kwargs : Any
            Unused. Accepts the options of the other converters, which
            :func:`jigsaw.convert.convert_to_jigsaw` passes to every layer.

        Returns
        -------
        DistributedGroupNorm
            Distributed layer with the weight and bias of ``model``.

        Raises
        ------
        AssertionError
            If ``num_groups`` or ``num_channels`` is not divisible by the
            column group size.
        """
        col_group = process_mesh_dict[col_kw]
        assert model.num_groups % col_group.size() == 0, (
            "num_groups is not divisible by process mesh columns"
        )
        assert model.num_channels % col_group.size() == 0, (
            "num_channels is not divisible by process mesh columns"
        )
        local_groups = model.num_groups // col_group.size()
        local_channels = model.num_channels // col_group.size()
        col_rank = col_group.rank()

        instance = cls(
            num_local_groups=local_groups,
            num_channels=local_channels,
            eps=model.eps,
            affine=model.affine,
            device=device,
            dtype=model.weight.dtype if model.weight is not None else None,
            bias=model.bias is not None,
            channel_group=col_group,
        )
        if model.affine:
            instance.group_norm.weight.data = (
                model.weight.data[
                    col_rank * local_channels : (col_rank + 1) * local_channels
                ]
                .clone()
                .to(device)
            )
            if model.bias is not None:
                instance.group_norm.bias.data = (
                    model.bias.data[
                        col_rank * local_channels : (col_rank + 1) * local_channels
                    ]
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
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Merge the sharded weight and bias into global tensors.

        Parameters
        ----------
        from_process_mesh : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        row_kw : str
            Key of the row process group. Unused.
        col_kw : str
            Key of the column process group.
        model_kw : str
            Key of the model process group. Unused.

        Returns
        -------
        global_weight : torch.Tensor | None
            Merged weight, or ``None`` if the layer is not affine.
        global_bias : torch.Tensor | None
            Merged bias, or ``None`` if the layer is not affine or has no bias.
        """
        if not self.affine:
            return None, None
        global_weight = gather_1d_results(
            self.group_norm.weight, from_process_mesh, dim=0, pm_key=col_kw
        )
        global_bias = None
        if self.group_norm.bias is not None:
            global_bias = gather_1d_results(
                self.group_norm.bias, from_process_mesh, dim=0, pm_key=col_kw
            )
        return global_weight, global_bias

    def convert_to_torch(
        self,
        process_mesh_dict: dict,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> torch.nn.GroupNorm:
        """Convert the layer to a ``torch.nn.GroupNorm``.

        Parameters
        ----------
        process_mesh_dict : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        row_kw : str
            Key of the row process group. Unused.
        col_kw : str
            Key of the column process group.
        model_kw : str
            Key of the model process group. Unused.

        Returns
        -------
        torch.nn.GroupNorm
            Layer with the merged weight and bias.
        """
        weight, bias = self.merge_weights(process_mesh_dict, row_kw, col_kw, model_kw)
        num_channels = self.num_channels * process_mesh_dict[col_kw].size()
        group_norm = torch.nn.GroupNorm(
            self.num_global_groups,
            num_channels,
            eps=self.eps,
            affine=self.affine,
            device=self.device,
            dtype=self.dtype,
        )
        if self.affine:
            assert weight is not None
            group_norm.weight.data = weight
            if bias is not None:
                group_norm.bias.data = bias
            elif group_norm.bias is not None:
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
