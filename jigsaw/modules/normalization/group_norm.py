"""Distributed GroupNorm."""

from typing import Any, Optional

import torch
import torch.distributed as dist

from jigsaw.gather import gather_1d_results


class DistributedGroupNorm(torch.nn.Module):
    """GroupNorm over a channel-sharded input."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        device: torch.device,
        channel_group: Optional[dist.ProcessGroup] = None,
        eps: float = 1e-5,
        affine: bool = True,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the layer.

        Parameters
        ----------
        num_groups : int
            Global (unsharded) number of normalization groups. Must be
            divisible by ``channel_group.size()``, so that no group straddles
            two shards.
        num_channels : int
            Global (unsharded) number of channels. Must be divisible by
            ``channel_group.size()``.
        device : torch.device
            Device of the module.
        channel_group : dist.ProcessGroup, optional
            Process group across which the channels are partitioned. ``None``
            means the channels are not partitioned.
        eps : float
            Term added to the variance for numerical stability.
        affine : bool
            Whether the layer has learnable affine parameters.
        bias : bool
            Whether the affine transformation has a bias.
        dtype : torch.dtype, optional
            Data type of the parameters.
        """
        super().__init__()

        channel_size = channel_group.size() if channel_group is not None else 1
        assert num_groups % channel_size == 0 and num_channels % channel_size == 0, (
            f"num_groups={num_groups} and num_channels={num_channels} must both "
            f"be divisible by the channel group size {channel_size}"
        )
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.num_local_groups = num_groups // channel_size
        self.num_local_channels = num_channels // channel_size
        self.channel_group = channel_group
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
            self.num_local_channels,
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
        instance = cls(
            num_groups=model.num_groups,
            num_channels=model.num_channels,
            device=device,
            channel_group=col_group,
            eps=model.eps,
            affine=model.affine,
            bias=model.bias is not None,
            dtype=model.weight.dtype if model.weight is not None else None,
        )
        local_channels = instance.num_local_channels
        col_rank = col_group.rank()
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
        group_norm = torch.nn.GroupNorm(
            self.num_groups,
            self.num_channels,
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
        assert self.num_groups % to_channel.size() == 0, (
            "num_groups must be divisible by the target channel group size"
        )
        assert self.num_channels % to_channel.size() == 0, (
            "num_channels must be divisible by the target channel group size"
        )

        new_num_channels = self.num_channels // to_channel.size()
        new_num_local_groups = self.num_groups // to_channel.size()
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

        self.num_local_channels = new_num_channels
        self.num_local_groups = new_num_local_groups
        self.channel_group = to_channel
        self.group_norm = self._build_group_norm()
        if weight_slice is not None:
            self.group_norm.weight.data = weight_slice
        if bias_slice is not None and self.group_norm.bias is not None:
            self.group_norm.bias.data = bias_slice
