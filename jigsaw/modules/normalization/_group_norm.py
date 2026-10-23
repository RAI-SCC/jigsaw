"""Distributed GroupNorm."""

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from jigsaw import parallel
from jigsaw.gather import gather_1d_results

from . import _functional as functional


def _sync_groups(
    spatial_group: "dist.ProcessGroup | Sequence[dist.ProcessGroup | None] | None",
) -> list[dist.ProcessGroup]:
    """Return the spatial groups of size > 1 the statistics are reduced over."""
    if spatial_group is None or isinstance(spatial_group, dist.ProcessGroup):
        spatial_group = [spatial_group]
    return [g for g in spatial_group if g is not None and g.size() > 1]


class DistributedGroupNorm(torch.nn.Module):
    """GroupNorm over a channel- and spatially sharded input.

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
        Whether the affine transformation has a bias. Ignored if ``affine``
        is ``False``.
    dtype : torch.dtype, optional
        Data type of the parameters.
    spatial_group : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
        Process group(s) across which the spatial dimensions are partitioned,
        e.g. the ``spatial_group`` of the preceding convolution. The group
        statistics are reduced over them, and the weight and bias are
        replicated over them. ``None`` means the spatial dimensions are not
        partitioned.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        device: torch.device,
        channel_group: dist.ProcessGroup | None = None,
        eps: float = 1e-5,
        affine: bool = True,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        spatial_group: "dist.ProcessGroup | Sequence[dist.ProcessGroup | None] | None" = None,
    ) -> None:
        """Initialize the layer."""
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
        self._spatial_sync_groups = _sync_groups(spatial_group)
        self.eps = eps
        self.affine = affine
        self.device = device
        if affine:
            self.weight = torch.nn.Parameter(
                torch.ones(self.num_local_channels, device=device, dtype=dtype)
            )
            if bias:
                self.bias = torch.nn.Parameter(
                    torch.zeros(self.num_local_channels, device=device, dtype=dtype)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        for p in self.parameters():
            parallel.mark_replicated(p, self._spatial_sync_groups)

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
        just ``num_channels``) to be divisible by the column group size.

        Parameters
        ----------
        model : torch.nn.GroupNorm
            Layer to convert.
        process_mesh_dict : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        device : torch.device
            Device of the converted module.
        row_kw : str
            Key of the row process group across which the spatial dimensions
            are partitioned, as for :class:`DistributedConv2d`. The group
            statistics are reduced over it, and the weight and bias are
            replicated over it.
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
            spatial_group=process_mesh_dict[row_kw],
        )
        local_channels = instance.num_local_channels
        col_rank = col_group.rank()
        for name in ("weight", "bias"):
            src = getattr(model, name)
            if src is not None:
                getattr(instance, name).data = (
                    src.data[
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
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
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
            Merged bias, or ``None`` if the layer has no bias.
        """
        global_weight, global_bias = (
            None
            if p is None
            else gather_1d_results(p, from_process_mesh, dim=0, pm_key=col_kw)
            for p in (self.weight, self.bias)
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
        # Older ``torch.nn.GroupNorm`` (e.g. 2.9) has no ``bias=`` argument, so a
        # missing bias is dropped after construction instead.
        group_norm = torch.nn.GroupNorm(
            self.num_groups,
            self.num_channels,
            eps=self.eps,
            affine=self.affine,
            device=self.device,
            dtype=weight.dtype if weight is not None else None,
        )
        if weight is not None:
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
        if not self._spatial_sync_groups:
            return F.group_norm(
                x, self.num_local_groups, self.weight, self.bias, self.eps
            )
        weight, bias = parallel.sync_grad(self.weight, self.bias)
        return functional._GroupNorm.apply(  # type: ignore[no-any-return]
            x, self.num_local_groups, weight, bias, self._spatial_sync_groups, self.eps
        )[0]

    def reshard_parameters(
        self,
        from_process_mesh: dict[str, dist.ProcessGroup],
        to_process_mesh: dict[str, dist.ProcessGroup],
        pm_row_key: str = "row",
        pm_col_key: str = "col",
    ) -> None:
        """Reshard the affine parameters from one process mesh onto another.

        The module is rebound to the new mesh: its channels to
        ``to_process_mesh[pm_col_key]`` and its spatial dimensions to
        ``to_process_mesh[pm_row_key]``.

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

        for name in ("weight", "bias"):
            p = getattr(self, name)
            if p is not None:
                global_p = gather_1d_results(
                    p, from_process_mesh, dim=0, pm_key=pm_col_key
                )
                setattr(
                    self,
                    name,
                    torch.nn.Parameter(
                        global_p[
                            col_rank * new_num_channels : (col_rank + 1)
                            * new_num_channels
                        ].contiguous()
                    ),
                )

        self.num_local_channels = new_num_channels
        self.num_local_groups = new_num_local_groups
        self.channel_group = to_channel
        self._spatial_sync_groups = _sync_groups(to_process_mesh[pm_row_key])
        # The re-sliced parameters are fresh Parameters, so the __init__ marks
        # are gone. Re-mark them over the new spatial group.
        for p in self.parameters():
            parallel.mark_replicated(p, self._spatial_sync_groups)

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
        Computed from ``num_channels``, not from the local shards.
        """
        return sum(self.num_channels for p in (self.weight, self.bias) if p is not None)
