"""Distributed LayerNorm."""

from typing import Any, Optional

import torch
import torch.distributed as dist

from jigsaw import parallel
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
        elementwise_affine: bool = True,
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
            Process group the weight and bias are replicated over; their
            gradients are summed over it by :func:`jigsaw.parallel.sync_grad`.
            Unused if ``elementwise_affine`` is ``False``: there are then no
            parameters to reduce gradients for.
        col_group : dist.ProcessGroup
            Process group across which the normalized dimension is partitioned.
        elementwise_affine : bool
            Whether the layer has a learnable weight and bias, as in
            ``torch.nn.LayerNorm``.
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.spatial_group = row_group
        self.channel_group = col_group
        self.elementwise_affine = elementwise_affine
        self.normalized_dim = local_dim
        self.eps = 1e-5

        if elementwise_affine:
            # Learnable parameters
            self.weight = torch.nn.Parameter(torch.ones(local_dim, device=device))
            self.bias = torch.nn.Parameter(torch.zeros(local_dim, device=device))
            self._mark_replicated()
            self.layernorm = functional.DistributedLNFunction.apply
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.layernorm = functional.DistributedNoParamLNFunction.apply

    def _mark_replicated(self) -> None:
        """Mark the weight and bias replicated over the row group."""
        for p in (self.weight, self.bias):
            parallel.mark_replicated(p, self.spatial_group)

    def get_parameter_count(self) -> int:
        """Return the number of local parameters."""
        if not self.elementwise_affine:
            return 0
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
        """Convert a ``torch.nn.LayerNorm`` to its distributed equivalent.

        Parameters
        ----------
        model : torch.nn.LayerNorm
            Layer to convert. Must normalize a single trailing dimension (no
            multi-axis ``normalized_shape``), and either have both a weight
            and a bias (``elementwise_affine=True``, ``bias=True``) or
            neither (``elementwise_affine=False``).
        process_mesh_dict : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        device : torch.device
            Device of the converted module.
        row_kw : str
            Key of the row process group. The converted layer replicates its
            weight and bias across this group, and reduces their gradients
            over it -- correct only if the input is also replicated (not
            sharded) across the same group, as it is for the row/column split
            used by :func:`jigsaw.convert.convert_to_jigsaw`. Unused if
            ``model`` has no weight or bias.
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
            Distributed layer equivalent to ``model``, with its weight and
            bias when it has them (``elementwise_affine=False`` otherwise).

        Raises
        ------
        NotImplementedError
            If ``model`` normalizes more than one dimension, or has exactly
            one of a weight or a bias (``bias=False`` with
            ``elementwise_affine=True``) -- convert that by hand.
        """
        if len(model.normalized_shape) != 1:
            raise NotImplementedError(
                "DistributedLayerNorm only normalizes a single trailing "
                f"dimension; got normalized_shape={model.normalized_shape}"
            )
        if (model.weight is None) != (model.bias is None):
            raise NotImplementedError(
                "DistributedLayerNorm always has both a weight and a bias; "
                "convert a LayerNorm without one of them by hand"
            )
        elementwise_affine = model.weight is not None

        normalized_dim = model.normalized_shape[0]
        col_group = process_mesh_dict[col_kw]
        assert normalized_dim % col_group.size() == 0, (
            "normalized_shape is not divisible by process mesh columns"
        )
        local_dim = normalized_dim // col_group.size()

        instance = cls(
            local_dim=local_dim,
            local_rank=process_mesh_dict[model_kw].rank(),
            device=device,
            row_group=process_mesh_dict[row_kw],
            col_group=col_group,
            elementwise_affine=elementwise_affine,
        )
        instance.eps = model.eps
        if elementwise_affine:
            assert model.weight is not None and model.bias is not None
            col_rank = col_group.rank()
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
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
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
        global_weight : torch.Tensor | None
            Merged weight, or ``None`` if the layer is not affine.
        global_bias : torch.Tensor | None
            Merged bias, or ``None`` if the layer is not affine.
        """
        if not self.elementwise_affine:
            return None, None
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
            Layer with the merged weight and bias, or with
            ``elementwise_affine=False`` if this layer has none.
        """
        weight, bias = self.merge_weights(process_mesh_dict, row_kw, col_kw, model_kw)
        normalized_dim = self.normalized_dim * process_mesh_dict[col_kw].size()
        if not self.elementwise_affine:
            return torch.nn.LayerNorm(
                normalized_dim,
                eps=self.eps,
                elementwise_affine=False,
                device=self.device,
            )
        assert weight is not None and bias is not None
        layer_norm = torch.nn.LayerNorm(
            normalized_dim, eps=self.eps, device=self.device, dtype=weight.dtype
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

        if self.elementwise_affine:
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
        if self.elementwise_affine:
            # Fresh Parameters: the __init__ mark is gone; mark on the new rows.
            self._mark_replicated()

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
        if self.elementwise_affine:
            weight, bias = parallel.sync_grad(self.weight, self.bias)
            return self.layernorm(  # type: ignore[no-any-return]
                x, weight, bias, self.channel_group, self.eps
            )[0]
        return self.layernorm(x, self.channel_group, self.eps)[0]  # type: ignore[no-any-return]
