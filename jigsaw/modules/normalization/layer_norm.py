"""Distributed LayerNorm."""

from typing import Any, Optional

import torch
import torch.distributed as dist

from jigsaw import parallel
from jigsaw.gather import gather_1d_results

from . import functional


class DistributedLayerNorm(torch.nn.Module):
    """LayerNorm over a channel-sharded last dimension."""

    def __init__(
        self,
        normalized_dim: int,
        local_rank: int,
        device: torch.device,
        row_group: dist.ProcessGroup,
        col_group: dist.ProcessGroup,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the layer.

        Parameters
        ----------
        normalized_dim : int
            Global (unsharded) size of the normalized dimension. Must be
            divisible by ``col_group.size()``.
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
        eps : float
            Term added to the variance for numerical stability.
        elementwise_affine : bool
            Whether the layer has a learnable weight (and bias, see ``bias``),
            as in ``torch.nn.LayerNorm``.
        bias : bool
            Whether the layer has a learnable bias. Ignored if
            ``elementwise_affine`` is ``False``.
        dtype : torch.dtype, optional
            Data type of the parameters. Defaults to the torch default dtype.
        """
        super().__init__()
        assert normalized_dim % col_group.size() == 0, (
            f"normalized_dim={normalized_dim} is not divisible by the column "
            f"group size {col_group.size()}"
        )
        local_dim = normalized_dim // col_group.size()
        self.local_rank = local_rank
        self.device = device
        self.spatial_group = row_group
        self.channel_group = col_group
        self.elementwise_affine = elementwise_affine
        self.normalized_dim = normalized_dim
        self.local_dim = local_dim
        self.eps = eps

        if elementwise_affine:
            # Learnable parameters
            self.weight = torch.nn.Parameter(
                torch.ones(local_dim, device=device, dtype=dtype)
            )
            if bias:
                self.bias = torch.nn.Parameter(
                    torch.zeros(local_dim, device=device, dtype=dtype)
                )
            else:
                self.register_parameter("bias", None)
            self.layernorm = functional._LayerNorm.apply
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.layernorm = functional._LayerNormNoAffine.apply

        for p in self.parameters():
            parallel.mark_replicated(p, self.spatial_group)

    def get_parameter_count(self) -> int:
        """Return the number of local parameters."""
        local = sum(p.numel() for p in (self.weight, self.bias) if p is not None)
        return local // dist.get_world_size(group=self.spatial_group)

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
            multi-axis ``normalized_shape``).
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
            Distributed layer equivalent to ``model``, with the shards of its
            weight and bias.

        Raises
        ------
        NotImplementedError
            If ``model`` normalizes more than one dimension.
        """
        if len(model.normalized_shape) != 1:
            raise NotImplementedError(
                "DistributedLayerNorm only normalizes a single trailing "
                f"dimension; got normalized_shape={model.normalized_shape}"
            )
        elementwise_affine = model.weight is not None

        col_group = process_mesh_dict[col_kw]
        instance = cls(
            normalized_dim=model.normalized_shape[0],
            local_rank=process_mesh_dict[model_kw].rank(),
            device=device,
            row_group=process_mesh_dict[row_kw],
            col_group=col_group,
            eps=model.eps,
            elementwise_affine=elementwise_affine,
            bias=model.bias is not None,
            dtype=model.weight.dtype if model.weight is not None else None,
        )
        local_dim = instance.local_dim
        col_rank = col_group.rank()
        for name in ("weight", "bias"):
            src = getattr(model, name)
            if src is not None:
                getattr(instance, name).data = (
                    src.data[col_rank * local_dim : (col_rank + 1) * local_dim]
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
            Layer with the same ``eps``, ``elementwise_affine`` and ``bias``,
            and the merged weight and bias.
        """
        weight, bias = self.merge_weights(process_mesh_dict, row_kw, col_kw, model_kw)
        layer_norm = torch.nn.LayerNorm(
            self.normalized_dim,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
            bias=bias is not None,
            device=self.device,
            dtype=weight.dtype if weight is not None else None,
        )
        if weight is not None:
            layer_norm.weight.data = weight
        if bias is not None:
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
        to_col = to_process_mesh[pm_col_key]
        assert self.normalized_dim % to_col.size() == 0, (
            "global normalized dim must be divisible by the target channel group size"
        )
        new_local_dim = self.normalized_dim // to_col.size()
        col_rank = to_col.rank()

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
                            col_rank * new_local_dim : (col_rank + 1) * new_local_dim
                        ].contiguous()
                    ),
                )

        self.local_dim = new_local_dim
        self.spatial_group = to_process_mesh[pm_row_key]
        self.channel_group = to_col
        self.local_rank = to_process_mesh.get("model", to_col).rank()
        # The re-sliced parameters are fresh Parameters, so the __init__ marks
        # are gone. Re-mark them over the new spatial group.
        for p in self.parameters():
            parallel.mark_replicated(p, self.spatial_group)

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
