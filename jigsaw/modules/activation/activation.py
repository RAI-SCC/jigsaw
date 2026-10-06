"""Distributed activation layers."""

from typing import Any

import torch
import torch.distributed as dist

from . import functional


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

    @classmethod
    def convert_to_jigsaw(
        cls,
        model: torch.nn.Softmax,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
        **kwargs: Any,
    ) -> "DistributedSoftMax":
        """Convert a ``torch.nn.Softmax`` to a :class:`DistributedSoftMax`.

        Parameters
        ----------
        model : torch.nn.Softmax
            Layer to convert. Unused beyond ``dim``, since the operation has
            no parameters.
        process_mesh_dict : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        device : torch.device
            Device of the converted module. Unused.
        row_kw : str
            Key of the row process group. Unused.
        col_kw : str
            Key of the column process group across which ``model.dim`` is
            partitioned -- correct only when the input is actually sharded
            along that dimension across ``col_kw``, as it is for the
            row/column split used by :func:`jigsaw.convert.convert_to_jigsaw`.
        model_kw : str
            Key of the model process group. Unused.
        **kwargs : Any
            Unused. Accepts the options of the other converters, which
            :func:`jigsaw.convert.convert_to_jigsaw` passes to every layer.

        Returns
        -------
        DistributedSoftMax
            Distributed layer over ``process_mesh_dict[col_kw]``.

        Raises
        ------
        NotImplementedError
            If ``model.dim`` is ``None``. Plain ``nn.Softmax`` then infers the
            dimension from the input shape at call time (with a deprecation
            warning); replicating that heuristic here without seeing the
            input would risk silently normalizing over the wrong axis.
        """
        if model.dim is None:
            raise NotImplementedError(
                "DistributedSoftMax requires an explicit dim; convert a "
                "Softmax with dim=None by hand"
            )
        return cls(group=process_mesh_dict[col_kw], dim=model.dim)

    def convert_to_torch(self) -> torch.nn.Softmax:
        """Convert the layer to a ``torch.nn.Softmax``.

        Returns
        -------
        torch.nn.Softmax
            Layer with the same ``dim``.
        """
        return torch.nn.Softmax(dim=self.dim)

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
