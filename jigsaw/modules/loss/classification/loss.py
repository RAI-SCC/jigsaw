"""Distributed classification losses as ``torch.nn`` modules."""

from typing import Optional

import torch
import torch.distributed as dist

from ..common import GroupsArg
from . import functional as fc


class DistributedNLLLoss(torch.nn.Module):
    """Negative log-likelihood loss of a class-sharded input."""

    def __init__(
        self,
        channel_group: Optional[dist.ProcessGroup] = None,
        reduce_groups: GroupsArg = None,
        *,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        dim: int = 1,
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        channel_group : dist.ProcessGroup, optional
            Group across which the class dimension is partitioned.
        reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
            Groups across which the batch and spatial dimensions are partitioned.
        weight : torch.Tensor, optional
            Per-class weight of the global class dimension.
        ignore_index : int
            Target value that contributes neither loss nor normalization.
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        dim : int
            Class dimension of the input.
        """
        super().__init__()
        self.channel_group = channel_group
        self.reduce_groups = reduce_groups
        self.register_buffer("weight", weight)
        self.weight: Optional[torch.Tensor]
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def get_parameter_count(self) -> int:
        """Return the number of local parameters."""
        return 0

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the local shard.

        Parameters
        ----------
        log_probs : torch.Tensor
            Log-probabilities with the class dimension holding the local shard.
        target : torch.Tensor
            Class indices.

        Returns
        -------
        torch.Tensor
            Local loss for ``"none"``, otherwise the reduced scalar loss.
        """
        return fc.distributed_nll_loss(
            log_probs,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
            channel_group=self.channel_group,
            reduce_groups=self.reduce_groups,
        )


class DistributedCrossEntropyLoss(torch.nn.Module):
    """Cross-entropy loss of a class-sharded input."""

    def __init__(
        self,
        channel_group: Optional[dist.ProcessGroup] = None,
        reduce_groups: GroupsArg = None,
        *,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        dim: int = 1,
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        channel_group : dist.ProcessGroup, optional
            Group across which the class dimension is partitioned.
        reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
            Groups across which the batch and spatial dimensions are partitioned.
        weight : torch.Tensor, optional
            Per-class weight of the global class dimension.
        ignore_index : int
            Target value that contributes neither loss nor normalization. Only applies
            to class-index targets.
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        label_smoothing : float
            Label smoothing in ``[0, 1)``.
        dim : int
            Class dimension of the input.
        """
        super().__init__()
        self.channel_group = channel_group
        self.reduce_groups = reduce_groups
        self.register_buffer("weight", weight)
        self.weight: Optional[torch.Tensor]
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.dim = dim

    def get_parameter_count(self) -> int:
        """Return the number of local parameters."""
        return 0

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the local shard.

        Parameters
        ----------
        logits : torch.Tensor
            Unnormalized scores with the class dimension holding the local shard.
        target : torch.Tensor
            Class indices or class probabilities.

        Returns
        -------
        torch.Tensor
            Local loss for ``"none"``, otherwise the reduced scalar loss.
        """
        return fc.distributed_cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            dim=self.dim,
            channel_group=self.channel_group,
            reduce_groups=self.reduce_groups,
        )


class DistributedBCELoss(torch.nn.Module):
    """Binary cross-entropy loss of a sharded input."""

    def __init__(
        self,
        reduce_groups: GroupsArg = None,
        *,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
            Every group across which the input is partitioned.
        weight : torch.Tensor, optional
            Elementwise weight broadcastable to the input.
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        """
        super().__init__()
        self.reduce_groups = reduce_groups
        self.register_buffer("weight", weight)
        self.weight: Optional[torch.Tensor]
        self.reduction = reduction

    def get_parameter_count(self) -> int:
        """Return the number of local parameters."""
        return 0

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the local shard.

        Parameters
        ----------
        probs : torch.Tensor
            Predicted probabilities in ``[0, 1]``.
        target : torch.Tensor
            Targets in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Local loss for ``"none"``, otherwise the reduced scalar loss.
        """
        return fc.distributed_binary_cross_entropy(
            probs,
            target,
            weight=self.weight,
            reduction=self.reduction,
            reduce_groups=self.reduce_groups,
        )


class DistributedBCEWithLogitsLoss(torch.nn.Module):
    """Binary cross-entropy with logits of a sharded input."""

    def __init__(
        self,
        reduce_groups: GroupsArg = None,
        *,
        weight: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
            Every group across which the input is partitioned.
        weight : torch.Tensor, optional
            Elementwise weight broadcastable to the input.
        pos_weight : torch.Tensor, optional
            Weight of the positive examples, broadcastable to the class dimension.
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        """
        super().__init__()
        self.reduce_groups = reduce_groups
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)
        self.weight: Optional[torch.Tensor]
        self.pos_weight: Optional[torch.Tensor]
        self.reduction = reduction

    def get_parameter_count(self) -> int:
        """Return the number of local parameters."""
        return 0

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the local shard.

        Parameters
        ----------
        logits : torch.Tensor
            Unnormalized scores.
        target : torch.Tensor
            Targets in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Local loss for ``"none"``, otherwise the reduced scalar loss.
        """
        return fc.distributed_bce_with_logits(
            logits,
            target,
            weight=self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
            reduce_groups=self.reduce_groups,
        )
