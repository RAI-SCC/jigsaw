"""Distributed regression losses as ``torch.nn`` modules."""

import torch

from .._common import GroupsArg
from . import functional as F


class DistributedMSELoss(torch.nn.Module):
    """Mean-squared-error loss of a sharded input."""

    def __init__(
        self,
        reduce_groups: GroupsArg = None,
        *,
        reduction: str = "mean",
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
            Every group across which the input is partitioned.
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        """
        super().__init__()
        self.reduce_groups = reduce_groups
        self.reduction = reduction

    def get_parameter_count(self) -> int:
        """Return the number of local parameters."""
        return 0

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the local shard.

        Parameters
        ----------
        input : torch.Tensor
            Predictions.
        target : torch.Tensor
            Targets with the shape of ``input``.

        Returns
        -------
        torch.Tensor
            Local loss for ``"none"``, otherwise the reduced scalar loss.
        """
        return F.distributed_mse_loss(
            input,
            target,
            reduction=self.reduction,
            reduce_groups=self.reduce_groups,
        )


class DistributedL1Loss(torch.nn.Module):
    """Mean-absolute-error loss of a sharded input."""

    def __init__(
        self,
        reduce_groups: GroupsArg = None,
        *,
        reduction: str = "mean",
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
            Every group across which the input is partitioned.
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        """
        super().__init__()
        self.reduce_groups = reduce_groups
        self.reduction = reduction

    def get_parameter_count(self) -> int:
        """Return the number of local parameters."""
        return 0

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the local shard.

        Parameters
        ----------
        input : torch.Tensor
            Predictions.
        target : torch.Tensor
            Targets with the shape of ``input``.

        Returns
        -------
        torch.Tensor
            Local loss for ``"none"``, otherwise the reduced scalar loss.
        """
        return F.distributed_l1_loss(
            input,
            target,
            reduction=self.reduction,
            reduce_groups=self.reduce_groups,
        )


class DistributedSmoothL1Loss(torch.nn.Module):
    """Smooth-L1 loss of a sharded input."""

    def __init__(
        self,
        reduce_groups: GroupsArg = None,
        *,
        beta: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
            Every group across which the input is partitioned.
        beta : float
            Transition point between the quadratic and the linear regime.
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        """
        super().__init__()
        self.reduce_groups = reduce_groups
        self.beta = beta
        self.reduction = reduction

    def get_parameter_count(self) -> int:
        """Return the number of local parameters."""
        return 0

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the local shard.

        Parameters
        ----------
        input : torch.Tensor
            Predictions.
        target : torch.Tensor
            Targets with the shape of ``input``.

        Returns
        -------
        torch.Tensor
            Local loss for ``"none"``, otherwise the reduced scalar loss.
        """
        return F.distributed_smooth_l1_loss(
            input,
            target,
            beta=self.beta,
            reduction=self.reduction,
            reduce_groups=self.reduce_groups,
        )
