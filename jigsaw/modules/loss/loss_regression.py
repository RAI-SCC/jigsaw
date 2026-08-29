"""Distributed regression losses wrapped as ``torch.nn`` modules.

These losses are elementwise, so ``reduce_groups`` must list *every* group across
which the input is partitioned (spatial and channel alike).  The ``"mean"`` /
``"sum"`` reduction is all-reduced over those groups so the returned scalar
matches the value the non-distributed loss would produce on the gathered tensor.
With ``reduce_groups=()`` every module reduces to a plain call of the
corresponding ``torch.nn.functional`` loss on the local tensor.
"""

import torch

from . import functional_regression as fr
from .common import GroupsArg


class DistributedMSELoss(torch.nn.Module):
    """Mean-squared-error / L2 loss over a tensor sharded arbitrarily on a mesh."""

    def __init__(
        self,
        reduce_groups: GroupsArg = None,
        *,
        reduction: str = "mean",
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        reduce_groups : GroupsArg
            Every group across which the input is partitioned.
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        """
        super().__init__()
        self.reduce_groups = reduce_groups
        self.reduction = reduction

    def get_parameter_count(self) -> int:
        """Return ``0``; the loss holds no trainable parameters."""
        return 0

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the distributed MSE loss for this rank's shard."""
        return fr.distributed_mse_loss(
            input,
            target,
            reduction=self.reduction,
            reduce_groups=self.reduce_groups,
        )


class DistributedL1Loss(torch.nn.Module):
    """Mean-absolute-error / L1 loss over a tensor sharded arbitrarily on a mesh."""

    def __init__(
        self,
        reduce_groups: GroupsArg = None,
        *,
        reduction: str = "mean",
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        reduce_groups : GroupsArg
            Every group across which the input is partitioned.
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        """
        super().__init__()
        self.reduce_groups = reduce_groups
        self.reduction = reduction

    def get_parameter_count(self) -> int:
        """Return ``0``; the loss holds no trainable parameters."""
        return 0

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the distributed L1 loss for this rank's shard."""
        return fr.distributed_l1_loss(
            input,
            target,
            reduction=self.reduction,
            reduce_groups=self.reduce_groups,
        )


class DistributedSmoothL1Loss(torch.nn.Module):
    """Smooth-L1 (Huber) loss over a tensor sharded arbitrarily on a mesh."""

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
        reduce_groups : GroupsArg
            Every group across which the input is partitioned.
        beta : float
            Transition point between the quadratic and linear regimes.
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        """
        super().__init__()
        self.reduce_groups = reduce_groups
        self.beta = beta
        self.reduction = reduction

    def get_parameter_count(self) -> int:
        """Return ``0``; the loss holds no trainable parameters."""
        return 0

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the distributed smooth-L1 loss for this rank's shard."""
        return fr.distributed_smooth_l1_loss(
            input,
            target,
            beta=self.beta,
            reduction=self.reduction,
            reduce_groups=self.reduce_groups,
        )
