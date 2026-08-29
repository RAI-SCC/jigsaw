"""Distributed classification losses wrapped as ``torch.nn`` modules.

Each module mirrors its ``torch.nn`` counterpart but is aware of a process mesh:

* ``channel_group`` shards the class dimension (``dim``, ``1`` by convention).
  It may be ``None`` / size ``1`` when the class dimension is replicated.
* ``reduce_groups`` lists every group across which the *remaining* (batch and
  spatial) dimensions are partitioned.  The ``"mean"`` / ``"sum"`` reduction is
  all-reduced over those groups so the returned scalar matches the value the
  non-distributed loss would produce on the gathered tensor.

For :class:`DistributedBCELoss` / :class:`DistributedBCEWithLogitsLoss` there is
no distinguished class dimension, so ``reduce_groups`` must list *every* group
the input is sharded over.

With ``channel_group=None`` and ``reduce_groups=()`` every module reduces to a
plain call of the corresponding ``torch.nn.functional`` loss on the local
tensor.
"""

from typing import Optional, Sequence

import torch
import torch.distributed as dist

from . import functional

_Groups = Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]


class DistributedNLLLoss(torch.nn.Module):
    """Negative log-likelihood loss over a class dimension sharded on a mesh."""

    def __init__(
        self,
        channel_group: Optional[dist.ProcessGroup] = None,
        reduce_groups: _Groups = None,
        *,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        dim: int = 1,
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        channel_group : Optional[dist.ProcessGroup]
            Group across which the class dimension ``dim`` is partitioned.
        reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
            Groups across which the batch/spatial dimensions are partitioned.
        weight : Optional[torch.Tensor]
            Per-class weight of length ``C_global`` (replicated on every rank).
        ignore_index : int
            Target value whose elements are skipped.
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        dim : int
            Class dimension of the input.
        """
        super().__init__()
        self.channel_group = channel_group
        self.reduce_groups = reduce_groups
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def get_parameter_count(self) -> int:
        """Return ``0``; the loss holds no trainable parameters."""
        return 0

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the distributed NLL loss for this rank's shard."""
        return functional.distributed_nll_loss(
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
    """Cross-entropy loss over a class dimension sharded on a mesh."""

    def __init__(
        self,
        channel_group: Optional[dist.ProcessGroup] = None,
        reduce_groups: _Groups = None,
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
        channel_group : Optional[dist.ProcessGroup]
            Group across which the class dimension ``dim`` is partitioned.
        reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
            Groups across which the batch/spatial dimensions are partitioned.
        weight : Optional[torch.Tensor]
            Per-class weight of length ``C_global`` (replicated on every rank).
        ignore_index : int
            Target value whose elements are skipped (class-index targets only).
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        label_smoothing : float
            Label smoothing in ``[0, 1)``; requires ``weight is None``.
        dim : int
            Class dimension of the input.
        """
        super().__init__()
        self.channel_group = channel_group
        self.reduce_groups = reduce_groups
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.dim = dim

    def get_parameter_count(self) -> int:
        """Return ``0``; the loss holds no trainable parameters."""
        return 0

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the distributed cross-entropy loss for this rank's shard."""
        return functional.distributed_cross_entropy(
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
    """Binary cross-entropy over a tensor sharded arbitrarily on a mesh."""

    def __init__(
        self,
        reduce_groups: _Groups = None,
        *,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
            Every group across which the input is partitioned.
        weight : Optional[torch.Tensor]
            Elementwise weight broadcastable to this rank's input shard.
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        """
        super().__init__()
        self.reduce_groups = reduce_groups
        self.weight = weight
        self.reduction = reduction

    def get_parameter_count(self) -> int:
        """Return ``0``; the loss holds no trainable parameters."""
        return 0

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the distributed BCE loss for this rank's shard."""
        return functional.distributed_binary_cross_entropy(
            probs,
            target,
            weight=self.weight,
            reduction=self.reduction,
            reduce_groups=self.reduce_groups,
        )


class DistributedBCEWithLogitsLoss(torch.nn.Module):
    """Numerically stable BCE-with-logits over a mesh-sharded tensor."""

    def __init__(
        self,
        reduce_groups: _Groups = None,
        *,
        weight: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
            Every group across which the input is partitioned.
        weight : Optional[torch.Tensor]
            Elementwise weight broadcastable to this rank's input shard.
        pos_weight : Optional[torch.Tensor]
            Weight of positive examples, broadcastable to the class dimension.
        reduction : str
            ``"none"``, ``"sum"`` or ``"mean"``.
        """
        super().__init__()
        self.reduce_groups = reduce_groups
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

    def get_parameter_count(self) -> int:
        """Return ``0``; the loss holds no trainable parameters."""
        return 0

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the distributed BCE-with-logits loss for this rank's shard."""
        return functional.distributed_bce_with_logits(
            logits,
            target,
            weight=self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
            reduce_groups=self.reduce_groups,
        )


class DistributedMSELoss(torch.nn.Module):
    """Mean-squared-error / L2 loss over a tensor sharded arbitrarily on a mesh."""

    def __init__(
        self,
        reduce_groups: _Groups = None,
        *,
        reduction: str = "mean",
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
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
        return functional.distributed_mse_loss(
            input,
            target,
            reduction=self.reduction,
            reduce_groups=self.reduce_groups,
        )


class DistributedL1Loss(torch.nn.Module):
    """Mean-absolute-error / L1 loss over a tensor sharded arbitrarily on a mesh."""

    def __init__(
        self,
        reduce_groups: _Groups = None,
        *,
        reduction: str = "mean",
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
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
        return functional.distributed_l1_loss(
            input,
            target,
            reduction=self.reduction,
            reduce_groups=self.reduce_groups,
        )


class DistributedSmoothL1Loss(torch.nn.Module):
    """Smooth-L1 (Huber) loss over a tensor sharded arbitrarily on a mesh."""

    def __init__(
        self,
        reduce_groups: _Groups = None,
        *,
        beta: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        reduce_groups : Optional[Sequence[Optional[dist.ProcessGroup]] | dist.ProcessGroup]
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
        return functional.distributed_smooth_l1_loss(
            input,
            target,
            beta=self.beta,
            reduction=self.reduction,
            reduce_groups=self.reduce_groups,
        )
