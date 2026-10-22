"""Distributed semantic-segmentation metrics based on a confusion matrix."""

from typing import Optional, Sequence

import torch
import torch.distributed as dist

from ..activation._functional import distributed_argmax
from ..loss._common import GroupsArg, _active, _normalize_groups


@torch.no_grad()
def distributed_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    ignore_index: Optional[int] = None,
    dim: int = 1,
    channel_group: Optional[dist.ProcessGroup] = None,
    reduce_groups: GroupsArg = None,
    pred_is_logits: Optional[bool] = None,
) -> torch.Tensor:
    """Compute the confusion matrix of a sharded prediction.

    Rows of the matrix index the true class and columns the predicted class.

    Parameters
    ----------
    pred : torch.Tensor
        Logits with the class dimension ``dim`` holding the local shard, or global
        class indices with the shape of ``target``.
    target : torch.Tensor
        Class indices.
    num_classes : int
        Global number of classes.
    ignore_index : int, optional
        Target value to exclude from the matrix.
    dim : int
        Class dimension of ``pred`` if it holds logits.
    channel_group : dist.ProcessGroup, optional
        Group across which the class dimension is partitioned.
    reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
        Every group across which the batch and spatial dimensions are partitioned.
    pred_is_logits : bool, optional
        Whether ``pred`` holds logits. Inferred from the number of dimensions of
        ``pred`` if ``None``.

    Returns
    -------
    torch.Tensor
        ``int64`` matrix of shape ``(num_classes, num_classes)``, identical on every
        rank of ``reduce_groups``.
    """
    groups = _normalize_groups(reduce_groups)

    if pred_is_logits is None:
        pred_is_logits = pred.dim() == target.dim() + 1

    if pred_is_logits:
        dim = dim % pred.dim()
        if _active(channel_group):
            assert pred.shape[dim] * channel_group.size() == num_classes, (  # type: ignore[union-attr]
                "num_classes must equal the global size of the sharded class dim"
            )
        pred_idx = distributed_argmax(pred, dim, channel_group)
    else:
        pred_idx = pred

    pred_idx = pred_idx.reshape(-1).long()
    target_idx = target.reshape(-1).long()

    keep = torch.ones_like(target_idx, dtype=torch.bool)
    if ignore_index is not None:
        keep &= target_idx != ignore_index
    keep &= (target_idx >= 0) & (target_idx < num_classes)
    keep &= (pred_idx >= 0) & (pred_idx < num_classes)

    pair = target_idx[keep] * num_classes + pred_idx[keep]
    cm = torch.bincount(pair, minlength=num_classes * num_classes)
    cm = cm.reshape(num_classes, num_classes).to(torch.int64)

    for group in groups:
        if _active(group):
            dist.all_reduce(cm, op=dist.ReduceOp.SUM, group=group)
    return cm


def _count_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Divide by a non-negative count, giving ``nan`` where it is zero."""
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1),
        torch.full_like(numerator, float("nan")),
    )


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Divide, giving ``nan`` where the denominator is zero."""
    return torch.where(
        denominator != 0,
        numerator / denominator.masked_fill(denominator == 0, 1.0),
        torch.full_like(numerator, float("nan")),
    )


def confusion_to_metrics(
    confusion: torch.Tensor,
    *,
    exclude: Optional[Sequence[int]] = None,
) -> dict[str, torch.Tensor]:
    """Derive segmentation metrics from a confusion matrix.

    Classes absent from the matrix yield ``nan`` and are excluded from the ``mean_*``
    entries.

    Parameters
    ----------
    confusion : torch.Tensor
        Matrix of shape ``(C, C)`` with rows indexing the true class and columns the
        predicted class.
    exclude : Sequence[int], optional
        Classes whose rows and columns are zeroed before the metrics are computed.

    Returns
    -------
    dict[str, torch.Tensor]
        Metrics keyed by ``confusion``, ``support``, ``per_class_iou``, ``mean_iou``,
        ``per_class_dice``, ``mean_dice``, ``per_class_precision``,
        ``mean_precision``, ``per_class_recall``, ``mean_recall``,
        ``pixel_accuracy``, ``mean_accuracy``, ``frequency_weighted_iou`` and
        ``cohen_kappa``.
    """
    cm = confusion.to(torch.float64)
    if exclude:
        cm = cm.clone()
        idx = torch.as_tensor(list(exclude), device=cm.device, dtype=torch.long)
        cm[idx, :] = 0.0
        cm[:, idx] = 0.0

    true_per_class = cm.sum(dim=1)
    pred_per_class = cm.sum(dim=0)
    tp = cm.diagonal()
    fp = pred_per_class - tp
    fn = true_per_class - tp
    total = cm.sum()

    per_class_iou = _count_ratio(tp, tp + fp + fn)
    per_class_dice = _count_ratio(2.0 * tp, 2.0 * tp + fp + fn)
    per_class_precision = _count_ratio(tp, tp + fp)
    per_class_recall = _count_ratio(tp, true_per_class)

    freq = _count_ratio(true_per_class, total.expand_as(true_per_class))
    fw_iou = torch.where(
        torch.isnan(per_class_iou), torch.zeros_like(freq), freq * per_class_iou
    ).sum()

    p_observed = _count_ratio(tp.sum(), total)
    p_expected = _count_ratio((true_per_class * pred_per_class).sum(), total * total)
    cohen_kappa = _ratio(p_observed - p_expected, 1.0 - p_expected)

    return {
        "confusion": cm.to(torch.int64),
        "support": true_per_class.to(torch.int64),
        "per_class_iou": per_class_iou,
        "mean_iou": torch.nanmean(per_class_iou),
        "per_class_dice": per_class_dice,
        "mean_dice": torch.nanmean(per_class_dice),
        "per_class_precision": per_class_precision,
        "mean_precision": torch.nanmean(per_class_precision),
        "per_class_recall": per_class_recall,
        "mean_recall": torch.nanmean(per_class_recall),
        "pixel_accuracy": p_observed,
        "mean_accuracy": torch.nanmean(per_class_recall),
        "frequency_weighted_iou": fw_iou,
        "cohen_kappa": cohen_kappa,
    }


class DistributedSegmentationMetrics(torch.nn.Module):
    """Confusion-matrix accumulator for distributed semantic segmentation.

    Parameters
    ----------
    num_classes : int
        Global number of classes.
    channel_group : dist.ProcessGroup, optional
        Group across which the class dimension of the predictions is partitioned.
    reduce_groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None], optional
        Every group across which the batch and spatial dimensions are partitioned.
    ignore_index : int, optional
        Target value to exclude from the matrix.
    dim : int
        Class dimension of the predictions.
    exclude : Sequence[int], optional
        Classes whose rows and columns are zeroed by :meth:`compute`.
    """

    def __init__(
        self,
        num_classes: int,
        channel_group: Optional[dist.ProcessGroup] = None,
        reduce_groups: GroupsArg = None,
        *,
        ignore_index: Optional[int] = None,
        dim: int = 1,
        exclude: Optional[Sequence[int]] = None,
    ) -> None:
        """Initialize the accumulator."""
        super().__init__()
        self.num_classes = num_classes
        self.channel_group = channel_group
        self.reduce_groups = reduce_groups
        self.ignore_index = ignore_index
        self.dim = dim
        self.exclude = exclude
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def get_parameter_count(self) -> int:
        """Return the global number of parameters of this layer.

        Returns
        -------
        int
            Global (unsharded) count of this layer's parameters, including plain
            ``torch.nn`` children; children with ``get_parameter_count`` report
            their own. No communication.
        """
        return 0

    def reset(self) -> None:
        """Zero the running confusion matrix."""
        self.confusion = torch.zeros_like(self.confusion)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Add a batch to the running confusion matrix.

        This is a collective call on the ranks of ``reduce_groups``.

        Parameters
        ----------
        pred : torch.Tensor
            Logits with the class dimension holding the local shard, or global class
            indices.
        target : torch.Tensor
            Class indices.

        Returns
        -------
        torch.Tensor
            Confusion matrix of the batch.
        """
        cm = distributed_confusion_matrix(
            pred,
            target,
            self.num_classes,
            ignore_index=self.ignore_index,
            dim=self.dim,
            channel_group=self.channel_group,
            reduce_groups=self.reduce_groups,
        )
        self.confusion = self.confusion.to(cm.device) + cm
        return cm

    forward = update

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute the metrics of the accumulated confusion matrix.

        Returns
        -------
        dict[str, torch.Tensor]
            Metrics as returned by :func:`confusion_to_metrics`.
        """
        return confusion_to_metrics(self.confusion, exclude=self.exclude)
