"""Distributed semantic-segmentation metrics built on a confusion matrix.

The confusion matrix ``cm[t, p]`` counts pixels with true class ``t`` and
predicted class ``p``.  It is the single quantity that needs communication:

* if the class dimension of the prediction logits is sharded across
  ``channel_group`` the arg-max is distributed (see
  :func:`jigsaw.modules.loss.common.distributed_argmax`);
* the ``C x C`` count matrix is summed across every ``reduce_groups`` group, i.e.
  every group over which the batch / spatial dimensions are partitioned.  It is
  *not* reduced over ``channel_group`` -- after the distributed arg-max every
  channel rank holds the same prediction map for its spatial shard.

Every scalar metric (mIoU, Dice, pixel accuracy, ...) is then a pure function of
the reduced matrix and needs no further communication.
"""

from typing import Optional, Sequence

import torch
import torch.distributed as dist

from .common import GroupsArg, _active, _normalize_groups, distributed_argmax


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
    """Accumulate a distributed ``[num_classes, num_classes]`` confusion matrix.

    Parameters
    ----------
    pred : torch.Tensor
        Either prediction logits/scores with the (possibly sharded) class
        dimension at ``dim`` -- shape ``target.shape`` with ``dim`` inserted --
        or already-arg-maxed global class indices with the same shape as
        ``target``.
    target : torch.Tensor
        Ground-truth class indices, shape ``[N, *spatial]``.
    num_classes : int
        Global number of classes ``C``.
    ignore_index : Optional[int]
        Target value to exclude from the matrix entirely.
    dim : int
        Class dimension of ``pred`` when it holds logits.
    channel_group : Optional[dist.ProcessGroup]
        Group sharding the class dimension (only used for the distributed
        arg-max when ``pred`` holds logits).
    reduce_groups : GroupsArg
        Every group over which the batch / spatial dimensions are partitioned.
    pred_is_logits : Optional[bool]
        Force the interpretation of ``pred``; inferred from its rank otherwise.

    Returns
    -------
    torch.Tensor
        ``int64`` confusion matrix of shape ``[num_classes, num_classes]``,
        identical on every rank of ``reduce_groups``.  Rows index the true
        class, columns the predicted class.
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
    """``numerator / denominator`` for integer-count denominators; ``nan`` when it is ``0``.

    The denominator is a sum of non-negative counts, so it is either ``0`` (the
    ``nan`` case) or ``>= 1`` -- clamping to ``1`` only rewrites the ignored branch.
    """
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1),
        torch.full_like(numerator, float("nan")),
    )


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """``numerator / denominator`` with ``nan`` where ``denominator == 0`` (any magnitude)."""
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

    Absent classes (a zero row *and* column) yield ``nan`` in their per-class
    entry and are dropped from the ``mean_*`` aggregates (``nanmean``).

    Parameters
    ----------
    confusion : torch.Tensor
        ``[C, C]`` matrix, rows = true class, columns = predicted class.
    exclude : Optional[Sequence[int]]
        Class indices to drop before computing anything -- their rows and
        columns are zeroed, so the excluded pixels count towards neither the
        per-class metrics nor pixel accuracy.

    Returns
    -------
    dict[str, torch.Tensor]
        Keys: ``confusion``, ``support``, ``per_class_iou``, ``mean_iou``,
        ``per_class_dice``, ``mean_dice``, ``per_class_precision``,
        ``mean_precision``, ``per_class_recall``, ``mean_recall``,
        ``pixel_accuracy``, ``mean_accuracy``, ``frequency_weighted_iou``,
        ``cohen_kappa``.  Every value is a tensor (0-dim for the scalars).
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

    ``update`` (also the ``forward``) is collective -- it must be called in
    lock-step on every rank of ``reduce_groups`` -- and folds one batch into the
    running matrix.  ``compute`` and ``reset`` are purely local.
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
        """Initialize the accumulator.

        Parameters
        ----------
        num_classes : int
            Global number of classes.
        channel_group : Optional[dist.ProcessGroup]
            Group sharding the class dimension of the prediction logits.
        reduce_groups : GroupsArg
            Every group over which the batch / spatial dimensions are partitioned.
        ignore_index : Optional[int]
            Target value excluded from the matrix.
        dim : int
            Class dimension of the prediction logits.
        exclude : Optional[Sequence[int]]
            Class indices dropped when :meth:`compute` derives the metrics.
        """
        super().__init__()
        self.num_classes = num_classes
        self.channel_group = channel_group
        self.reduce_groups = reduce_groups
        self.ignore_index = ignore_index
        self.dim = dim
        self.exclude = exclude
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def get_parameter_count(self) -> int:
        """Return ``0``; the metric holds no trainable parameters."""
        return 0

    def reset(self) -> None:
        """Zero the running confusion matrix."""
        self.confusion = torch.zeros_like(self.confusion)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Fold one batch into the running matrix and return that batch's matrix."""
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
        """Return the segmentation metrics for the matrix accumulated so far."""
        return confusion_to_metrics(self.confusion, exclude=self.exclude)
