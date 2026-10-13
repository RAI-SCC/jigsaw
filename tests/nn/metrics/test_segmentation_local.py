"""Single-rank checks of the segmentation metrics (no sharding, runs on any rank count).

``test_segmentation.py`` compares sharded inputs against a serial reference and so is
``dist_required``. Here the prediction is whole and no group is active, which leaves the
serial code paths: an explicit ``pred_is_logits``, a ``ProcessGroup`` passed where a
sequence of groups is expected, and the trivial ``nn.Module`` surface of the accumulator.
"""

import torch
import torch.distributed as dist

from jigsaw.nn.metrics import _segmentation as seg

NUM_CLASSES = 4


def _case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    logits = torch.randn(2, NUM_CLASSES, 5, 3)
    target = torch.randint(0, NUM_CLASSES, (2, 5, 3))
    expected = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    for t, p in zip(target.flatten(), logits.argmax(1).flatten()):
        expected[t, p] += 1
    return logits, target, expected


class TestConfusionMatrixSerial:
    def test_logits_are_inferred_from_the_extra_dimension(self):
        logits, target, expected = _case()
        cm = seg.distributed_confusion_matrix(logits, target, NUM_CLASSES)
        assert torch.equal(cm, expected)

    def test_explicit_pred_is_logits_true(self):
        logits, target, expected = _case()
        cm = seg.distributed_confusion_matrix(
            logits, target, NUM_CLASSES, pred_is_logits=True
        )
        assert torch.equal(cm, expected)

    def test_explicit_pred_is_logits_false_takes_indices_as_they_are(self):
        logits, target, expected = _case()
        cm = seg.distributed_confusion_matrix(
            logits.argmax(1), target, NUM_CLASSES, pred_is_logits=False
        )
        assert torch.equal(cm, expected)

    def test_single_group_is_accepted_where_a_sequence_is_expected(self, device):
        logits, target, expected = (t.to(device) for t in _case())
        world = dist.group.WORLD
        cm = seg.distributed_confusion_matrix(
            logits, target, NUM_CLASSES, reduce_groups=world
        )
        # A world of one rank is inactive, so the local matrix is already global.
        # With more ranks every rank holds the same data and the matrix is summed.
        assert torch.equal(cm, expected * dist.get_world_size())


class TestSegmentationMetricModule:
    def test_has_no_parameters(self):
        assert (
            seg.DistributedSegmentationMetrics(NUM_CLASSES).get_parameter_count() == 0
        )
