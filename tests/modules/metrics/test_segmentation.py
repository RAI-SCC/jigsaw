"""Parity tests for the distributed segmentation metrics.

Global ``logits [N, C, H, W]`` and ``target [N, H, W]`` are built identically on
every rank; the class dimension ``C`` is sharded across ``col`` and the spatial
dimension ``H`` across ``row``.  Hence ``channel_group == col`` and
``reduce_groups == (row,)``.  The distributed confusion matrix / metrics are
compared against a plain single-process computation on the global tensors.

Boolean and float comparisons run through ``tests.parity`` (``all_ranks_ok`` /
``assert_close``) so a mismatch on a subset of ranks fails the job instead of
hanging the rest in the next collective, matching the linear / attention /
normalization suites.
"""

import pytest
import torch

import jigsaw.modules.metrics as jmetrics
from jigsaw.gather import gather_1d_results
from jigsaw.modules.activation import distributed_argmax
from jigsaw.modules.metrics import _segmentation as seg
from tests.parity import all_ranks_ok, assert_close

pytestmark = [pytest.mark.dist_required]

MESHES_4 = [
    "process_mesh_dict_2x2",
    "process_mesh_dict_1x4",
    "process_mesh_dict_4x1",
    "process_mesh_dict_2x1x2",
    "process_mesh_dict_2x2x1",
]

N, C, H, W = 3, 8, 16, 12


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _shard(t: torch.Tensor, pmd: dict, dims: dict) -> torch.Tensor:
    out = t
    for key, d in dims.items():
        group = pmd[key]
        if group.size() > 1:
            out = out.chunk(group.size(), dim=d)[group.rank()]
    return out.contiguous()


def _gather(t: torch.Tensor, pmd: dict, dims: list) -> torch.Tensor:
    out = t
    for key, d in dims:
        out = gather_1d_results(out, pmd, dim=d, pm_key=key)
    return out


def _global_logits(device: torch.device) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(N, C, H, W, device=device)


def _global_target(device: torch.device, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, C, (N, H, W), generator=g).to(device)


def _ref_confusion(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    pred_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = (
        logits.argmax(dim=1).reshape(-1) if pred_idx is None else pred_idx.reshape(-1)
    )
    tgt = target.reshape(-1)
    keep = torch.ones_like(tgt, dtype=torch.bool)
    if ignore_index is not None:
        keep &= tgt != ignore_index
    pair = tgt[keep].long() * num_classes + pred[keep].long()
    return (
        torch.bincount(pair, minlength=num_classes * num_classes)
        .reshape(num_classes, num_classes)
        .to(torch.int64)
    )


def _ref_metrics(cm: torch.Tensor) -> dict:
    cm = cm.to(torch.float64)
    tp = cm.diagonal()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    support = cm.sum(1)
    total = cm.sum()
    iou = tp / (tp + fp + fn)
    dice = 2 * tp / (2 * tp + fp + fn)
    recall = tp / support
    precision = tp / (tp + fp)
    freq = support / total
    fw_iou = torch.where(support > 0, freq * torch.nan_to_num(iou), 0.0).sum()
    po = tp.sum() / total
    pe = (support * cm.sum(0)).sum() / (total * total)
    return {
        "per_class_iou": iou,
        "mean_iou": torch.nanmean(iou),
        "per_class_dice": dice,
        "mean_dice": torch.nanmean(dice),
        "per_class_recall": recall,
        "mean_recall": torch.nanmean(recall),
        "per_class_precision": precision,
        "mean_precision": torch.nanmean(precision),
        "pixel_accuracy": po,
        "mean_accuracy": torch.nanmean(recall),
        "frequency_weighted_iou": fw_iou,
        "cohen_kappa": (po - pe) / (1 - pe),
    }


def _assert_metrics_match(
    got: dict, ref: dict, device: torch.device, model_group: object
) -> None:
    """Every derived metric agrees with the single-process reference.

    ``allclose`` with ``equal_nan`` (absent classes yield ``nan``) rather than
    ``assert_close``, but still reduced across the model group so one rank's
    disagreement fails the job rather than hanging it.
    """
    for key, ref_val in ref.items():
        ok = torch.allclose(got[key].double(), ref_val, atol=1e-9, equal_nan=True)
        assert all_ranks_ok(ok, device, model_group), key


# --------------------------------------------------------------------------- #
# distributed arg-max over a sharded class dimension
# --------------------------------------------------------------------------- #
class TestDistributedArgmax:
    pytestmark = pytest.mark.dist(4)

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    def test_distributed_argmax(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """Global arg-max over a class dim split across ``col`` matches ``torch.argmax``."""
        pmd = request.getfixturevalue(mesh_name)
        logits = _global_logits(device)
        local = _shard(logits, pmd, {"col": 1, "row": 2})

        got = distributed_argmax(local, dim=1, group=pmd["col"])
        got = _gather(got, pmd, [("row", 1)])
        assert all_ranks_ok(
            torch.equal(got, logits.argmax(dim=1)), device, pmd["model"]
        )


# --------------------------------------------------------------------------- #
# confusion matrix
# --------------------------------------------------------------------------- #
class TestConfusionMatrix:
    pytestmark = pytest.mark.dist(4)

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    def test_confusion_matrix_from_logits(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """Distributed confusion matrix (logits input) matches the single-process one."""
        pmd = request.getfixturevalue(mesh_name)
        logits = _global_logits(device)
        target = _global_target(device)

        cm = seg.distributed_confusion_matrix(
            _shard(logits, pmd, {"col": 1, "row": 2}),
            _shard(target, pmd, {"row": 1}),
            C,
            channel_group=pmd["col"],
            reduce_groups=(pmd["row"],),
        )
        assert all_ranks_ok(
            torch.equal(cm, _ref_confusion(logits, target, C)), device, pmd["model"]
        )
        assert all_ranks_ok(int(cm.sum()) == N * H * W, device, pmd["model"])

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    def test_confusion_matrix_ignore_index(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """``ignore_index`` pixels are dropped from the matrix."""
        pmd = request.getfixturevalue(mesh_name)
        logits = _global_logits(device)
        target = _global_target(device)
        target[:, ::3, :] = 255  # void band

        cm = seg.distributed_confusion_matrix(
            _shard(logits, pmd, {"col": 1, "row": 2}),
            _shard(target, pmd, {"row": 1}),
            C,
            ignore_index=255,
            channel_group=pmd["col"],
            reduce_groups=(pmd["row"],),
        )
        assert all_ranks_ok(
            torch.equal(cm, _ref_confusion(logits, target, C, ignore_index=255)),
            device,
            pmd["model"],
        )
        assert all_ranks_ok(
            int(cm.sum()) == int((target != 255).sum()), device, pmd["model"]
        )

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    def test_confusion_matrix_from_indices(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """Pre-arg-maxed global class indices are accepted directly."""
        pmd = request.getfixturevalue(mesh_name)
        logits = _global_logits(device)
        target = _global_target(device)
        pred = logits.argmax(dim=1)

        cm = seg.distributed_confusion_matrix(
            _shard(pred, pmd, {"row": 1}),
            _shard(target, pmd, {"row": 1}),
            C,
            reduce_groups=(pmd["row"],),
        )
        assert all_ranks_ok(
            torch.equal(cm, _ref_confusion(logits, target, C)), device, pmd["model"]
        )


# --------------------------------------------------------------------------- #
# metrics derived from the confusion matrix
# --------------------------------------------------------------------------- #
class TestSegmentationMetrics:
    pytestmark = pytest.mark.dist(4)

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    def test_segmentation_metrics(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """Every derived metric matches a straight single-process computation."""
        pmd = request.getfixturevalue(mesh_name)
        logits = _global_logits(device)
        target = _global_target(device)

        cm = seg.distributed_confusion_matrix(
            _shard(logits, pmd, {"col": 1, "row": 2}),
            _shard(target, pmd, {"row": 1}),
            C,
            channel_group=pmd["col"],
            reduce_groups=(pmd["row"],),
        )
        got = seg.confusion_to_metrics(cm)
        ref = _ref_metrics(_ref_confusion(logits, target, C))
        _assert_metrics_match(got, ref, device, pmd["model"])

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    def test_segmentation_metrics_exclude_class(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """``exclude`` zeroes class rows/columns before deriving the metrics."""
        pmd = request.getfixturevalue(mesh_name)
        logits = _global_logits(device)
        target = _global_target(device)

        cm = seg.distributed_confusion_matrix(
            _shard(logits, pmd, {"col": 1, "row": 2}),
            _shard(target, pmd, {"row": 1}),
            C,
            channel_group=pmd["col"],
            reduce_groups=(pmd["row"],),
        )
        got = seg.confusion_to_metrics(cm, exclude=[0, 3])

        ref_cm = _ref_confusion(logits, target, C).clone()
        ref_cm[[0, 3], :] = 0
        ref_cm[:, [0, 3]] = 0
        ref = _ref_metrics(ref_cm)
        _assert_metrics_match(got, ref, device, pmd["model"])
        assert all_ranks_ok(
            bool(
                torch.isnan(got["per_class_iou"][0])
                and torch.isnan(got["per_class_iou"][3])
            ),
            device,
            pmd["model"],
        )

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    def test_absent_class_is_nan(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """A class that appears in neither prediction nor target yields ``nan`` per-class IoU."""
        pmd = request.getfixturevalue(mesh_name)
        logits = _global_logits(device)
        target = _global_target(device, seed=7)
        # class 0 never occurs as a target and is never the arg-max
        target[target == 0] = 1
        logits[:, 0] = -1e4

        cm = seg.distributed_confusion_matrix(
            _shard(logits, pmd, {"col": 1, "row": 2}),
            _shard(target, pmd, {"row": 1}),
            C,
            channel_group=pmd["col"],
            reduce_groups=(pmd["row"],),
        )
        got = seg.confusion_to_metrics(cm)
        ref = _ref_metrics(_ref_confusion(logits, target, C))

        assert all_ranks_ok(
            bool(torch.isnan(got["per_class_iou"][0])), device, pmd["model"]
        )
        assert all_ranks_ok(
            bool(not torch.isnan(got["mean_iou"])), device, pmd["model"]
        )
        assert_close(
            "mean_iou",
            got["mean_iou"].double(),
            ref["mean_iou"],
            1e-9,
            1e-5,
            device,
            pmd["model"],
        )


# --------------------------------------------------------------------------- #
# stateful module
# --------------------------------------------------------------------------- #
class TestMetricModule:
    pytestmark = pytest.mark.dist(4)

    @pytest.mark.parametrize("mesh_name", MESHES_4)
    def test_metric_module_accumulation(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """``DistributedSegmentationMetrics`` accumulates to the all-batch reference."""
        pmd = request.getfixturevalue(mesh_name)
        metric = jmetrics.DistributedSegmentationMetrics(
            C, channel_group=pmd["col"], reduce_groups=(pmd["row"],), ignore_index=255
        )

        ref_cm = torch.zeros(C, C, dtype=torch.int64, device=device)
        for step in range(3):
            logits = _global_logits(device) + step
            target = _global_target(device, seed=step)
            target[:, step, :] = 255

            metric.update(
                _shard(logits, pmd, {"col": 1, "row": 2}),
                _shard(target, pmd, {"row": 1}),
            )
            ref_cm += _ref_confusion(logits, target, C, ignore_index=255)

        assert all_ranks_ok(torch.equal(metric.confusion, ref_cm), device, pmd["model"])
        got = metric.compute()
        ref = _ref_metrics(ref_cm)
        _assert_metrics_match(got, ref, device, pmd["model"])

        metric.reset()
        assert all_ranks_ok(int(metric.confusion.sum()) == 0, device, pmd["model"])


# --------------------------------------------------------------------------- #
# eight ranks
# --------------------------------------------------------------------------- #
class TestEightWay:
    pytestmark = pytest.mark.dist(8)

    def test_eightway_metrics(
        self, request: pytest.FixtureRequest, device: torch.device
    ) -> None:
        """Confusion matrix and metrics stay at parity on an eight-rank ``4 x 2`` mesh."""
        pmd = request.getfixturevalue("process_mesh_dict_4x2")
        logits = _global_logits(device)
        target = _global_target(device)

        cm = seg.distributed_confusion_matrix(
            _shard(logits, pmd, {"col": 1, "row": 2}),
            _shard(target, pmd, {"row": 1}),
            C,
            channel_group=pmd["col"],
            reduce_groups=(pmd["row"],),
        )
        assert all_ranks_ok(
            torch.equal(cm, _ref_confusion(logits, target, C)), device, pmd["model"]
        )

        got = seg.confusion_to_metrics(cm)
        ref = _ref_metrics(_ref_confusion(logits, target, C))
        _assert_metrics_match(got, ref, device, pmd["model"])
