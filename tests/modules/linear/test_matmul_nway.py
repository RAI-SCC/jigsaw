"""Distributed matmul kernels (XWT / XW / XTW / fused XW+XTW) across 2, 4 and 8 ranks.

One parametrized module in place of the old
``test_matmul_{two,four,eight}way.py`` trio (and ``test_matmul_p2p.py``, which
was a strict subset of the two-way CPU case). Every ``way`` runs the same four
checks against its own mesh, rank count and set of kernels; the kernel name is
``<op>_<way>`` and the per-rank operand shapes come from ``SHAPES`` because each
op shards a different axis.
"""

import pytest
import torch

import jigsaw.modules.linear._distributed_matmul as dist_mm
import jigsaw.modules.linear._distributed_matmul_batchedp2p as dist_mm_batchedp2p
import jigsaw.modules.linear._distributed_matmul_p2p as dist_mm_p2p
from jigsaw.gather import gather_results
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]

# CPU is served by the point-to-point implementations; run every CPU test against
# both the plain and the batched (`batch_isend_irecv`) variant.
CPU_IMPLS = [
    pytest.param(dist_mm_p2p, id="p2p"),
    pytest.param(dist_mm_batchedp2p, id="batched_p2p"),
]

# (kernel-name suffix, mesh fixture), each pinned to the rank count it needs.
WAYS = [
    pytest.param(
        ("twoway", "process_mesh_dict_1x2"),
        marks=pytest.mark.dist(2),
        id="twoway",
    ),
    pytest.param(
        ("fourway", "process_mesh_dict_2x2"),
        marks=pytest.mark.dist(4),
        id="fourway",
    ),
    pytest.param(
        ("eightway", "process_mesh_dict_4x2"),
        marks=pytest.mark.dist(8),
        id="eightway",
    ),
]

# Per-rank (already sharded) operand shapes for each (way, op).
SHAPES = {
    "twoway": {
        "xwt": ((1, 4, 8), (1, 4, 8)),
        "xw": ((1, 4, 8), (1, 16, 4)),
        "xtw": ((1, 16, 2), (1, 16, 4)),
        "fused": ((1, 4, 8), (1, 16, 4), (1, 4, 4)),
    },
    "fourway": {
        "xwt": ((1, 4, 16), (1, 4, 16)),
        "xw": ((1, 4, 16), (1, 16, 8)),
        "xtw": ((2, 32, 64), (2, 32, 64)),
        "fused": ((1, 4, 16), (1, 16, 8), (1, 4, 8)),
    },
    "eightway": {
        "xwt": ((1, 4, 12), (1, 6, 12)),
        "xw": ((1, 4, 12), (1, 6, 12)),
        "xtw": ((2, 32, 64), (2, 32, 64)),
        "fused": ((1, 4, 16), (1, 8, 8), (1, 4, 8)),
    },
}


def _impl(device: torch.device, cpu_impl):
    """Pick the matmul implementation module for the current device."""
    if device.type == "cuda":
        if cpu_impl is not dist_mm_p2p:
            pytest.skip("CUDA uses the collective implementation, not the p2p ones")
        return dist_mm
    return cpu_impl


def _randn(shapes, device: torch.device):
    """Seed off the model-group rank (so data-parallel replicas match) and draw."""
    return [torch.randn(*shape, device=device) for shape in shapes]


class TestMatmulNway:
    """XWT / XW / XTW / fused kernels across 2, 4 and 8 ranks."""

    @pytest.mark.parametrize("cpu_impl", CPU_IMPLS)
    @pytest.mark.parametrize("way", WAYS)
    def test_xwt(
        self, way, cpu_impl, device: torch.device, request: pytest.FixtureRequest
    ) -> None:
        """XWT (``x @ w.mT``) across the model group."""
        suffix, mesh_name = way
        mesh = request.getfixturevalue(mesh_name)
        rank = mesh["model"].rank()
        torch.manual_seed(42 + rank)

        x, w = _randn(SHAPES[suffix]["xwt"], device)
        x_all = gather_results(x, process_mesh_dict=mesh)
        w_all = gather_results(w, process_mesh_dict=mesh)

        result = getattr(_impl(device, cpu_impl), f"xw_transpose_{suffix}")(
            x, w, rank, device, model_group=mesh["model"]
        )
        result = gather_results(result, process_mesh_dict=mesh)

        assert_close("xwt", result, x_all @ w_all.mT, 1e-5, 1e-5, device, mesh["model"])

    @pytest.mark.parametrize("cpu_impl", CPU_IMPLS)
    @pytest.mark.parametrize("way", WAYS)
    def test_xw(
        self, way, cpu_impl, device: torch.device, request: pytest.FixtureRequest
    ) -> None:
        """XW (``x @ w``) across the model group."""
        suffix, mesh_name = way
        mesh = request.getfixturevalue(mesh_name)
        rank = mesh["model"].rank()
        torch.manual_seed(42 + rank)

        x, w = _randn(SHAPES[suffix]["xw"], device)
        x_all = gather_results(x, process_mesh_dict=mesh)
        w_all = gather_results(w, process_mesh_dict=mesh)

        result = getattr(_impl(device, cpu_impl), f"xw_{suffix}")(
            x, w, rank, device, model_group=mesh["model"]
        )
        result = gather_results(result, process_mesh_dict=mesh)

        assert_close("xw", result, x_all @ w_all, 1e-5, 1e-5, device, mesh["model"])

    @pytest.mark.parametrize("cpu_impl", CPU_IMPLS)
    @pytest.mark.parametrize("way", WAYS)
    def test_xtw(
        self, way, cpu_impl, device: torch.device, request: pytest.FixtureRequest
    ) -> None:
        """XTW (``x.mT @ w``) across the model group."""
        suffix, mesh_name = way
        mesh = request.getfixturevalue(mesh_name)
        rank = mesh["model"].rank()
        torch.manual_seed(42 + rank)

        x, w = _randn(SHAPES[suffix]["xtw"], device)
        x_all = gather_results(x, process_mesh_dict=mesh)
        w_all = gather_results(w, process_mesh_dict=mesh)

        result = getattr(_impl(device, cpu_impl), f"x_transpose_w_{suffix}")(
            x, w, rank, device, model_group=mesh["model"]
        )
        result = gather_results(result, process_mesh_dict=mesh)

        assert_close("xtw", result, x_all.mT @ w_all, 1e-5, 1e-5, device, mesh["model"])

    @pytest.mark.parametrize("cpu_impl", CPU_IMPLS)
    @pytest.mark.parametrize("way", WAYS)
    def test_fused_xw_xtw(
        self, way, cpu_impl, device: torch.device, request: pytest.FixtureRequest
    ) -> None:
        """Fused kernel returning both ``x @ w1`` and ``x.mT @ w2``."""
        suffix, mesh_name = way
        mesh = request.getfixturevalue(mesh_name)
        rank = mesh["model"].rank()
        torch.manual_seed(42 + rank)

        x, w1, w2 = _randn(SHAPES[suffix]["fused"], device)
        x_all = gather_results(x, process_mesh_dict=mesh)
        w1_all = gather_results(w1, process_mesh_dict=mesh)
        w2_all = gather_results(w2, process_mesh_dict=mesh)

        result_xw, result_xtw = getattr(
            _impl(device, cpu_impl), f"fused_xw_xtw_{suffix}"
        )(x, w1, w2, rank, device, model_group=mesh["model"])
        result_xw = gather_results(result_xw, process_mesh_dict=mesh)
        result_xtw = gather_results(result_xtw, process_mesh_dict=mesh)

        assert_close(
            "fused xw", result_xw, x_all @ w1_all, 1e-5, 1e-5, device, mesh["model"]
        )
        assert_close(
            "fused xtw",
            result_xtw,
            x_all.mT @ w2_all,
            1e-5,
            1e-5,
            device,
            mesh["model"],
        )
