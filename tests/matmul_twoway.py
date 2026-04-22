import pytest
import torch
import torch.distributed as dist

import jigsaw.distributed_matmul as dist_mm
from tests import conftest
from tests.conftest import make_model_group

MESHES = {"2x2": (2, 2)}


@pytest.fixture(params=MESHES.keys())
def model_group(request):
    """Return model group in process mesh."""
    return make_model_group(MESHES[request.param])


def test_xwt_twoway(device, model_group):
    """Test for XWT multiplication across two processes."""
    rank = dist.get_rank()
    torch.manual_seed(42 + rank)

    x = torch.randn(1, 4, 16 // 2, device=device)
    w = torch.randn(1, 4, 16 // 2, device=device)

    x_all = conftest.gather_results_twoway(x, group=model_group)
    w_all = conftest.gather_results_twoway(w, group=model_group)

    result = dist_mm.xw_transpose_twoway(x, w, rank, device, model_group=model_group)
    result = conftest.gather_results_twoway(result, group=model_group)

    assert torch.allclose(x_all @ w_all.mT, result, atol=1e-5)


def test_xw_twoway(device, model_group):
    """Test for XW multiplication across two processes."""
    rank = dist.get_rank()
    torch.manual_seed(42 + rank)

    rank = dist.get_rank()
    x = torch.randn(1, 4, 16 // 2, device=device)
    w = torch.randn(1, 16, 8 // 2, device=device)
    x_all = conftest.gather_results_twoway(x, group=model_group)
    w_all = conftest.gather_results_twoway(w, group=model_group)
    result = dist_mm.xw_twoway(x, w, rank, device, model_group=model_group)
    result = conftest.gather_results_twoway(result, group=model_group)

    assert torch.allclose(x_all @ (w_all), result)


def test_xtw_twoway(device, model_group):
    """Test for XTW multiplication across two processes."""
    rank = dist.get_rank()
    torch.manual_seed(42 + rank)

    rank = dist.get_rank()
    x = torch.randn(1, 16, 4 // 2, device=device)
    w = torch.randn(1, 16, 8 // 2, device=device)
    x_all = conftest.gather_results_twoway(x, group=model_group)
    w_all = conftest.gather_results_twoway(w, group=model_group)
    result = dist_mm.x_transpose_w_twoway(x, w, rank, device, model_group=model_group)
    result = conftest.gather_results_twoway(result, group=model_group)

    assert torch.allclose((x_all.mT) @ (w_all), result)


def test_fused_xw_xtw_twoway(device, model_group):
    """Test for XTW multiplication across two processes."""
    rank = dist.get_rank()
    torch.manual_seed(42 + rank)

    rank = dist.get_rank()
    x = torch.randn(1, 4, 16 // 2, device=device)
    w1 = torch.randn(1, 16, 8 // 2, device=device)
    w2 = torch.randn(1, 4, 8 // 2, device=device)
    x_all = conftest.gather_results_twoway(x, group=model_group)
    w1_all = conftest.gather_results_twoway(w1, group=model_group)
    w2_all = conftest.gather_results_twoway(w2, group=model_group)
    result_xw, result_xtw = dist_mm.fused_xw_xtw_twoway(
        x, w1, w2, rank, device, model_group=model_group
    )
    result_xw = conftest.gather_results_twoway(result_xw, group=model_group)
    result_xtw = conftest.gather_results_twoway(result_xtw, group=model_group)

    assert torch.allclose((x_all) @ (w1_all), result_xw)
    assert torch.allclose((x_all.mT) @ (w2_all), result_xtw)
