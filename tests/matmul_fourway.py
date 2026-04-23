import pytest
import torch
import torch.distributed as dist

import jigsaw.distributed_matmul as dist_mm
from tests import conftest


@pytest.fixture()
def process_mesh():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 2], mesh_dim_names=["row", "col"]
    )


def test_xwt_fourway(device, process_mesh):
    """Test for XWT multiplication across four processes."""
    rank = dist.get_rank()
    torch.manual_seed(42 + rank)

    x = torch.randn(1, 4, 16, device=device)
    w = torch.randn(1, 4, 16, device=device)

    x_all = conftest.gather_results(x, process_mesh=process_mesh)
    w_all = conftest.gather_results(w, process_mesh=process_mesh)

    result = dist_mm.xw_transpose_fourway(
        x, w, rank, device, model_group=process_mesh["model"]
    )
    result = conftest.gather_results(result, process_mesh=process_mesh)

    assert torch.allclose(x_all @ w_all.mT, result, atol=1e-5)


def test_xw_fourway(device, process_mesh):
    """Test for XW multiplication across four processes."""
    model_rank = process_mesh["model"].rank()

    torch.manual_seed(42 + model_rank)

    rank = dist.get_rank()
    x = torch.randn(1, 4, 16, device=device)
    w = torch.randn(1, 16, 8, device=device)
    x_all = conftest.gather_results(x, process_mesh=process_mesh)
    w_all = conftest.gather_results(w, process_mesh=process_mesh)
    result = dist_mm.xw_fourway(x, w, rank, device, model_group=process_mesh["model"])
    result = conftest.gather_results(result, process_mesh=process_mesh)

    assert torch.allclose(x_all @ (w_all), result, atol=1e-5)


def test_xtw_fourway(device, process_mesh):
    """Test for XW multiplication across four processes."""
    model_rank = process_mesh["model"].rank()

    torch.manual_seed(42 + model_rank)

    rank = dist.get_rank()
    x = torch.randn(2, 32, 64, device=device)
    w = torch.randn(2, 32, 64, device=device)
    x_all = conftest.gather_results(x, process_mesh=process_mesh)
    w_all = conftest.gather_results(w, process_mesh=process_mesh)
    result = dist_mm.x_transpose_w_fourway(
        x, w, rank, device, model_group=process_mesh["model"]
    )
    result = conftest.gather_results(result, process_mesh=process_mesh)
    assert torch.allclose((x_all.mT) @ (w_all), result, atol=1e-5)


def test_fused_xw_xtw_fourway(device, process_mesh):
    """Test for fused XW XTW multiplication across four processes."""
    model_rank = process_mesh["model"].rank()
    torch.manual_seed(42 + model_rank)

    rank = dist.get_rank()
    x = torch.randn(1, 4, 16, device=device)
    w1 = torch.randn(1, 16, 8, device=device)
    w2 = torch.randn(1, 4, 8, device=device)
    x_all = conftest.gather_results(x, process_mesh=process_mesh)
    w1_all = conftest.gather_results(w1, process_mesh=process_mesh)
    w2_all = conftest.gather_results(w2, process_mesh=process_mesh)
    result_xw, result_xtw = dist_mm.fused_xw_xtw_fourway(
        x, w1, w2, rank, device, model_group=process_mesh["model"]
    )
    result_xw = conftest.gather_results(result_xw, process_mesh=process_mesh)
    result_xtw = conftest.gather_results(result_xtw, process_mesh=process_mesh)

    assert torch.allclose((x_all) @ (w1_all), result_xw)
    assert torch.allclose((x_all.mT) @ (w2_all), result_xtw)
