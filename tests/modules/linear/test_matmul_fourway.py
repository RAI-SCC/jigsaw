import pytest
import torch
import torch.distributed as dist

import jigsaw.modules.linear.distributed_matmul as dist_mm
import jigsaw.modules.linear.distributed_matmul_p2p as dist_mm_p2p
from jigsaw.utils_aggregate import gather_results

pytestmark = [pytest.mark.dist_required, pytest.mark.dist(4)]


def test_xwt_fourway(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Test for XWT multiplication across four processes."""
    rank = dist.get_rank()
    torch.manual_seed(42 + rank)

    x = torch.randn(1, 4, 16, device=device)
    w = torch.randn(1, 4, 16, device=device)

    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_2x2)
    w_all = gather_results(w, process_mesh_dict=process_mesh_dict_2x2)

    if device.type == "cuda":
        result = dist_mm.xw_transpose_fourway(
            x, w, rank, device, model_group=process_mesh_dict_2x2["model"]
        )
    else:
        result = dist_mm_p2p.xw_transpose_fourway(
            x, w, rank, device, model_group=process_mesh_dict_2x2["model"]
        )
    result = gather_results(result, process_mesh_dict=process_mesh_dict_2x2)

    assert torch.allclose(x_all @ w_all.mT, result, atol=1e-5)


def test_xw_fourway(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Test for XW multiplication across four processes."""
    model_rank = process_mesh_dict_2x2["model"].rank()

    torch.manual_seed(42 + model_rank)

    rank = dist.get_rank()
    x = torch.randn(1, 4, 16, device=device)
    w = torch.randn(1, 16, 8, device=device)
    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_2x2)
    w_all = gather_results(w, process_mesh_dict=process_mesh_dict_2x2)

    if device.type == "cuda":
        result = dist_mm.xw_fourway(
            x, w, rank, device, model_group=process_mesh_dict_2x2["model"]
        )
    else:
        result = dist_mm_p2p.xw_fourway(
            x, w, rank, device, model_group=process_mesh_dict_2x2["model"]
        )
    result = gather_results(result, process_mesh_dict=process_mesh_dict_2x2)

    assert torch.allclose(x_all @ w_all, result, atol=1e-5)


def test_xtw_fourway(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Test for XTW multiplication across four processes."""
    model_rank = process_mesh_dict_2x2["model"].rank()

    torch.manual_seed(42 + model_rank)

    rank = dist.get_rank()
    x = torch.randn(2, 32, 64, device=device)
    w = torch.randn(2, 32, 64, device=device)
    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_2x2)
    w_all = gather_results(w, process_mesh_dict=process_mesh_dict_2x2)
    if device.type == "cuda":
        result = dist_mm.x_transpose_w_fourway(
            x, w, rank, device, model_group=process_mesh_dict_2x2["model"]
        )
    else:
        result = dist_mm_p2p.x_transpose_w_fourway(
            x, w, rank, device, model_group=process_mesh_dict_2x2["model"]
        )
    result = gather_results(result, process_mesh_dict=process_mesh_dict_2x2)
    assert torch.allclose((x_all.mT) @ (w_all), result, atol=1e-5)


def test_fused_xw_xtw_fourway(
    device: torch.device, process_mesh_dict_2x2: dict
) -> None:
    """Test for fused XW XTW multiplication across four processes."""
    model_rank = process_mesh_dict_2x2["model"].rank()
    torch.manual_seed(42 + model_rank)

    rank = dist.get_rank()
    x = torch.randn(1, 4, 16, device=device)
    w1 = torch.randn(1, 16, 8, device=device)
    w2 = torch.randn(1, 4, 8, device=device)
    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_2x2)
    w1_all = gather_results(w1, process_mesh_dict=process_mesh_dict_2x2)
    w2_all = gather_results(w2, process_mesh_dict=process_mesh_dict_2x2)

    if device.type == "cuda":
        result_xw, result_xtw = dist_mm.fused_xw_xtw_fourway(
            x, w1, w2, rank, device, model_group=process_mesh_dict_2x2["model"]
        )
    else:
        result_xw, result_xtw = dist_mm_p2p.fused_xw_xtw_fourway(
            x, w1, w2, rank, device, model_group=process_mesh_dict_2x2["model"]
        )
    result_xw = gather_results(result_xw, process_mesh_dict=process_mesh_dict_2x2)
    result_xtw = gather_results(result_xtw, process_mesh_dict=process_mesh_dict_2x2)
    assert torch.allclose(x_all @ w1_all, result_xw, atol=1e-5)
    assert torch.allclose(x_all.mT @ w2_all, result_xtw, atol=1e-5)
