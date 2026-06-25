import pytest
import torch
import torch.distributed as dist

import jigsaw.distributed_matmul as dist_mm
import jigsaw.distributed_matmul_p2p as dist_mm_p2p
from jigsaw.utils_aggregate import gather_results

pytestmark = [pytest.mark.distributed_required, pytest.mark.dist(8)]


def test_xwt_eightway(device: torch.device, process_mesh_dict_4x2: dict) -> None:
    """Test for XWT multiplication across four processes."""
    rank = dist.get_rank()
    torch.manual_seed(42 + rank)

    # global dimensions
    x_in, x_out, w_out = 16, 24, 24
    x = torch.randn(1, x_in // 4, x_out // 2, device=device)
    w = torch.randn(1, x_out // 4, w_out // 2, device=device)

    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_4x2)
    w_all = gather_results(w, process_mesh_dict=process_mesh_dict_4x2)

    if device.type == "cuda":
        result = dist_mm.xw_transpose_eightway(
            x, w, rank, device, model_group=process_mesh_dict_4x2["model"]
        )
    else:
        result = dist_mm_p2p.xw_transpose_eightway(
            x, w, rank, device, model_group=process_mesh_dict_4x2["model"]
        )
    result = gather_results(result, process_mesh_dict=process_mesh_dict_4x2)

    assert torch.allclose(x_all @ w_all.mT, result, atol=1e-5)


def test_xw_eightway(device: torch.device, process_mesh_dict_4x2: dict) -> None:
    """Test for XW multiplication across four processes."""
    model_rank = process_mesh_dict_4x2["model"].rank()

    torch.manual_seed(42 + model_rank)

    rank = dist.get_rank()

    x_in, x_out, w_out = 16, 24, 24
    x = torch.randn(1, x_in // 4, x_out // 2, device=device)
    w = torch.randn(1, x_out // 4, w_out // 2, device=device)

    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_4x2)
    w_all = gather_results(w, process_mesh_dict=process_mesh_dict_4x2)

    if device.type == "cuda":
        result = dist_mm.xw_eightway(
            x, w, rank, device, model_group=process_mesh_dict_4x2["model"]
        )
    else:
        result = dist_mm_p2p.xw_eightway(
            x, w, rank, device, model_group=process_mesh_dict_4x2["model"]
        )
    result = gather_results(result, process_mesh_dict=process_mesh_dict_4x2)
    print(f"difference {(x_all @ w_all).sum() - result.sum()}")
    assert torch.allclose(x_all @ w_all, result, atol=1e-5)


def test_xtw_eightway(device: torch.device, process_mesh_dict_4x2: dict) -> None:
    """Test for XTW multiplication across four processes."""
    model_rank = process_mesh_dict_4x2["model"].rank()

    torch.manual_seed(42 + model_rank)

    rank = dist.get_rank()
    x = torch.randn(2, 32, 64, device=device)
    w = torch.randn(2, 32, 64, device=device)
    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_4x2)
    w_all = gather_results(w, process_mesh_dict=process_mesh_dict_4x2)
    if device.type == "cuda":
        result = dist_mm.x_transpose_w_eightway(
            x, w, rank, device, model_group=process_mesh_dict_4x2["model"]
        )
    else:
        result = dist_mm_p2p.x_transpose_w_eightway(
            x, w, rank, device, model_group=process_mesh_dict_4x2["model"]
        )
    result = gather_results(result, process_mesh_dict=process_mesh_dict_4x2)
    assert torch.allclose((x_all.mT) @ (w_all), result, atol=1e-5)


def test_fused_xw_xtw_eightway(
    device: torch.device, process_mesh_dict_4x2: dict
) -> None:
    """Test for fused XW XTW multiplication across four processes."""
    model_rank = process_mesh_dict_4x2["model"].rank()
    torch.manual_seed(42 + model_rank)

    rank = dist.get_rank()
    x = torch.randn(1, 16 // 4, 32 // 2, device=device)
    w1 = torch.randn(1, 32 // 4, 16 // 2, device=device)
    w2 = torch.randn(1, 16 // 4, 16 // 2, device=device)
    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_4x2)
    w1_all = gather_results(w1, process_mesh_dict=process_mesh_dict_4x2)
    w2_all = gather_results(w2, process_mesh_dict=process_mesh_dict_4x2)
    if device.type == "cuda":
        result_xw, result_xtw = dist_mm.fused_xw_xtw_eightway(
            x, w1, w2, rank, device, model_group=process_mesh_dict_4x2["model"]
        )
    else:
        result_xw, result_xtw = dist_mm_p2p.fused_xw_xtw_eightway(
            x, w1, w2, rank, device, model_group=process_mesh_dict_4x2["model"]
        )
    result_xw = gather_results(result_xw, process_mesh_dict=process_mesh_dict_4x2)
    result_xtw = gather_results(result_xtw, process_mesh_dict=process_mesh_dict_4x2)
    assert torch.allclose(x_all @ w1_all, result_xw, atol=1e-5)
    assert torch.allclose(x_all.mT @ w2_all, result_xtw, atol=1e-5)
