import pytest
import torch
import torch.distributed as dist

import jigsaw.modules.linear.distributed_matmul as dist_mm
import jigsaw.modules.linear.distributed_matmul_batchedp2p as dist_mm_batchedp2p
import jigsaw.modules.linear.distributed_matmul_p2p as dist_mm_p2p
from jigsaw.utils_aggregate import gather_results

pytestmark = [pytest.mark.dist_required, pytest.mark.dist(4)]

# CPU is served by the point-to-point implementations; run every CPU test against
# both the plain and the batched (`batch_isend_irecv`) variant.
CPU_IMPLS = [
    pytest.param(dist_mm_p2p, id="p2p"),
    pytest.param(dist_mm_batchedp2p, id="batched_p2p"),
]


def _impl(device: torch.device, cpu_impl):
    """Pick the matmul implementation module for the current device."""
    if device.type == "cuda":
        if cpu_impl is not dist_mm_p2p:
            pytest.skip("CUDA uses the collective implementation, not the p2p ones")
        return dist_mm
    return cpu_impl


@pytest.mark.parametrize("cpu_impl", CPU_IMPLS)
def test_xwt_fourway(
    device: torch.device, process_mesh_dict_2x2: dict, cpu_impl
) -> None:
    """Test for XWT multiplication across four processes."""
    rank = dist.get_rank()
    torch.manual_seed(42 + rank)

    x = torch.randn(1, 4, 16, device=device)
    w = torch.randn(1, 4, 16, device=device)

    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_2x2)
    w_all = gather_results(w, process_mesh_dict=process_mesh_dict_2x2)

    result = _impl(device, cpu_impl).xw_transpose_fourway(
        x, w, rank, device, model_group=process_mesh_dict_2x2["model"]
    )
    result = gather_results(result, process_mesh_dict=process_mesh_dict_2x2)

    assert torch.allclose(x_all @ w_all.mT, result, atol=1e-5)


@pytest.mark.parametrize("cpu_impl", CPU_IMPLS)
def test_xw_fourway(
    device: torch.device, process_mesh_dict_2x2: dict, cpu_impl
) -> None:
    """Test for XW multiplication across four processes."""
    model_rank = process_mesh_dict_2x2["model"].rank()

    torch.manual_seed(42 + model_rank)

    rank = dist.get_rank()
    x = torch.randn(1, 4, 16, device=device)
    w = torch.randn(1, 16, 8, device=device)
    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_2x2)
    w_all = gather_results(w, process_mesh_dict=process_mesh_dict_2x2)

    result = _impl(device, cpu_impl).xw_fourway(
        x, w, rank, device, model_group=process_mesh_dict_2x2["model"]
    )
    result = gather_results(result, process_mesh_dict=process_mesh_dict_2x2)

    assert torch.allclose(x_all @ w_all, result, atol=1e-5)


@pytest.mark.parametrize("cpu_impl", CPU_IMPLS)
def test_xtw_fourway(
    device: torch.device, process_mesh_dict_2x2: dict, cpu_impl
) -> None:
    """Test for XTW multiplication across four processes."""
    model_rank = process_mesh_dict_2x2["model"].rank()

    torch.manual_seed(42 + model_rank)

    rank = dist.get_rank()
    x = torch.randn(2, 32, 64, device=device)
    w = torch.randn(2, 32, 64, device=device)
    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_2x2)
    w_all = gather_results(w, process_mesh_dict=process_mesh_dict_2x2)
    result = _impl(device, cpu_impl).x_transpose_w_fourway(
        x, w, rank, device, model_group=process_mesh_dict_2x2["model"]
    )
    result = gather_results(result, process_mesh_dict=process_mesh_dict_2x2)
    assert torch.allclose((x_all.mT) @ (w_all), result, atol=1e-5)


@pytest.mark.parametrize("cpu_impl", CPU_IMPLS)
def test_fused_xw_xtw_fourway(
    device: torch.device, process_mesh_dict_2x2: dict, cpu_impl
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

    result_xw, result_xtw = _impl(device, cpu_impl).fused_xw_xtw_fourway(
        x, w1, w2, rank, device, model_group=process_mesh_dict_2x2["model"]
    )
    result_xw = gather_results(result_xw, process_mesh_dict=process_mesh_dict_2x2)
    result_xtw = gather_results(result_xtw, process_mesh_dict=process_mesh_dict_2x2)
    assert torch.allclose(x_all @ w1_all, result_xw, atol=1e-5)
    assert torch.allclose(x_all.mT @ w2_all, result_xtw, atol=1e-5)
