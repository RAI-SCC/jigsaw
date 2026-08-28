import pytest
import torch

import jigsaw.modules.linear.distributed_matmul as dist_mm
import jigsaw.modules.linear.distributed_matmul_batchedp2p as dist_mm_batchedp2p
import jigsaw.modules.linear.distributed_matmul_p2p as dist_mm_p2p
from jigsaw.utils_aggregate import gather_results

pytestmark = [pytest.mark.dist_required]

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
def test_xwt_twoway(
    device: torch.device, process_mesh_dict_2x1x2: dict, cpu_impl
) -> None:
    """Test for XWT multiplication across two processes."""
    model_group = process_mesh_dict_2x1x2["model"]
    local_rank = model_group.rank()
    torch.manual_seed(42 + local_rank)

    x = torch.randn(1, 4, 16 // 2, device=device)
    w = torch.randn(1, 4, 16 // 2, device=device)

    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_2x1x2)
    w_all = gather_results(w, process_mesh_dict=process_mesh_dict_2x1x2)

    result = _impl(device, cpu_impl).xw_transpose_twoway(
        x, w, local_rank, device, model_group=model_group
    )
    result = gather_results(result, process_mesh_dict=process_mesh_dict_2x1x2)

    assert torch.allclose(x_all @ w_all.mT, result, atol=1e-5)


@pytest.mark.parametrize("cpu_impl", CPU_IMPLS)
def test_xw_twoway(
    device: torch.device, process_mesh_dict_2x1x2: dict, cpu_impl
) -> None:
    """Test for XW multiplication across two processes."""
    model_group = process_mesh_dict_2x1x2["model"]
    local_rank = model_group.rank()
    torch.manual_seed(42 + local_rank)

    x = torch.randn(1, 4, 16 // 2, device=device)
    w = torch.randn(1, 16, 8 // 2, device=device)
    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_2x1x2)
    w_all = gather_results(w, process_mesh_dict=process_mesh_dict_2x1x2)
    result = _impl(device, cpu_impl).xw_twoway(
        x, w, local_rank, device, model_group=model_group
    )
    result = gather_results(result, process_mesh_dict=process_mesh_dict_2x1x2)

    assert torch.allclose(x_all @ (w_all), result, atol=1e-5)


@pytest.mark.parametrize("cpu_impl", CPU_IMPLS)
def test_xtw_twoway(
    device: torch.device, process_mesh_dict_2x1x2: dict, cpu_impl
) -> None:
    """Test for XTW multiplication across two processes."""
    model_group = process_mesh_dict_2x1x2["model"]
    local_rank = model_group.rank()
    torch.manual_seed(42 + local_rank)

    x = torch.randn(1, 16, 4 // 2, device=device)
    w = torch.randn(1, 16, 8 // 2, device=device)
    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_2x1x2)
    w_all = gather_results(w, process_mesh_dict=process_mesh_dict_2x1x2)
    result = _impl(device, cpu_impl).x_transpose_w_twoway(
        x, w, local_rank, device, model_group=model_group
    )
    result = gather_results(result, process_mesh_dict=process_mesh_dict_2x1x2)

    assert torch.allclose((x_all.mT) @ (w_all), result, atol=1e-5)


@pytest.mark.parametrize("cpu_impl", CPU_IMPLS)
def test_fused_xw_xtw_twoway(
    device: torch.device, process_mesh_dict_2x1x2: dict, cpu_impl
) -> None:
    """Test for fused XW XTW multiplication across two processes."""
    model_group = process_mesh_dict_2x1x2["model"]
    local_rank = model_group.rank()
    torch.manual_seed(42 + local_rank)

    x = torch.randn(1, 4, 16 // 2, device=device)
    w1 = torch.randn(1, 16, 8 // 2, device=device)
    w2 = torch.randn(1, 4, 8 // 2, device=device)
    x_all = gather_results(x, process_mesh_dict=process_mesh_dict_2x1x2)
    w1_all = gather_results(w1, process_mesh_dict=process_mesh_dict_2x1x2)
    w2_all = gather_results(w2, process_mesh_dict=process_mesh_dict_2x1x2)
    result_xw, result_xtw = _impl(device, cpu_impl).fused_xw_xtw_twoway(
        x, w1, w2, local_rank, device, model_group=model_group
    )
    result_xw = gather_results(result_xw, process_mesh_dict=process_mesh_dict_2x1x2)
    result_xtw = gather_results(result_xtw, process_mesh_dict=process_mesh_dict_2x1x2)

    assert torch.allclose((x_all) @ (w1_all), result_xw, atol=1e-5)
    assert torch.allclose((x_all.mT) @ (w2_all), result_xtw, atol=1e-5)
