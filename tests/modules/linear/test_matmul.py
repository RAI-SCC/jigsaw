import pytest
import torch

from jigsaw.modules.linear.linear import (
    DistributedMatMulXTW,
    DistributedMatMulXW,
    DistributedMatMulXWT,
)
from jigsaw.utils_aggregate import gather_results

pytestmark = [pytest.mark.dist_required]


class TestDistributedMatMul:
    """``DistributedMatMul{XWT,XW,XTW}`` modules on a 2x2 mesh."""

    pytestmark = [pytest.mark.dist(4)]

    def test_xwt(self, process_mesh_dict_2x2, device):
        process_mesh_dict = process_mesh_dict_2x2
        in_dim, mid_dim, out_dim = 10, 20, 8
        model = DistributedMatMulXWT(
            local_rank=process_mesh_dict["model"].rank(),
            device=device,
            parallelism=process_mesh_dict["model"].size(),
            model_group=process_mesh_dict["model"],
        )

        a = torch.randn(in_dim // 2, mid_dim // 2, device=device)
        b = torch.randn(mid_dim // 2, out_dim // 2, device=device).mT.contiguous()
        y = model(a, b)

        ref_a = gather_results(a, process_mesh_dict, pm_key_row="row", pm_key_col="col")
        ref_b = gather_results(b, process_mesh_dict, pm_key_row="row", pm_key_col="col")
        y_gathered = gather_results(
            y, process_mesh_dict, pm_key_row="row", pm_key_col="col"
        )

        y_ref = ref_a @ ref_b.mT
        assert torch.allclose(y_gathered, y_ref, atol=1e-5)

    def test_xw(self, process_mesh_dict_2x2, device):
        process_mesh_dict = process_mesh_dict_2x2
        in_dim, mid_dim, out_dim = 10, 20, 8
        model = DistributedMatMulXW(
            local_rank=process_mesh_dict["model"].rank(),
            device=device,
            parallelism=process_mesh_dict["model"].size(),
            model_group=process_mesh_dict["model"],
        )

        a = torch.randn(in_dim // 2, mid_dim // 2, device=device)
        b = torch.randn(mid_dim // 2, out_dim // 2, device=device)
        y = model(a, b)

        ref_a = gather_results(a, process_mesh_dict, pm_key_row="row", pm_key_col="col")
        ref_b = gather_results(b, process_mesh_dict, pm_key_row="row", pm_key_col="col")
        y_gathered = gather_results(
            y, process_mesh_dict, pm_key_row="row", pm_key_col="col"
        )

        y_ref = ref_a @ ref_b
        assert torch.allclose(y_gathered, y_ref, atol=1e-5)

    def test_xtw(self, process_mesh_dict_2x2, device):
        process_mesh_dict = process_mesh_dict_2x2
        in_dim, mid_dim, out_dim = 10, 20, 8
        model = DistributedMatMulXTW(
            local_rank=process_mesh_dict["model"].rank(),
            device=device,
            parallelism=process_mesh_dict["model"].size(),
            model_group=process_mesh_dict["model"],
        )

        a = torch.randn(in_dim // 2, mid_dim // 2, device=device).mT.contiguous()
        b = torch.randn(mid_dim // 2, out_dim // 2, device=device)
        y = model(a, b)

        ref_a = gather_results(a, process_mesh_dict, pm_key_row="row", pm_key_col="col")
        ref_b = gather_results(b, process_mesh_dict, pm_key_row="row", pm_key_col="col")
        y_gathered = gather_results(
            y, process_mesh_dict, pm_key_row="row", pm_key_col="col"
        )

        y_ref = ref_a.mT @ ref_b
        assert torch.allclose(y_gathered, y_ref, atol=1e-5)
