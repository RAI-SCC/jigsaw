import pytest
import torch

from jigsaw.gather import gather_results
from jigsaw.modules.linear._linear import (
    DistributedMatMulXTW,
    DistributedMatMulXW,
    DistributedMatMulXWT,
)
from tests.parity import assert_close

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
        assert_close(
            "xwt", y_gathered, y_ref, 1e-5, 1e-5, device, process_mesh_dict["model"]
        )

    def test_xw(self, process_mesh_dict_2x2, device):
        process_mesh_dict = process_mesh_dict_2x2
        in_dim, mid_dim, out_dim = 10, 20, 8
        model = DistributedMatMulXW(
            local_rank=process_mesh_dict["model"].rank(),
            device=device,
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
        assert_close(
            "xw", y_gathered, y_ref, 1e-5, 1e-5, device, process_mesh_dict["model"]
        )

    def test_xtw(self, process_mesh_dict_2x2, device):
        process_mesh_dict = process_mesh_dict_2x2
        in_dim, mid_dim, out_dim = 10, 20, 8
        model = DistributedMatMulXTW(
            local_rank=process_mesh_dict["model"].rank(),
            device=device,
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
        assert_close(
            "xtw", y_gathered, y_ref, 1e-5, 1e-5, device, process_mesh_dict["model"]
        )
