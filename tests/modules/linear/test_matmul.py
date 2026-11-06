from typing import cast

import pytest
import torch

from jigsaw.gather import gather_results
from jigsaw.modules.linear import LinearConfig
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


#: (module, local operands from a (10/2, 20/2) and b (20/2, 8/2), reference op)
_OPS = {
    "xwt": (DistributedMatMulXWT, lambda a, b: (a, b.mT), lambda x, w: x @ w.mT),
    "xw": (DistributedMatMulXW, lambda a, b: (a, b), lambda x, w: x @ w),
    "xtw": (DistributedMatMulXTW, lambda a, b: (a.mT, b), lambda x, w: x.mT @ w),
}


def _strided(t: torch.Tensor) -> torch.Tensor:
    """Return ``t``'s values in a non-contiguous (column-major) layout."""
    return t.mT.contiguous().mT


@pytest.mark.dist(4)
@pytest.mark.parametrize("need_x", [True, False], ids=["x", "no_x"])
@pytest.mark.parametrize("need_w", [True, False], ids=["w", "no_w"])
@pytest.mark.parametrize("comm", ["all_to_all", "isend", "batch_isend_irecv"])
@pytest.mark.parametrize("op", _OPS.values(), ids=_OPS.keys())
def test_strided_operands_and_summed_loss(
    process_mesh_dict_2x2: dict,
    device: torch.device,
    op: tuple,
    comm: str,
    need_w: bool,
    need_x: bool,
) -> None:
    """Non-contiguous operands and a stride-0 ``grad_output`` match torch."""
    if comm == "all_to_all" and device.type != "cuda":
        pytest.skip("all_to_all needs CUDA")
    pm = process_mesh_dict_2x2
    group = pm["model"]
    module_cls, operands, reference = op
    model = module_cls(
        device=device,
        model_group=group,
        config=LinearConfig(group, linear_layer=False, device=device, comm=comm),
    )
    torch.manual_seed(group.rank())
    a = torch.randn(10 // 2, 20 // 2, device=device)
    b = torch.randn(20 // 2, 8 // 2, device=device)
    x, w = (_strided(t.contiguous()) for t in operands(a, b))
    assert not (x.is_contiguous() or w.is_contiguous())
    x_ref = gather_results(x.contiguous(), pm)
    w_ref = gather_results(w.contiguous(), pm)
    for t, need in ((x, need_x), (w, need_w), (x_ref, need_x), (w_ref, need_w)):
        t.requires_grad_(need)

    y = model(x, w)
    y_ref = reference(x_ref, w_ref)
    assert_close(
        "output", gather_results(y.detach(), pm), y_ref, 1e-5, 1e-5, device, group
    )
    if not (need_x or need_w):
        return
    # Each rank holds a disjoint tile of the output: local sums add up to the global one.
    y.sum().backward()
    y_ref.sum().backward()
    for name, t, t_ref, need in (("x", x, x_ref, need_x), ("w", w, w_ref, need_w)):
        if need:
            assert_close(
                f"{name} grad",
                gather_results(cast(torch.Tensor, t.grad), pm),
                t_ref.grad,
                1e-5,
                1e-5,
                device,
                group,
            )
