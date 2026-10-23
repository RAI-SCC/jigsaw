"""Backward of the linear Functions computes only the gradients autograd needs."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.distributed as dist

import jigsaw.modules.linear as linear
from jigsaw.gather import gather_1d_results, gather_results
from jigsaw.modules.linear._linear import (
    DistributedMatMulXTW,
    DistributedMatMulXW,
    DistributedMatMulXWT,
)
from tests.parity import assert_close
from tests.process_mesh import make_process_mesh

pytestmark = [pytest.mark.dist_required]


@pytest.fixture(scope="session")
def process_mesh_dict_dp2_2x2(device: torch.device) -> dict:
    """2x2 model group in each of two dp replicas: group and global ranks differ."""
    return make_process_mesh(
        device_type=device.type,
        mesh_shape=[2, 2, 2],
        mesh_dim_names=["dp", "row", "col"],
    )


#: The distributed meshes the XWT kernels implement (1x1 is not distributed).
MESHES = [
    pytest.param("process_mesh_dict_1x2", marks=pytest.mark.dist(2), id="1x2"),
    pytest.param("process_mesh_dict_2x2", marks=pytest.mark.dist(4), id="2x2"),
    pytest.param("process_mesh_dict_4x2", marks=pytest.mark.dist(8), id="4x2"),
    pytest.param("process_mesh_dict_dp2_2x2", marks=pytest.mark.dist(8), id="dp2-2x2"),
]

#: 2x2 model groups, without and with dp replicas.
MESHES_2X2 = [
    pytest.param("process_mesh_dict_2x2", marks=pytest.mark.dist(4), id="2x2"),
    pytest.param("process_mesh_dict_dp2_2x2", marks=pytest.mark.dist(8), id="dp2-2x2"),
]

#: Which of (x, weight, bias) require a gradient.
REQUIRES = {
    "all": (True, True, True),
    "x_only": (True, False, False),
    "weight_only": (False, True, False),
    "weight_and_bias": (False, True, True),
    "bias_only": (False, False, True),
}


def _count_kernel_calls(config: Any) -> list[int]:
    """Wrap ``config.mm_kernels`` so each call is counted; return the counters."""
    calls = [0] * len(config.mm_kernels)

    def wrap(i: int, kernel: Callable) -> Callable:
        def counted(*args: Any, **kwargs: Any) -> Any:
            calls[i] += 1
            return kernel(*args, **kwargs)

        return counted

    config.mm_kernels = [wrap(i, k) for i, k in enumerate(config.mm_kernels)]
    return calls


@pytest.mark.parametrize("requires", REQUIRES.values(), ids=REQUIRES.keys())
@pytest.mark.parametrize("mesh_name", MESHES)
def test_linear_backward_skips_unneeded_grads(
    device: torch.device,
    request: pytest.FixtureRequest,
    mesh_name: str,
    requires: tuple[bool, bool, bool],
) -> None:
    """Needed gradients match ``torch.nn.Linear``; the rest are not computed."""
    pm = request.getfixturevalue(mesh_name)
    need_x, need_w, need_b = requires
    model_group = pm["model"]
    local_rank = dist.get_rank(group=model_group)
    torch.manual_seed(local_rank)
    rows, cols = pm["row"].size(), pm["col"].size()
    in_dim, out_dim, seq = 8, 8, 8

    layer = linear.DistributedLinear(
        in_dim,
        out_dim,
        local_rank=local_rank,
        device=device,
        model_group=model_group,
        col_group=pm["col"],
        row_group=pm["row"],
        bias=True,
    ).to(device)
    assert isinstance(layer.bias, torch.Tensor)
    ref = torch.nn.Linear(in_dim, out_dim, bias=True).to(device)
    ref.weight.data = gather_results(layer.weight.data, process_mesh_dict=pm)
    ref.bias.data = gather_1d_results(
        layer.bias.data, process_mesh_dict=pm, pm_key="col"
    )
    for module in (layer, ref):
        module.weight.requires_grad_(need_w)
        module.bias.requires_grad_(need_b)
    calls = _count_kernel_calls(layer.config)

    x = torch.randn(1, seq // rows, in_dim // cols, device=device)
    x_ref = gather_results(x, process_mesh_dict=pm)
    x.requires_grad_(need_x)
    x_ref.requires_grad_(need_x)
    y = layer(x)
    grad_y = torch.randn_like(y)
    y.backward(grad_y)
    ref(x_ref).backward(gather_results(grad_y, process_mesh_dict=pm))

    # mm_kernels: 0 xwt (forward), 1 xw, 2 xtw, 3 fused xw + xtw.
    expected = [1, 0, 0, 0]
    if need_x and need_w:
        expected[3] = 1
    elif need_x:
        expected[1] = 1
    elif need_w:
        expected[2] = 1
    assert calls == expected

    assert (x.grad is not None) == need_x
    assert (layer.weight.grad is not None) == need_w
    assert (layer.bias.grad is not None) == need_b
    if need_x:
        assert isinstance(x.grad, torch.Tensor)
        x_grad = gather_results(x.grad, process_mesh_dict=pm)
        assert_close("x grad", x_grad, x_ref.grad, 1e-5, 1e-5, device, model_group)
    if need_w:
        assert isinstance(layer.weight.grad, torch.Tensor)
        w_grad = gather_results(layer.weight.grad, process_mesh_dict=pm)
        assert_close(
            "weight grad", w_grad, ref.weight.grad, 1e-5, 1e-5, device, model_group
        )
    if need_b:
        assert isinstance(layer.bias.grad, torch.Tensor)
        b_grad = gather_1d_results(layer.bias.grad, process_mesh_dict=pm, pm_key="col")
        assert_close(
            "bias grad", b_grad, ref.bias.grad, 1e-5, 1e-5, device, model_group
        )


#: Module, how to build the operands from global ``a`` and ``b``, and the reference.
MATMULS = {
    "xwt": (DistributedMatMulXWT, lambda a, b: (a, b.mT), lambda x, w: x @ w.mT),
    "xw": (DistributedMatMulXW, lambda a, b: (a, b), lambda x, w: x @ w),
    "xtw": (DistributedMatMulXTW, lambda a, b: (a.mT, b), lambda x, w: x.mT @ w),
}


@pytest.mark.parametrize("need_x", [True, False], ids=["x", "no_x"])
@pytest.mark.parametrize("need_w", [True, False], ids=["w", "no_w"])
@pytest.mark.parametrize("op", MATMULS.values(), ids=MATMULS.keys())
@pytest.mark.parametrize("mesh_name", MESHES_2X2)
def test_matmul_backward_skips_unneeded_grads(
    device: torch.device,
    request: pytest.FixtureRequest,
    mesh_name: str,
    op: tuple,
    need_x: bool,
    need_w: bool,
) -> None:
    """Each matmul Function returns only the gradients that are needed."""
    if not (need_x or need_w):
        pytest.skip("nothing requires a gradient, so backward never runs")
    pm = request.getfixturevalue(mesh_name)
    module_cls, operands, reference = op
    model_group = pm["model"]
    torch.manual_seed(model_group.rank())
    model = module_cls(
        local_rank=model_group.rank(), device=device, model_group=model_group
    )

    a = torch.randn(10 // 2, 20 // 2, device=device)
    b = torch.randn(20 // 2, 8 // 2, device=device)
    x, w = (t.contiguous() for t in operands(a, b))
    x_ref = gather_results(x, pm)
    w_ref = gather_results(w, pm)
    for t, need in ((x, need_x), (w, need_w), (x_ref, need_x), (w_ref, need_w)):
        t.requires_grad_(need)

    y = model(x, w)
    grad_y = torch.randn_like(y)
    y.backward(grad_y)
    reference(x_ref, w_ref).backward(gather_results(grad_y, pm))

    for name, t, t_ref, need in (
        ("x", x, x_ref, need_x),
        ("w", w, w_ref, need_w),
    ):
        assert (t.grad is not None) == need
        if need:
            assert isinstance(t.grad, torch.Tensor)
            assert_close(
                f"{name} grad",
                gather_results(t.grad, pm),
                t_ref.grad,
                1e-5,
                1e-5,
                device,
                model_group,
            )
