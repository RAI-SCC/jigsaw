"""Distributed conv / transpose conv under ``torch.vmap``, vs a per-member loop.

Layers are built without any flag; the batching rule shared by ``_ConvNd`` and
``_ConvTransposeNd`` detects vmapped weights (stacked ensemble members) at call
time and makes one un-vmapped call, so the collectives run once.

Backward is checked through ordinary autograd on the vmapped forward.
``vmap(grad(...))`` is out of scope: the ``c10d`` collectives in the backward
have no batching rule.

Regimes on a 2x2 mesh (channel and spatial groups both split): no-halo
(``stride == kernel_size``), halo, and channel-local grouped conv (no custom
Function, the native conv's own batching rule).
"""

import pytest
import torch
import torch.distributed as dist
from torch.func import functional_call, stack_module_state

import jigsaw.modules.convolution as convolution
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]

E = 2
IN = OUT = 4
H = W = 8
_FWD_TOL = dict(atol=1e-5, rtol=1e-4)
_GRAD_TOL = dict(atol=1e-4, rtol=1e-3)

_CASES = {
    "no_halo": dict(kernel_size=2, stride=2, padding=0),
    "halo": dict(kernel_size=3, stride=1, padding=1),
    "channel_local": dict(kernel_size=2, stride=2, padding=0, groups=2),
}
_MODULES = {
    "conv2d": convolution.DistributedConv2d,
    "conv2d_transpose": convolution.DistributedConvTranspose2d,
}


def _build(cls, mesh: dict, device: torch.device, conv_kw: dict) -> torch.nn.Module:
    return cls(
        IN,
        OUT,
        device=device,
        spatial_group=mesh["row"],
        channel_group=mesh["col"],
        bias=True,
        channel_comm="p2p",
        **conv_kw,
    ).to(device)


def _local_x(mesh: dict, device: torch.device) -> torch.Tensor:
    """``(E, N, C_in / col, H, W / row)`` slice of one global tensor."""
    torch.manual_seed(20240607)
    x = torch.randn(E, 3, IN, H, W, device=device)
    col, row = mesh["col"], mesh["row"]
    c, w = IN // col.size(), W // row.size()
    x = x[:, :, col.rank() * c : (col.rank() + 1) * c]
    x = x[..., row.rank() * w : (row.rank() + 1) * w]
    return x.contiguous()


def _check(name, got, ref, mesh, device, tol) -> None:
    assert_close(name, got, ref, **tol, device=device, group=mesh["model"])


@pytest.mark.dist(4)
@pytest.mark.parametrize("x_shared", [False, True], ids=["x_batched", "x_shared"])
@pytest.mark.parametrize("regime", list(_CASES))
@pytest.mark.parametrize("module", list(_MODULES))
def test_member_weights(
    device: torch.device,
    process_mesh_dict_2x2: dict,
    module: str,
    regime: str,
    x_shared: bool,
) -> None:
    """Stacked per-member weights: forward and backward match a loop."""
    mesh = process_mesh_dict_2x2
    members = [_build(_MODULES[module], mesh, device, _CASES[regime]) for _ in range(E)]
    base = _build(_MODULES[module], mesh, device, _CASES[regime])
    params, buffers = stack_module_state(members)

    x = _local_x(mesh, device)
    if x_shared:
        x = x[:1].contiguous()
    x.requires_grad_(True)
    x_m = [x[0 if x_shared else e] for e in range(E)]

    got = torch.vmap(
        lambda p, b, xi: functional_call(base, (p, b), (xi,)),
        in_dims=(0, 0, None if x_shared else 0),
    )(params, buffers, x[0] if x_shared else x)
    ref = torch.stack([m(xi) for m, xi in zip(members, x_m)])
    _check(f"{module} {regime} forward", got, ref, mesh, device, _FWD_TOL)

    g = torch.randn_like(ref)
    (got * g).sum().backward()
    grad_x = x.grad.clone()
    x.grad = None
    (ref * g).sum().backward()
    _check(f"{module} {regime} grad_x", grad_x, x.grad, mesh, device, _GRAD_TOL)

    # functional_call bypasses the parameters' grad-sync hooks; replay the
    # spatial-group SUM the member modules got.
    for name, p in params.items():
        grad = p.grad.clone()
        for group in base._spatial_sync_groups:
            dist.all_reduce(grad, group=group)
        ref_grad = torch.stack([dict(m.named_parameters())[name].grad for m in members])
        _check(
            f"{module} {regime} grad[{name}]", grad, ref_grad, mesh, device, _GRAD_TOL
        )


@pytest.mark.dist(4)
@pytest.mark.parametrize("regime", list(_CASES))
@pytest.mark.parametrize("module", list(_MODULES))
def test_shared_weights(
    device: torch.device, process_mesh_dict_2x2: dict, module: str, regime: str
) -> None:
    """Only the input is vmapped: the axis folds into the batch."""
    mesh = process_mesh_dict_2x2
    model = _build(_MODULES[module], mesh, device, _CASES[regime])
    x = _local_x(mesh, device).requires_grad_(True)

    got = torch.vmap(model)(x)
    ref = torch.stack([model(xi) for xi in x])
    _check(f"{module} {regime} forward", got, ref, mesh, device, _FWD_TOL)

    g = torch.randn_like(ref)
    (got * g).sum().backward()
    grad_x = x.grad.clone()
    x.grad = None
    (ref * g).sum().backward()
    _check(f"{module} {regime} grad_x", grad_x, x.grad, mesh, device, _GRAD_TOL)
