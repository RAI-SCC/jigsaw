"""Distributed conv / transpose-conv under ``vmap(functional_call(...))``.

There is no ``bayesian`` construction flag any more: the ensemble (members) axis
is owned by
:meth:`~jigsaw.modules.convolution.conv.functional.ConvNd.vmap` /
:meth:`~jigsaw.modules.convolution.conv_transpose.functional.ConvTransposeNd.vmap`,
which expose it as an extra leading batch dim and make one un-vmapped ``apply``:
the channel-gather / halo collectives run once (``E`` x the payload, not ``E``
rounds) and the per-member weights ride a ``torch.vmap`` of only the local
``F.conv*`` / ``convolution_backward``.

Reference is a per-member Python loop over stand-alone modules holding the same
weights, so a mismatch is the vmap members handling, not the sharding. Unlike the
attention ensemble-vmap tests (NCCL ``all_to_all``), the conv channel driver is
gloo-safe (``channel_comm="p2p"``), so this runs on CPU.

The backward is checked through *ordinary* autograd on the ``vmap``-ed forward
(the members axis rides ``ConvNd.backward`` as a real leading batch dim, so the
collectives are the same ones the forward already ran). ``vmap(grad(...))`` --
functorch composing the backward *inside* ``vmap`` -- is a separate,
backend-agnostic gap: the distributed collectives (``reduce_scatter`` /
``all_to_all`` / ``send``) have no functorch batching rule. That is shared with
the linear / attention ensemble path and out of scope here.

Every case is exercised in both regimes: no-halo (``stride == kernel_size``) and
halo (``stride != kernel_size``), on a 2x2 mesh so the channel group *and* the
spatial group are really split.
"""

import pytest
import torch
import torch.distributed as dist
from torch.func import functional_call, stack_module_state, vmap

import jigsaw.modules.convolution as convolution
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]

E = 2
IN = OUT = 4
H = W = 8
_FWD_TOL = dict(atol=1e-5, rtol=1e-4)
_GRAD_TOL = dict(atol=1e-4, rtol=1e-3)


def _copy_weights(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    with torch.no_grad():
        for (_, p_dst), (_, p_src) in zip(
            dst.named_parameters(), src.named_parameters(), strict=True
        ):
            p_dst.copy_(p_src)


def _build(cls, mesh: dict, device: torch.device, **conv_kw) -> torch.nn.Module:
    kernel_size = conv_kw.pop("kernel_size", 2)
    return cls(
        IN,
        OUT,
        kernel_size,
        device=device,
        spatial_group=mesh["row"],
        channel_group=mesh["col"],
        bias=True,
        dtype=torch.float32,
        channel_comm="p2p",
        **conv_kw,
    ).to(device)


def _local_x(mesh: dict, device: torch.device) -> torch.Tensor:
    """``(E, N, C_in / col, H, W / row)`` -- the same global tensor on every rank,
    channel-sharded along ``col`` and width-sharded along ``row``."""
    torch.manual_seed(20240607)
    x = torch.randn(E, 3, IN, H, W, device=device)
    col, row = mesh["col"], mesh["row"]
    c = IN // col.size()
    w = W // row.size()
    x = x[:, :, col.rank() * c : (col.rank() + 1) * c]
    x = x[..., row.rank() * w : (row.rank() + 1) * w]
    return x.contiguous()


def _members_and_ref(cls, mesh, device, conv_kw):
    b_models = [_build(cls, mesh, device, **conv_kw) for _ in range(E)]
    params, buffers = stack_module_state(b_models)
    for p in params.values():
        p.requires_grad_(True)
    ref_models = [_build(cls, mesh, device, **conv_kw) for _ in range(E)]
    for rm, bm in zip(ref_models, b_models, strict=True):
        _copy_weights(rm, bm)
    return b_models[0], params, buffers, ref_models, _local_x(mesh, device)


def _forward(cls, mesh, device, conv_kw):
    base, params, buffers, ref_models, x_e = _members_and_ref(
        cls, mesh, device, conv_kw
    )
    got = vmap(lambda p, b, x: functional_call(base, (p, b), (x,)))(
        params, buffers, x_e
    )
    ref = torch.stack([ref_models[i](x_e[i].contiguous()) for i in range(E)])
    return got, ref


def _grad(cls, mesh, device, conv_kw):
    """Ordinary autograd through the ``vmap``-ed forward vs a per-member loop."""
    base, params, buffers, ref_models, x_e = _members_and_ref(
        cls, mesh, device, conv_kw
    )
    y = vmap(lambda p, b, x: functional_call(base, (p, b), (x,)))(
        params, buffers, x_e
    )
    y.float().pow(2).sum().backward()
    got = {k: params[k].grad.clone() for k in params}
    # ``functional_call`` does not fire the parameters' post-accumulate
    # grad-sync hooks, so replay the spatial-group SUM the reference modules got.
    for g in base._spatial_sync_groups:
        for t in got.values():
            dist.all_reduce(t, group=g)

    ref_list = []
    for i in range(E):
        ref_models[i].zero_grad(set_to_none=True)
        ref_models[i](x_e[i].contiguous()).float().pow(2).sum().backward()
        ref_list.append(
            {n: p.grad for n, p in ref_models[i].named_parameters()}
        )
    ref = {k: torch.stack([r[k] for r in ref_list]) for k in got}
    return got, ref


_CASES = {
    "no_halo": dict(stride=2, padding=0),  # stride == kernel_size
    "halo": dict(kernel_size=3, stride=1, padding=1),  # stride != kernel_size
}
_MODULES = {
    "conv2d": convolution.DistributedConv2D,
    "conv2d_transpose": convolution.DistributedConv2DTranspose,
}


@pytest.mark.dist(4)
@pytest.mark.parametrize("regime", list(_CASES))
@pytest.mark.parametrize("module", list(_MODULES))
def test_ensemble_vmap_forward(
    device: torch.device, process_mesh_dict_2x2: dict, module: str, regime: str
) -> None:
    got, ref = _forward(
        _MODULES[module], process_mesh_dict_2x2, device, _CASES[regime]
    )
    assert_close(
        f"{module} {regime} ensemble forward",
        got,
        ref,
        **_FWD_TOL,
        device=device,
        group=process_mesh_dict_2x2["model"],
    )


@pytest.mark.dist(4)
@pytest.mark.parametrize("regime", list(_CASES))
@pytest.mark.parametrize("module", list(_MODULES))
def test_ensemble_vmap_grad(
    device: torch.device, process_mesh_dict_2x2: dict, module: str, regime: str
) -> None:
    got, ref = _grad(
        _MODULES[module], process_mesh_dict_2x2, device, _CASES[regime]
    )
    for k in got:
        assert_close(
            f"{module} {regime} ensemble grad[{k}]",
            got[k],
            ref[k],
            **_GRAD_TOL,
            device=device,
            group=process_mesh_dict_2x2["model"],
        )
