"""``torch.vmap`` batching rules and parameter counts of the normalization modules.

Every rank feeds identical data through a module over the world group, so the sharded
result is consistent across ranks at any world size; the vmapped result must equal the
per-sample loop through the same module.
"""

import pytest
import torch
import torch.distributed as dist
from torch.func import functional_call, stack_module_state

from jigsaw.modules import normalization as norm
from tests.parity import assert_close

E, B, S, C = 3, 2, 4, 6


@pytest.fixture
def batched(device):
    g = torch.Generator().manual_seed(0)
    return torch.randn(E, B, S, C, generator=g, dtype=torch.double).to(device)


def _loop(module, x):
    return torch.stack([module(x[i]) for i in range(x.shape[0])])


def test_noparam_layernorm_vmap_matches_loop(batched, device):
    module = norm.DistributedLayerNorm(
        C * dist.get_world_size(),
        dist.get_rank(),
        device,
        dist.group.WORLD,
        dist.group.WORLD,
        elementwise_affine=False,
    )
    assert module.get_parameter_count() == 0
    assert torch.allclose(torch.vmap(module)(batched), _loop(module, batched))


def test_layernorm_vmap_matches_loop_and_counts_parameters(batched, device):
    module = norm.DistributedLayerNorm(
        C * dist.get_world_size(),
        dist.get_rank(),
        device,
        dist.group.WORLD,
        dist.group.WORLD,
    ).double()
    with torch.no_grad():
        module.weight.copy_(torch.linspace(0.5, 1.5, C))
        module.bias.copy_(torch.linspace(-1.0, 1.0, C))
    assert torch.allclose(torch.vmap(module)(batched), _loop(module, batched))
    # Global: weight and bias over the full normalized_dim, C * world size.
    assert module.get_parameter_count() == 2 * C * dist.get_world_size()


def _layernorm(device, affine: bool = True) -> norm.DistributedLayerNorm:
    return norm.DistributedLayerNorm(
        C * dist.get_world_size(),
        dist.get_rank(),
        device,
        dist.group.WORLD,
        dist.group.WORLD,
        elementwise_affine=affine,
    ).double()


def _close(name: str, got: torch.Tensor, ref: torch.Tensor, device) -> None:
    assert_close(name, got, ref, 1e-10, 1e-10, device, dist.group.WORLD)


@pytest.mark.parametrize("affine", [True, False], ids=["affine", "no-affine"])
def test_layernorm_vmap_non_leading_in_dim(batched, device, affine):
    """The vmapped axis need not be the leading one; the output follows ``out_dims``."""
    module = _layernorm(device, affine)
    if affine:
        with torch.no_grad():
            module.weight.copy_(torch.linspace(0.5, 1.5, C))
            module.bias.copy_(torch.linspace(-1.0, 1.0, C))
    x = batched.transpose(0, 1).contiguous()  # (B, E, S, C), vmapped over E

    got = torch.vmap(module, in_dims=1)(x)
    ref = torch.stack([module(x[:, i]) for i in range(E)])
    _close("output", got, ref, device)


def test_layernorm_vmap_over_member_weights(batched, device):
    """Stacked per-member weight and bias: output and gradients match a loop."""
    g = torch.Generator().manual_seed(1)
    base = _layernorm(device)
    members = [_layernorm(device) for _ in range(E)]
    with torch.no_grad():
        for m in members:
            m.weight.copy_(torch.randn(C, generator=g, dtype=torch.double))
            m.bias.copy_(torch.randn(C, generator=g, dtype=torch.double))
    params, _ = stack_module_state(members)
    ref_params = {k: v.detach().clone().requires_grad_() for k, v in params.items()}
    x = batched.clone().requires_grad_()
    x_ref = batched.clone().requires_grad_()

    got = torch.vmap(lambda p, xi: functional_call(base, p, (xi,)))(params, x)
    ref = torch.stack(
        [
            functional_call(base, {k: v[e] for k, v in ref_params.items()}, (x_ref[e],))
            for e in range(E)
        ]
    )
    _close("output", got, ref, device)

    grad = torch.randn(ref.shape, generator=g, dtype=torch.double).to(device)
    (got * grad).sum().backward()
    (ref * grad).sum().backward()
    _close("grad_x", x.grad, x_ref.grad, device)
    _close("grad_weight", params["weight"].grad, ref_params["weight"].grad, device)
    _close("grad_bias", params["bias"].grad, ref_params["bias"].grad, device)
