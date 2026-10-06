"""``torch.vmap`` batching rules and parameter counts of the normalization modules.

Every rank feeds identical data through a module over the world group, so the sharded
result is consistent across ranks at any world size; the vmapped result must equal the
per-sample loop through the same module.
"""

import pytest
import torch
import torch.distributed as dist

from jigsaw.modules import normalization as norm

E, B, S, C = 3, 2, 4, 6


@pytest.fixture
def batched(device):
    g = torch.Generator().manual_seed(0)
    return torch.randn(E, B, S, C, generator=g, dtype=torch.double).to(device)


def _loop(module, x):
    return torch.stack([module(x[i]) for i in range(x.shape[0])])


def test_noparam_layernorm_vmap_matches_loop(batched, device):
    module = norm.DistributedLayerNorm(
        C,
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
        C, dist.get_rank(), device, dist.group.WORLD, dist.group.WORLD
    ).double()
    with torch.no_grad():
        module.weight.copy_(torch.linspace(0.5, 1.5, C))
        module.bias.copy_(torch.linspace(-1.0, 1.0, C))
    assert torch.allclose(torch.vmap(module)(batched), _loop(module, batched))
    assert module.get_parameter_count() == 2 * C // dist.get_world_size()
