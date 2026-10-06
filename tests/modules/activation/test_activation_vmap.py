"""``torch.vmap`` batching rule and parameter count of the activation modules.

Every rank feeds identical data through the module over the world group, so the sharded
result is consistent across ranks at any world size; the vmapped result must equal the
per-sample loop through the same module.
"""

import pytest
import torch
import torch.distributed as dist

from jigsaw.modules.activation import activation as act

E, B, S, C = 3, 2, 4, 6


@pytest.fixture
def batched(device):
    g = torch.Generator().manual_seed(0)
    return torch.randn(E, B, S, C, generator=g, dtype=torch.double).to(device)


def _loop(module, x):
    return torch.stack([module(x[i]) for i in range(x.shape[0])])


@pytest.mark.parametrize("dim", [-1, 1])
def test_softmax_vmap_matches_loop(batched, dim):
    module = act.DistributedSoftMax(dist.group.WORLD, dim=dim)
    assert module.get_parameter_count() == 0
    assert torch.allclose(torch.vmap(module)(batched), _loop(module, batched))
