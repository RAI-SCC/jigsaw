"""``torch.vmap`` batching rule and parameter count of the activation modules.

Every rank feeds identical data through the module over the world group, so the sharded
result is consistent across ranks at any world size; the vmapped result must equal the
per-sample loop through the same module.
"""

import pytest
import torch
import torch.distributed as dist

from jigsaw.modules.activation import _activation as act
from tests.parity import assert_close

E, B, S, C = 3, 2, 4, 6


@pytest.fixture
def batched(device):
    g = torch.Generator().manual_seed(0)
    return torch.randn(E, B, S, C, generator=g, dtype=torch.double).to(device)


def _loop(module, x):
    return torch.stack([module(x[i]) for i in range(x.shape[0])])


@pytest.mark.parametrize("dim", [-1, 1])
def test_softmax_vmap_matches_loop(batched, dim):
    module = act.DistributedSoftmax(dist.group.WORLD, dim=dim)
    assert module.get_parameter_count() == 0
    assert torch.allclose(torch.vmap(module)(batched), _loop(module, batched))


@pytest.mark.parametrize("dim", [-1, 0])
def test_softmax_vmap_batch_dim_not_leading(batched, device, dim):
    """A batch dimension away from the front is moved there before the softmax."""
    group = dist.group.WORLD
    module = act.DistributedSoftmax(group, dim=dim)
    out = torch.vmap(module, in_dims=1)(batched.movedim(0, 1))
    assert_close(
        "vmap in_dims=1", out, _loop(module, batched), 1e-12, 1e-12, device, group
    )
