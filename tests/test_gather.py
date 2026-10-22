"""Tests for the weight-averaging helpers in jigsaw.gather."""

import pytest
import torch
import torch.distributed as dist

from jigsaw.gather import average_weights, average_weights_module


class TestAveraging:
    def test_average_weights_keeps_identical_replicas_unchanged(self, device):
        tensors = [
            torch.full((2,), 3.0, device=device),
            torch.arange(4.0, device=device),
        ]
        expected = [t.clone() for t in tensors]
        average_weights([tensors[0], None, tensors[1]], dist.group.WORLD)
        for got, want in zip(tensors, expected, strict=True):
            assert torch.equal(got, want)

    @pytest.mark.parametrize("bias_only", [False, True])
    def test_average_weights_module(self, device, bias_only):
        torch.manual_seed(0)  # identical replicas on every rank
        model = torch.nn.Linear(3, 2).to(device)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        average_weights_module(model, dist.group.WORLD, bias_only=bias_only)
        for key, value in model.state_dict().items():
            assert torch.allclose(value, before[key])

    @pytest.mark.dist(2)
    def test_average_weights_averages_across_ranks(self, device):
        rank = dist.get_rank()
        weight = torch.full((2,), float(rank), device=device)
        average_weights([weight], dist.group.WORLD)
        assert torch.allclose(weight.cpu(), torch.full((2,), 0.5))

    @pytest.mark.dist(2)
    def test_bias_only_leaves_weights_local(self, device):
        rank = dist.get_rank()
        model = torch.nn.Linear(2, 2).to(device)
        with torch.no_grad():
            model.weight.fill_(float(rank))
            model.bias.fill_(float(rank))
        average_weights_module(model, dist.group.WORLD, bias_only=True)
        assert torch.allclose(model.bias.cpu(), torch.full((2,), 0.5))
        assert torch.all(model.weight == float(rank))
