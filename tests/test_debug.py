"""Unit tests for the non-distributed helpers in :mod:`jigsaw.debug`."""

import torch

from jigsaw import debug


def test_count_parameters_skips_frozen():
    model = torch.nn.Linear(3, 2)
    assert debug.count_parameters(model) == 3 * 2 + 2
    model.bias.requires_grad_(False)
    assert debug.count_parameters(model) == 3 * 2
