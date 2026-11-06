"""``reshard_parameters`` keeps each parameter's ``requires_grad``.

The reshard replaces the weight and bias by new parameters, which would
otherwise default to ``requires_grad=True`` and unfreeze a frozen one.
"""

import pytest
import torch

import jigsaw.convert as convert
from tests.parity import all_ranks_ok
from tests.process_mesh import make_process_mesh

#: (from mesh, to mesh shape, to mesh axis names), one pair per rank count.
_PAIRS = [
    pytest.param(dict(row=1, col=1), [1, 1], ["row", "col"], id="1x1-to-1x1"),
    pytest.param(dict(row=2, col=2), [4, 1, 1], ["dp", "row", "col"], id="2x2-to-dp4"),
    pytest.param(dict(row=2, col=2), [1, 4], ["row", "col"], id="2x2-to-1x4"),
    pytest.param(dict(row=4, col=2), [2, 4], ["row", "col"], id="4x2-to-2x4"),
]

#: Every layer with a ``reshard_parameters``; all dims divide 1, 2 and 4.
_LAYERS = {
    "linear": lambda d: torch.nn.Linear(24, 16, device=d),
    "conv2d": lambda d: torch.nn.Conv2d(24, 16, 3, device=d),
    "conv_transpose2d": lambda d: torch.nn.ConvTranspose2d(24, 16, 3, device=d),
    "layer_norm": lambda d: torch.nn.LayerNorm(24, device=d),
    "group_norm": lambda d: torch.nn.GroupNorm(4, 24, device=d),
}


@pytest.mark.parametrize("mesh,to_shape,to_names", _PAIRS, indirect=["mesh"])
@pytest.mark.parametrize("layer", _LAYERS)
@pytest.mark.parametrize("frozen", ["weight", "bias"])
def test_reshard_keeps_requires_grad(
    mesh: dict,
    device: torch.device,
    to_shape: list[int],
    to_names: list[str],
    layer: str,
    frozen: str,
) -> None:
    """Only the parameter frozen before the reshard is frozen after it."""
    to_mesh = make_process_mesh(device.type, to_shape, to_names)
    torch.manual_seed(0)
    module = convert.convert_to_jigsaw(_LAYERS[layer](device), mesh, device)
    getattr(module, frozen).requires_grad_(False)

    module.reshard_parameters(mesh, to_mesh)

    params = dict(module.named_parameters())
    ok = sorted(params) == ["bias", "weight"] and all(
        p.requires_grad == (name != frozen) for name, p in params.items()
    )
    assert all_ranks_ok(ok, device, to_mesh["model"])
