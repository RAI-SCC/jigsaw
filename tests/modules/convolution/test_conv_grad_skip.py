"""Backward of the convolutions computes only the gradients autograd needs."""

from typing import Any

import pytest
import torch

import jigsaw.convert as convert
from jigsaw.modules.convolution._conv import distributed as conv_dist
from jigsaw.modules.convolution._conv_transpose import distributed as conv_t_dist
from tests.parity import REPLICATED, ShardSpec, check_parity

pytestmark = [pytest.mark.dist_required, pytest.mark.dist(4)]

# (torch class, weight axis split over col, kernel, stride, padding, module of
# its input-gradient channel reduce)
_LAYERS = {
    "conv": (torch.nn.Conv2d, 0, 4, 4, 0, conv_dist),
    "conv_halo": (torch.nn.Conv2d, 0, 3, 2, 1, conv_dist),
    "conv_transpose": (torch.nn.ConvTranspose2d, 1, 2, 2, 0, conv_t_dist),
    "conv_transpose_halo": (torch.nn.ConvTranspose2d, 1, 4, 2, 1, conv_t_dist),
}

# (input requires grad, frozen parameters)
_REQUIRES = {
    "x_only": (True, ("weight", "bias")),
    "weight_only": (False, ("bias",)),
    "weight_and_bias": (False, ()),
    "bias_only": (False, ("weight",)),
}


@pytest.mark.parametrize("requires", _REQUIRES.values(), ids=_REQUIRES.keys())
@pytest.mark.parametrize("layer", _LAYERS.values(), ids=_LAYERS.keys())
def test_conv_backward_skips_unneeded_grads(
    device: torch.device,
    process_mesh_dict_2x2: dict,
    monkeypatch: pytest.MonkeyPatch,
    layer: tuple,
    requires: tuple[bool, tuple[str, ...]],
) -> None:
    """Needed gradients match torch; the input-gradient reduce runs only if needed."""
    torch_cls, w_axis, kernel, stride, padding, dist_module = layer
    input_requires_grad, frozen = requires
    mesh = process_mesh_dict_2x2

    reduces = 0
    reduce_fn = dist_module._channel_reduce_grad_input

    def counted(*args: Any) -> Any:
        nonlocal reduces
        reduces += 1
        return reduce_fn(*args)

    monkeypatch.setattr(dist_module, "_channel_reduce_grad_input", counted)

    torch.manual_seed(0)  # identical serial reference on every rank
    serial = torch_cls(
        4, 8, kernel_size=kernel, stride=stride, padding=padding, device=device
    )
    data_spec = ShardSpec(col=1, row=3)
    check_parity(
        serial_module=serial,
        build_distributed=lambda m: convert.convert_to_jigsaw(m, mesh, device),
        input_shape=(2, 4, 8, 16),
        input_spec=data_spec,
        output_spec=data_spec,
        param_rules={
            "weight": ShardSpec(col=w_axis, row=REPLICATED),
            "bias": ShardSpec(col=0, row=REPLICATED),
        },
        process_mesh_dict=mesh,
        device=device,
        atol=1e-4,
        rtol=1e-4,
        input_requires_grad=input_requires_grad,
        frozen=frozen,
    )
    assert reduces == int(input_requires_grad)
