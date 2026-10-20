"""Unit tests for the helpers in :mod:`jigsaw.debug`."""

import pytest
import torch
import torch.distributed as dist

import jigsaw
import jigsaw.modules.linear as linear
from jigsaw import debug
from tests.parity import all_ranks_ok


def test_count_parameters_skips_frozen():
    model = torch.nn.Linear(3, 2)
    assert debug.count_parameters(model) == 3 * 2 + 2
    model.bias.requires_grad_(False)
    assert debug.count_parameters(model) == 3 * 2


_MESH_CACHE: dict = {}


def _dp_mesh(device: torch.device) -> dict:
    # Built once per device: nothing frees a mesh's communicators.
    if device.type not in _MESH_CACHE:
        _MESH_CACHE[device.type] = jigsaw.make_process_mesh(
            {"dp": 2, "row": 2, "col": 1},
            model_parallel_axes=("row", "col"),
            device_type=device.type,
        )
    return _MESH_CACHE[device.type]


@pytest.mark.dist(4)
@pytest.mark.parametrize("data_parallel", [False, True])
@pytest.mark.parametrize("bias", [True, False])
def test_count_parameters_with_mesh_is_the_whole_model(
    device: torch.device, process_mesh_dict_2x2: dict, data_parallel: bool, bias: bool
) -> None:
    """Shards are summed, the row-replicated bias and dp copies counted once."""
    mesh = _dp_mesh(device) if data_parallel else process_mesh_dict_2x2
    in_dim, out_dim = 6, 8
    layer = linear.DistributedLinear(
        in_dim,
        out_dim,
        local_rank=dist.get_rank(group=mesh["model"]),
        device=device,
        model_group=mesh["model"],
        col_group=mesh["col"],
        row_group=mesh["row"],
        bias=bias,
    ).to(device)

    total = debug.count_parameters(layer, mesh)
    expected = in_dim * out_dim + (out_dim if bias else 0)
    assert all_ranks_ok(total == expected, device, dist.group.WORLD)
    assert debug.count_parameters(layer) == layer.weight.numel() + (
        layer.bias.numel() if bias else 0
    )
