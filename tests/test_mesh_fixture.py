"""The test suite's ``mesh`` fixture and shared mesh cache in ``tests/process_mesh``."""

import math

import pytest
import torch.distributed as dist

from tests.process_mesh import _get_mesh, _requested_mesh_axes


class _Item:
    """Stands in for a ``pytest.Item`` carrying only a ``mesh`` marker."""

    def __init__(self, mark: pytest.MarkDecorator) -> None:
        self.mark = mark.mark

    def get_closest_marker(self, name: str) -> pytest.Mark | None:
        return self.mark if name == self.mark.name else None


@pytest.mark.mesh(row=2, col=2)
def test_marker_builds_the_mesh(mesh: dict) -> None:
    assert set(mesh) == {"row", "col", "model"}
    assert (mesh["row"].size(), mesh["col"].size(), mesh["model"].size()) == (2, 2, 4)


@pytest.mark.parametrize(
    "mesh",
    [
        dict(row=1, col=1),
        dict(row=4, col=1),
        dict(row=1, col=4),
        dict(dp=2, row=2, col=1),
        dict(dp=4, row=1, col=1),
        dict(sh=2, sw=2, col=1),
        dict(dp=2, row=2, col=2),
        dict(sd=2, sh=2, sw=2, col=1),
    ],
    indirect=True,
)
def test_parameter_builds_the_mesh_and_leading_dp_is_replicated(
    request: pytest.FixtureRequest, mesh: dict
) -> None:
    axes = request.node.callspec.params["mesh"]
    assert set(mesh) == {*axes, "model"}
    assert {name: mesh[name].size() for name in axes} == axes
    assert mesh["model"].size() == math.prod(s for n, s in axes.items() if n != "dp")
    assert math.prod(axes.values()) == dist.get_world_size()


@pytest.mark.mesh(row=2, col=2)
def test_fixtures_share_groups(mesh: dict, process_mesh_dict_2x2: dict) -> None:
    """The ``mesh`` fixture and the named fixtures draw on one cache."""
    assert all(mesh[k] is process_mesh_dict_2x2[k] for k in ("row", "col", "model"))


def test_cache_returns_the_same_groups_in_a_new_dict(device) -> None:
    world = dist.get_world_size()
    first = _get_mesh(device.type, {"row": world, "col": 1})
    second = _get_mesh(device.type, {"row": -1, "col": 1})
    assert first is not second
    assert all(first[k] is second[k] for k in first)
    assert first.device_mesh is second.device_mesh
    second.pop("row")
    assert "row" in _get_mesh(device.type, {"row": world, "col": 1})


def test_marker_rejects_positional_sizes() -> None:
    with pytest.raises(TypeError, match="keywords"):
        _requested_mesh_axes(_Item(pytest.mark.mesh(2, 2)))


@pytest.mark.parametrize("size", [0, -1, 2.0])
def test_marker_rejects_bad_sizes(size: object) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        _requested_mesh_axes(_Item(pytest.mark.mesh(row=size, col=1)))
