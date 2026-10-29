import math
from collections.abc import Mapping, Sequence

import pytest
import torch
import torch.distributed as dist

from jigsaw.mesh import ProcessMesh
from jigsaw.mesh import make_process_mesh as _build_process_mesh

# Nothing frees a mesh's communicators, so every test mesh comes from this cache.
# Bound at import since test_examples monkeypatches ``jigsaw.make_process_mesh``.
_MESHES: dict[tuple, ProcessMesh] = {}

#: Leading axes with these names are data-parallel replicas in the ``mesh`` fixture.
_REPLICA_AXES = ("dp",)


def _get_mesh(
    device_type: str,
    axes: Mapping[str, int],
    model_axes: Sequence[str] | None = None,
    model_key: str = "model",
) -> ProcessMesh:
    """Return the cached mesh for ``axes``, building it on first use.

    A ``-1`` size is resolved first, so it shares the mesh with the explicit sizes.
    Each call returns a new dict over the cached groups.
    """
    world = dist.get_world_size()
    known = math.prod(s for s in axes.values() if s != -1)
    sizes = {k: world // known if s == -1 and known else s for k, s in axes.items()}
    model = tuple(sizes) if model_axes is None else tuple(model_axes)
    key = (device_type, tuple(sizes.items()), model, model_key)
    if key not in _MESHES:
        _MESHES[key] = _build_process_mesh(
            sizes,
            device_type=device_type,
            model_parallel_axes=model,
            model_key=model_key,
        )
    cached = _MESHES[key]
    return ProcessMesh(cached, cached.device_mesh)


def _requested_mesh_axes(item: pytest.Item) -> dict[str, int] | None:
    """Return the axes ``item`` asks the ``mesh`` fixture for, or ``None``.

    An indirect ``mesh`` parameter wins over the marker, so a module-level marker
    can set a default.

    Raises
    ------
    TypeError
        If the marker has positional arguments.
    ValueError
        If an axis size is not a positive integer.
    """
    callspec = getattr(item, "callspec", None)
    if callspec is not None and "mesh" in callspec.params:
        axes = dict(callspec.params["mesh"])
    else:
        marker = item.get_closest_marker("mesh")
        if marker is None:
            return None
        if marker.args:
            raise TypeError(
                "mesh marker takes axis sizes as keywords, e.g. mesh(row=2, col=2)"
            )
        axes = dict(marker.kwargs)
    bad = {k: s for k, s in axes.items() if not isinstance(s, int) or s < 1}
    if not axes or bad:
        raise ValueError(f"mesh axes must be positive integers, got {axes}")
    return axes


@pytest.fixture
def mesh(request: pytest.FixtureRequest, device: torch.device) -> ProcessMesh:
    """Return the cached mesh from a ``mesh`` marker or indirect parameter.

    Leading ``dp`` axes are replicas; the ``model`` group spans the others.
    """
    axes = _requested_mesh_axes(request.node)
    if axes is None:
        pytest.fail(
            "the mesh fixture needs a mesh marker or an indirect mesh parameter",
            pytrace=False,
        )
    names = list(axes)
    n_replica = next(
        (i for i, name in enumerate(names) if name not in _REPLICA_AXES), len(names)
    )
    return _get_mesh(device.type, axes, names[n_replica:])


def make_process_mesh(
    device_type: str = "cuda",
    mesh_shape: Sequence[int] = (2, 2),
    mesh_dim_names: Sequence[str] = ("row", "col"),
    model_name: str = "model",
) -> dict:
    """Return the model axes and model group of a cached mesh.

    The last two axes are model-parallel; leading replica axes such as ``dp``
    stay out of the dict.
    """
    axes = dict(zip(mesh_dim_names, mesh_shape, strict=True))
    model_axes = mesh_dim_names[-2:]
    mesh = _get_mesh(device_type, axes, model_axes, model_name)
    return {key: mesh[key] for key in (*model_axes, model_name)}


def make_multi_spatial_mesh(
    device_type: str,
    spatial_names: list[str],
    spatial_sizes: list[int],
    channel_size: int,
    channel_name: str = "col",
    model_name: str = "model",
) -> dict:
    """Return a mesh that splits the domain along *several* spatial axes.

    The returned dict has one group per entry of ``spatial_names`` (aligned with
    the convolution's spatial axes, e.g. ``["sh", "sw"]`` or
    ``["sd", "sh", "sw"]``), a channel group under ``channel_name`` and a
    ``model_name`` group spanning every rank (spatial x channel).
    """
    axes = {
        **dict(zip(spatial_names, spatial_sizes, strict=True)),
        channel_name: channel_size,
    }
    return _get_mesh(device_type, axes, model_key=model_name)


@pytest.fixture(scope="session")
def process_mesh_dict_2x2(device: torch.Tensor) -> dict:
    """Return process mesh."""
    return make_process_mesh(
        device_type=device.type, mesh_shape=[2, 2], mesh_dim_names=["row", "col"]
    )


@pytest.fixture(scope="session")
def spatial_mesh_2x2_hw(device: torch.Tensor) -> dict:
    """2-way height x 2-way width split, no channel parallelism (world 4)."""
    return make_multi_spatial_mesh(device.type, ["sh", "sw"], [2, 2], 1)


@pytest.fixture(scope="session")
def spatial_mesh_3x2_hw(device: torch.Tensor) -> dict:
    """3-way height x 2-way width split, no channel parallelism (world 6)."""
    return make_multi_spatial_mesh(device.type, ["sh", "sw"], [3, 2], 1)


@pytest.fixture(scope="session")
def spatial_mesh_1x4_hw(device: torch.Tensor) -> dict:
    """width-only 4-way split, height not split, no channel parallelism (world 4)."""
    return make_multi_spatial_mesh(device.type, ["sh", "sw"], [1, 4], 1)


@pytest.fixture(scope="session")
def spatial_mesh_4x1_hw(device: torch.Tensor) -> dict:
    """height-only 4-way split, width not split, no channel parallelism (world 4)."""
    return make_multi_spatial_mesh(device.type, ["sh", "sw"], [4, 1], 1)


@pytest.fixture(scope="session")
def spatial_mesh_5x2_hw(device: torch.Tensor) -> dict:
    """5-way height x 2-way width split, no channel parallelism (world 10)."""
    return make_multi_spatial_mesh(device.type, ["sh", "sw"], [5, 2], 1)


@pytest.fixture(scope="session")
def spatial_mesh_2x1_hw(device: torch.Tensor) -> dict:
    """height-only 2-way split, width not split, no channel parallelism (world 2)."""
    return make_multi_spatial_mesh(device.type, ["sh", "sw"], [2, 1], 1)


@pytest.fixture(scope="session")
def spatial_mesh_2x1_hw_x2c(device: torch.Tensor) -> dict:
    """2-way height, width not split, 2-way channel (world 4)."""
    return make_multi_spatial_mesh(device.type, ["sh", "sw"], [2, 1], 2)


@pytest.fixture(scope="session")
def spatial_mesh_2x2_hw_x2c(device: torch.Tensor) -> dict:
    """2-way height x 2-way width split, 2-way channel parallelism (world 8)."""
    return make_multi_spatial_mesh(device.type, ["sh", "sw"], [2, 2], 2)


@pytest.fixture(scope="session")
def spatial_mesh_2x1x1_dhw(device: torch.Tensor) -> dict:
    """conv3d: depth-only 2-way split, height and width not split (world 2)."""
    return make_multi_spatial_mesh(device.type, ["sd", "sh", "sw"], [2, 1, 1], 1)


@pytest.fixture(scope="session")
def spatial_mesh_2x2x1_dhw(device: torch.Tensor) -> dict:
    """conv3d: 2-way depth x 2-way height, width not split (world 4)."""
    return make_multi_spatial_mesh(device.type, ["sd", "sh", "sw"], [2, 2, 1], 1)


@pytest.fixture(scope="session")
def spatial_mesh_2x1x2_dhw(device: torch.Tensor) -> dict:
    """conv3d: 2-way depth x 2-way width, height not split (world 4)."""
    return make_multi_spatial_mesh(device.type, ["sd", "sh", "sw"], [2, 1, 2], 1)


@pytest.fixture(scope="session")
def spatial_mesh_2x2x2_dhw(device: torch.Tensor) -> dict:
    """conv3d: 2-way depth x 2-way height x 2-way width split (world 8)."""
    return make_multi_spatial_mesh(device.type, ["sd", "sh", "sw"], [2, 2, 2], 1)


@pytest.fixture(scope="session")
def spatial_mesh_2x2x1_dhw_x2c(device: torch.Tensor) -> dict:
    """conv3d: 2-way depth x 2-way height split, 2-way channel (world 8)."""
    return make_multi_spatial_mesh(device.type, ["sd", "sh", "sw"], [2, 2, 1], 2)


@pytest.fixture(scope="session")
def spatial_mesh_2x1x3_dhw(device: torch.Tensor) -> dict:
    """conv3d: 2-way depth x 3-way width, height not split (world 6)."""
    return make_multi_spatial_mesh(device.type, ["sd", "sh", "sw"], [2, 1, 3], 1)


@pytest.fixture(scope="session")
def spatial_mesh_2x1x3_dhw_x2c(device: torch.Tensor) -> dict:
    """conv3d: 2-way depth x 3-way width, height not split, 2-way channel (world 12)."""
    return make_multi_spatial_mesh(device.type, ["sd", "sh", "sw"], [2, 1, 3], 2)


@pytest.fixture(scope="session")
def spatial_mesh_2x2x3_dhw(device: torch.Tensor) -> dict:
    """conv3d: 2-way depth x 2-way height x 3-way width split (world 12)."""
    return make_multi_spatial_mesh(device.type, ["sd", "sh", "sw"], [2, 2, 3], 1)


@pytest.fixture(scope="session")
def spatial_mesh_2x2x3_dhw_x2c(device: torch.Tensor) -> dict:
    """conv3d: 2-way depth x 2-way height x 3-way width split, 2-way channel (world 24)."""
    return make_multi_spatial_mesh(device.type, ["sd", "sh", "sw"], [2, 2, 3], 2)


@pytest.fixture(scope="session")
def process_mesh_dict_4x1(device: torch.Tensor) -> dict:
    """Return process mesh."""
    return make_process_mesh(
        device_type=device.type, mesh_shape=[4, 1], mesh_dim_names=["row", "col"]
    )


@pytest.fixture(scope="session")
def process_mesh_dict_1x4(device: torch.Tensor) -> dict:
    """Return process mesh."""
    return make_process_mesh(
        device_type=device.type, mesh_shape=[1, 4], mesh_dim_names=["row", "col"]
    )


@pytest.fixture(scope="session")
def process_mesh_dict_2x3(device: torch.Tensor) -> dict:
    """Return a 2 (spatial) x 3 (channel) process mesh (needs 6 ranks)."""
    return make_process_mesh(
        device_type=device.type, mesh_shape=[2, 3], mesh_dim_names=["row", "col"]
    )


@pytest.fixture(scope="session")
def process_mesh_dict_3x2(device: torch.Tensor) -> dict:
    """Return a 3 (spatial) x 2 (channel) process mesh (needs 6 ranks)."""
    return make_process_mesh(
        device_type=device.type, mesh_shape=[3, 2], mesh_dim_names=["row", "col"]
    )


@pytest.fixture(scope="session")
def process_mesh_dict_2x1x2(device: torch.Tensor) -> dict:
    """Return process mesh."""
    return make_process_mesh(
        device_type=device.type,
        mesh_shape=[2, 1, 2],
        mesh_dim_names=["dp", "row", "col"],
    )


@pytest.fixture(scope="session")
def process_mesh_dict_4x1x1(device: torch.Tensor) -> dict:
    """Return process mesh."""
    return make_process_mesh(
        device_type=device.type,
        mesh_shape=[4, 1, 1],
        mesh_dim_names=["dp", "row", "col"],
    )


@pytest.fixture(scope="session")
def process_mesh_dict_2x2x1(device: torch.Tensor) -> dict:
    """Return process mesh."""
    return make_process_mesh(
        device_type=device.type,
        mesh_shape=[2, 2, 1],
        mesh_dim_names=["dp", "row", "col"],
    )


@pytest.fixture(scope="session")
def process_mesh_dict_4x2(device: torch.Tensor) -> dict:
    """Return process mesh."""
    return make_process_mesh(
        device_type=device.type,
        mesh_shape=[4, 2],
        mesh_dim_names=["row", "col"],
    )


@pytest.fixture(scope="session")
def process_mesh_dict_1x2(device: torch.Tensor) -> dict:
    """1 (spatial) x 2 (channel) mesh: a bare 2-rank model group, no replicas."""
    return make_process_mesh(
        device_type=device.type,
        mesh_shape=[1, 2],
        mesh_dim_names=["row", "col"],
    )


@pytest.fixture(scope="session")
def process_mesh_dict_1x1(device: torch.Tensor) -> dict:
    """Return process mesh."""
    return make_process_mesh(
        device_type=device.type,
        mesh_shape=[1, 1],
        mesh_dim_names=["row", "col"],
    )


@pytest.fixture(scope="session")
def process_mesh_dict_2x4(device: torch.Tensor) -> dict:
    """Return process mesh."""
    return make_process_mesh(
        device_type=device.type,
        mesh_shape=[2, 4],
        mesh_dim_names=["inter", "intra"],
    )


@pytest.fixture(scope="session")
def process_mesh_dict_2x4_rowcol(device: torch.Tensor) -> dict:
    """Return a 2 (spatial) x 4 (channel) process mesh keyed as row/col."""
    return make_process_mesh(
        device_type=device.type,
        mesh_shape=[2, 4],
        mesh_dim_names=["row", "col"],
    )
