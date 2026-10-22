from collections.abc import Sequence

import pytest
import torch

import jigsaw

# These fixtures are thin wrappers over ``jigsaw.make_process_mesh``; they keep
# the historical positional signatures and trim the result to the keys the tests
# have always seen (leading replica axes such as ``dp`` stay out of the dict).


def make_process_mesh(
    device_type: str = "cuda",
    mesh_shape: Sequence[int] = (2, 2),
    mesh_dim_names: Sequence[str] = ("row", "col"),
    model_name: str = "model",
) -> dict:
    """Return process mesh with rows, columns, and model group."""
    axes = dict(zip(mesh_dim_names, mesh_shape, strict=True))
    model_axes = mesh_dim_names[-2:]
    mesh = jigsaw.make_process_mesh(
        axes,
        device_type=device_type,
        model_parallel_axes=model_axes,
        model_key=model_name,
    )
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
    return jigsaw.make_process_mesh(axes, device_type=device_type, model_key=model_name)


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
        mesh_dim_names=["internode", "intranode"],
    )


@pytest.fixture(scope="session")
def process_mesh_dict_2x4_rowcol(device: torch.Tensor) -> dict:
    """Return a 2 (spatial) x 4 (channel) process mesh keyed as row/col."""
    return make_process_mesh(
        device_type=device.type,
        mesh_shape=[2, 4],
        mesh_dim_names=["row", "col"],
    )
