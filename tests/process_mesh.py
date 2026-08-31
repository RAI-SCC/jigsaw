import math

import pytest
import torch
import torch.distributed as dist


def make_process_mesh(
    device_type: str = "cuda",
    mesh_shape: list[int] = [2, 2],
    mesh_dim_names: list[str] = ["row", "col"],
    model_name: str = "model",
) -> dict:
    """Return process mesh with rows, columns, and model group."""
    assert len(mesh_shape) == len(mesh_dim_names)

    world_size = dist.get_world_size(group=None)
    if -1 in mesh_shape:
        index = mesh_shape.index(-1)
        mesh_shape[index] = world_size // -torch.prod(torch.tensor(mesh_shape)).item()

    # ``init_device_mesh`` happily builds a mesh that only spans the first
    # ``prod(mesh_shape)`` ranks when the world is larger; the ranks left out
    # then blow up in ``get_group`` while the rest block forever in the first
    # collective. Fail loudly and identically on every rank instead.
    mesh_numel = math.prod(mesh_shape)
    assert mesh_numel == world_size, (
        f"mesh_shape {mesh_shape} needs exactly {mesh_numel} ranks but the world "
        f"has {world_size}; launch with -np {mesh_numel} or mark the test "
        f"@pytest.mark.dist({mesh_numel})."
    )

    row_col_mesh = dist.device_mesh.init_device_mesh(
        device_type=device_type,
        mesh_shape=tuple(mesh_shape),
        mesh_dim_names=tuple(mesh_dim_names),
    )
    if len(mesh_shape) == 2:
        model_mesh = dist.device_mesh.init_device_mesh(
            device_type=device_type,
            mesh_shape=tuple([mesh_shape[-1] * mesh_shape[-2]]),
            mesh_dim_names=tuple([model_name]),
        )
    else:  #
        model_mesh = dist.device_mesh.init_device_mesh(
            device_type=device_type,
            mesh_shape=tuple(
                [math.prod(mesh_shape[:-2]), mesh_shape[-2] * mesh_shape[-1]]
            ),
            mesh_dim_names=tuple(["x", "model"]),
        )

    mesh = {
        f"{mesh_dim_names[-2]}": row_col_mesh.get_group(mesh_dim_names[-2]),
        f"{mesh_dim_names[-1]}": row_col_mesh.get_group(mesh_dim_names[-1]),
        f"{model_name}": model_mesh.get_group(model_name),
    }

    return mesh


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
    shape = tuple([*spatial_sizes, channel_size])
    names = tuple([*spatial_names, channel_name])
    nd_mesh = dist.device_mesh.init_device_mesh(
        device_type=device_type, mesh_shape=shape, mesh_dim_names=names
    )
    model_mesh = dist.device_mesh.init_device_mesh(
        device_type=device_type,
        mesh_shape=(math.prod(shape),),
        mesh_dim_names=(model_name,),
    )
    mesh = {name: nd_mesh.get_group(name) for name in names}
    mesh[model_name] = model_mesh.get_group(model_name)
    return mesh


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
