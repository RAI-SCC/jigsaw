import math
import os

import pytest
import torch
import torch.distributed as dist


def make_process_mesh(
    device_type: str = "nccl",
    mesh_shape: list[int] = [2, 2],
    mesh_dim_names: list[str] = ["row", "col"],
) -> dict:
    """Return process mesh with rows, columns, and model group."""
    assert len(mesh_shape) == len(mesh_dim_names)

    world_size = dist.get_world_size(group=None)
    if -1 in mesh_shape:
        index = mesh_shape.index(-1)
        mesh_shape[index] = world_size // -torch.prod(torch.tensor(mesh_shape)).item()

    row_col_mesh = dist.device_mesh.init_device_mesh(
        device_type=device_type,
        mesh_shape=tuple(mesh_shape),
        mesh_dim_names=tuple(mesh_dim_names),
    )
    if len(mesh_shape) == 2:
        model_mesh = dist.device_mesh.init_device_mesh(
            device_type=device_type,
            mesh_shape=tuple([mesh_shape[-1] * mesh_shape[-2]]),
            mesh_dim_names=tuple(["model"]),
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
        "row": row_col_mesh.get_group("row"),
        "col": row_col_mesh.get_group("col"),
        "model": model_mesh.get_group("model"),
    }

    return mesh


@pytest.fixture(scope="session")
def process_mesh_dict_2x2(device: torch.Tensor) -> dict:
    """Return process mesh."""
    return make_process_mesh(
        device_type=device.type, mesh_shape=[2, 2], mesh_dim_names=["row", "col"]
    )


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
def process_mesh_dict_env(device: torch.Tensor) -> dict:
    """
    Return process mesh whose shape is taken from the GRID_ROW / GRID_COL env variables.

    GRID_ROW is the number of ranks in the row (spatial / sequence) direction and
    GRID_COL the number of ranks in the column (channel) direction, matching the
    row / col naming used throughout jigsaw. Both default to 1, and their product
    must equal the world size.

    This is the torch-harmonics distributed-test convention with their h / w renamed
    to row / col. The rank layout is identical: `init_device_mesh` lays the world out
    as `arange(world_size).reshape(GRID_ROW, GRID_COL)`, so
    `row.rank() == world_rank // GRID_COL` and `col.rank() == world_rank % GRID_COL`.
    """
    grid_row = int(os.getenv("GRID_ROW", 1))
    grid_col = int(os.getenv("GRID_COL", 1))
    world_size = dist.get_world_size()
    assert grid_row * grid_col == world_size, (
        f"GRID_ROW ({grid_row}) * GRID_COL ({grid_col}) = {grid_row * grid_col} does "
        f"not match the world size ({world_size})"
    )
    return make_process_mesh(
        device_type=device.type,
        mesh_shape=[grid_row, grid_col],
        mesh_dim_names=["row", "col"],
    )
