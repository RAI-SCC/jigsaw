import pytest

from . import conftest


@pytest.fixture(scope="session")
def process_mesh_dict_2x2() -> dict:
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 2], mesh_dim_names=["row", "col"]
    )


@pytest.fixture(scope="session")
def process_mesh_dict_4x1() -> dict:
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[4, 1], mesh_dim_names=["row", "col"]
    )


@pytest.fixture(scope="session")
def process_mesh_dict_1x4() -> dict:
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[1, 4], mesh_dim_names=["row", "col"]
    )


@pytest.fixture(scope="session")
def process_mesh_dict_2x1x2() -> dict:
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 1, 2], mesh_dim_names=["dp", "row", "col"]
    )


@pytest.fixture(scope="session")
def process_mesh_dict_4x1x1() -> dict:
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[4, 1, 1], mesh_dim_names=["dp", "row", "col"]
    )


@pytest.fixture(scope="session")
def process_mesh_dict_2x2x1() -> dict:
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 2, 1], mesh_dim_names=["dp", "row", "col"]
    )
