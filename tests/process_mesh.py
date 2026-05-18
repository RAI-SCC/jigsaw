import conftest
import pytest 

@pytest.fixture(scope="session")
def process_mesh_2x2():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 2], mesh_dim_names=["row", "col"]
    )
    
@pytest.fixture(scope="session")
def process_mesh_4x1():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[4, 1], mesh_dim_names=["row", "col"]
    )


@pytest.fixture(scope="session")
def process_mesh_1x4():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[1, 4], mesh_dim_names=["row", "col"]
    )


@pytest.fixture(scope="session")
def process_mesh_2x1x2():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 1, 2], mesh_dim_names=["dp", "row", "col"]
    )


@pytest.fixture(scope="session")
def process_mesh_4x1x1():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[4, 1, 1], mesh_dim_names=["dp", "row", "col"]
    )


@pytest.fixture(scope="session")
def process_mesh_2x2x1():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 2, 1], mesh_dim_names=["dp", "row", "col"]
    )

