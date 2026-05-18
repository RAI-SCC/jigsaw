import conftest
import pytest 

@pytest.fixture(scope="session")
def process_mesh2x2():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[2, 2], mesh_dim_names=["row", "col"]
    )