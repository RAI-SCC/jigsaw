import gc
import math
import os
import subprocess
import sys
import warnings

import pytest
import torch
import torch.distributed as dist


def pytest_configure(config):
    """Configure rank 0 to be the only printing rank."""
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    if rank != 0:
        config.option.verbose = -1
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    config.addinivalue_line("markers", "no_dist_env: skip the dist_env autouse fixture")
    config.addinivalue_line("markers", "gpu_required: test requires GPU")
    config.addinivalue_line("markers", "distributed_required: test requires distributed environment")


@pytest.fixture(autouse=True, scope="session")
def dist_env(request: pytest.FixtureRequest):
    """Initialize the distributed process group once for the entire test session."""
    if all(item.get_closest_marker("no_dist_env") for item in request.session.items):
        yield
        return

    rank = int(os.getenv("SLURM_PROCID"))
    world_size = int(os.getenv("SLURM_NTASKS"))
    local_id = int(os.getenv("SLURM_LOCALID", rank))

    device_id = local_id if torch.cuda.device_count() != 1 else 0
    device = torch.device("cuda", device_id)
    torch.cuda.set_device(device)

    os.environ["MASTER_PORT"] = "12342"

    master_addr = (
        subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        .decode()
        .split("\n")[0]
        .strip()
    )

    os.environ["MASTER_ADDR"] = master_addr

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
        init_method="env://",
    )
    yield
    dist.destroy_process_group()


@pytest.fixture
def device() -> torch.device:
    """Return device."""
    local_id = int(os.getenv("SLURM_LOCALID", 0))
    device_id = local_id if torch.cuda.device_count() != 1 else 0
    return torch.device("cuda", device_id)


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


@pytest.fixture(autouse=True)
def cuda_cleanup():
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def pytest_collection_modifyitems(config, items):
    """Skip GPU/distributed tests based on environment."""
    has_gpu = torch.cuda.is_available()
    is_distributed = os.environ.get("SLURM_PROCID") is not None or os.environ.get("RANK") is not None

    for item in items:
        if item.get_closest_marker("gpu_required") and not has_gpu:
            item.add_marker(pytest.mark.skip(reason="GPU not available"))
        if item.get_closest_marker("distributed_required") and not is_distributed:
            item.add_marker(pytest.mark.skip(reason="Not running in distributed environment"))
