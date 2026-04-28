import math
import os
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


@pytest.fixture(autouse=True, scope="session")
def dist_env():
    """Initialize the distributed process group once for the entire test session."""
    rank = int(os.getenv("SLURM_PROCID", -1))
    world_size = int(os.getenv("SLURM_NTASKS", -1))
    local_id = int(os.getenv("SLURM_LOCALID", rank))

    device_id = local_id if torch.cuda.device_count() != 1 else 0
    device = torch.device("cuda", device_id)
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
        init_method="env://",
    )

    # """Configure rank 0 to be the only printing rank."""
    # rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    # if rank != 0:
    #     # sys.stdout = open(os.devnull, "w")
    #     sys.stderr = open(os.devnull, "w")

    # warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
    # warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    yield
    dist.destroy_process_group()


@pytest.fixture
def device():
    """Return device."""
    local_id = int(os.getenv("SLURM_LOCALID", 0))
    device_id = local_id if torch.cuda.device_count() != 1 else 0
    return torch.device("cuda", device_id)


def make_model_group(mesh):
    """Build the comm group from the last mesh dimension."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < math.prod(mesh):
        pytest.skip(f"Need {math.prod(mesh)} ranks, have {world_size}")

    if len(mesh) == 1:
        return dist.group.WORLD

    last_dim = mesh[-1]
    group_id = rank // last_dim
    num_groups = math.prod(mesh[:-1])
    my_group = None
    for g in range(num_groups):
        ranks_in_group = list(range(g * last_dim, (g + 1) * last_dim))
        pg = dist.new_group(ranks_in_group)
        if g == group_id:
            my_group = pg
    return my_group


def make_process_mesh(shape):
    mesh = dist.device_mesh.init_device_mesh(
        device_type="nccl", mesh_shape=shape, mesh_dim_names=("row", "col")
    )

    return mesh