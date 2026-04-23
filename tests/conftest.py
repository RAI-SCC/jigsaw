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


@pytest.fixture(autouse=True, scope="session")
def dist_env():
    """Initialize the distributed process group once for the entire test session."""
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
def device():
    """Return device."""
    local_id = int(os.getenv("SLURM_LOCALID", 0))
    device_id = local_id if torch.cuda.device_count() != 1 else 0
    return torch.device("cuda", device_id)


def make_process_mesh(
    device_type="nccl", mesh_shape=[2, 2], mesh_dim_names=["row", "col"]
):
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
        print(f"mesh shape! {mesh_shape[:-2]}")
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


def gather_results_as_list(x: torch.Tensor, process_mesh: dist.device_mesh = None):
    """Gather tensors split across arbitrary processes."""
    row_group, col_group = process_mesh["row"], process_mesh["col"]
    n_rows = row_group.size()
    n_cols = col_group.size()
    tensor_list = [
        torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for _ in range(n_rows * n_cols)
    ]
    return tensor_list


def gather_results(x: torch.Tensor, process_mesh: dist.device_mesh = None):
    """Gather tensors split across arbitrary processes."""
    row_group, col_group = process_mesh["row"], process_mesh["col"]
    n_rows = row_group.size()
    n_cols = col_group.size()
    tensor_list = [
        torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for _ in range(n_rows * n_cols)
    ]
    torch.distributed.all_gather(tensor_list, x, group=process_mesh["model"])
    batch_dims, h, w = x.shape[:-2], x.shape[-2], x.shape[-1]
    ndim = len(batch_dims)
    perm = list(range(2, 2 + ndim)) + [0, 2 + ndim, 1, 2 + ndim + 1]
    result = torch.stack(tensor_list)
    result = (
        result.reshape(n_rows, n_cols, *x.shape)
        .permute(*perm)
        .reshape(*batch_dims, n_rows * h, n_cols * w)
    )
    return result
