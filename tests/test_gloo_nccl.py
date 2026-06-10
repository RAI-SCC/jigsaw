import math
import os
import subprocess

import pytest
import torch
import torch.distributed as dist

pytestmark = [pytest.mark.gpu_required, pytest.mark.distributed_required]


@pytest.fixture(scope="session")
def dist_gloo_env():
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
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
        init_method="env://",
    )
    yield
    dist.destroy_process_group()


def make_process_mesh(
    device_type="nccl", mesh_shape=[4, 2], mesh_dim_names=["row", "col"]
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


@pytest.mark.no_dist_env
def test_backend_init(dist_gloo_env):
    _ = make_process_mesh("nccl", [4, 2])
    return


@pytest.mark.no_dist_env
def test_cpu_allreduce(dist_gloo_env, device):
    _ = make_process_mesh("nccl", [4, 2])
    x = torch.randn(40, 4, device=device)
    x_cpu = x.to("cpu")

    # all-reduce
    dist.all_reduce(x)
    dist.all_reduce(x_cpu)
    x_cpu = x_cpu.to(device)
    assert torch.allclose(x, x_cpu)
