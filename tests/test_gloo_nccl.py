"""Test the hybrid ``cpu:gloo,cuda:nccl`` backend being initialized.

Marked ``no_dist_env`` so the shared session group is *not* set up: this file
brings up its own hybrid-backend group. That means it only works when run on
its own (``pytest tests/test_gloo_nccl.py`` under ``srun -n8``); inside the full
suite another test initialises the default group first and the fixture skips.
"""

import math

import pytest
import torch
import torch.distributed as dist

from jigsaw.launch import resolve_launch_env

pytestmark = [
    pytest.mark.gpu_required,
    pytest.mark.dist_required,
    pytest.mark.hybrid_dist_env,
    pytest.mark.dist(8),
]


@pytest.fixture(scope="module")
def dist_gloo_env():
    """Bring up a ``cpu:gloo,cuda:nccl`` process group for this module."""
    if dist.is_initialized():
        pytest.skip("a default process group already exists; run this file on its own")

    try:
        env = resolve_launch_env(strict=True)  # writes MASTER_ADDR / MASTER_PORT
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if env.master_addr in ("", "localhost") and env.world_size > 1:
        pytest.skip("could not resolve MASTER_ADDR (no scontrol / nodelist)")

    device_id = env.local_rank if torch.cuda.device_count() != 1 else 0
    device = torch.device("cuda", device_id)
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=env.rank,
        world_size=env.world_size,
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


class TestGlooNcclBackend:
    @pytest.mark.no_dist_env
    def test_backend_init(self, dist_gloo_env):
        """Verify distributed backend initialization.

        Parameters
        ----------
        dist_gloo_env : Any
            Distributed environment configuration for the Gloo backend.

        """
        _ = make_process_mesh("nccl", [4, 2])
        return

    @pytest.mark.no_dist_env
    def test_cpu_allreduce(self, dist_gloo_env, device):
        """Verify CPU all-reduce communication.

        Parameters
        ----------
        dist_gloo_env : Any
            Distributed environment configuration for the Gloo backend.
        device : Any
            Device on which tensors and communication are placed.

        """
        _ = make_process_mesh("nccl", [4, 2])
        x = torch.randn(40, 4, device=device)
        x_cpu = x.to("cpu")

        # all-reduce
        dist.all_reduce(x)
        dist.all_reduce(x_cpu)
        x_cpu = x_cpu.to(device)
        assert torch.allclose(x, x_cpu)
