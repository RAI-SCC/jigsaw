"""Build a jigsaw process-mesh dict from a ``RxC`` spec.

Mirrors ``tests/process_mesh.py::make_process_mesh`` without the pytest
dependency: returns ``{row, col, model}`` process groups, where ``model`` spans
every rank of the ``row x col`` grid (the group the parameter-free matmul and the
converters expect).
"""

from __future__ import annotations

import math

import torch.distributed as dist

__all__ = ["parse_mesh", "make_mesh"]


def parse_mesh(spec: str) -> list[int]:
    """``"2x2"`` -> ``[2, 2]``. Only 2-D (``row x col``) specs are supported."""
    parts = spec.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"bad mesh spec {spec!r}; want e.g. '2x2' or '1x4'")
    return [int(p) for p in parts]


def make_mesh(
    device_type: str,
    mesh_shape: list[int],
    names: tuple[str, str] = ("row", "col"),
) -> dict:
    """Return ``{names[0], names[1], "model"}`` -> ``ProcessGroup``.

    ``mesh_shape`` must multiply to the current world size.
    """
    if len(mesh_shape) != 2 or len(names) != 2:
        raise ValueError("only 2-D row x col meshes are supported")
    world = dist.get_world_size()
    if math.prod(mesh_shape) != world:
        raise RuntimeError(
            f"mesh {mesh_shape} needs {math.prod(mesh_shape)} ranks but the world "
            f"has {world}; launch with that many tasks or change --mesh"
        )
    grid = dist.device_mesh.init_device_mesh(
        device_type=device_type,
        mesh_shape=tuple(mesh_shape),
        mesh_dim_names=tuple(names),
    )
    model = dist.device_mesh.init_device_mesh(
        device_type=device_type,
        mesh_shape=(mesh_shape[0] * mesh_shape[1],),
        mesh_dim_names=("model",),
    )
    return {
        names[0]: grid.get_group(names[0]),
        names[1]: grid.get_group(names[1]),
        "model": model.get_group("model"),
    }
