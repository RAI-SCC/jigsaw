"""Process mesh construction.

A process mesh maps axis names to process groups. The trailing axes are the
model-parallel axes (``row`` and ``col`` by default), leading axes are
data-parallel replicas, and a ``model`` group spans the model-parallel axes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist

from ._core.comm import check_backend

__all__ = ["ProcessMesh", "make_process_mesh"]


class ProcessMesh(dict):
    """A ``dict`` of axis name to process group that also holds the ``DeviceMesh``.

    Indexing works as on any ``process_mesh_dict``; :attr:`device_mesh` is the
    named n-d mesh the axis groups were cut from, so a 1-D slice of it can go to
    APIs that take a ``DeviceMesh``, e.g.
    ``DistributedDataParallel(model, device_mesh=mesh.device_mesh["dp"])``.
    The ``model`` group is not a dimension of it.

    Parameters
    ----------
    groups : Mapping[str, dist.ProcessGroup]
        Axis name to process group.
    device_mesh : torch.distributed.device_mesh.DeviceMesh
        The n-d mesh the groups were cut from.

    Attributes
    ----------
    device_mesh : torch.distributed.device_mesh.DeviceMesh
        The n-d mesh over every axis, in the order they were given.
    """

    device_mesh: dist.device_mesh.DeviceMesh

    def __init__(
        self,
        groups: Mapping[str, dist.ProcessGroup],
        device_mesh: dist.device_mesh.DeviceMesh,
    ) -> None:
        super().__init__(groups)
        self.device_mesh = device_mesh


def _resolve_device_type(device_type: str | None) -> str:
    if device_type is not None:
        return device_type
    # A Gloo default group communicates host tensors, so its mesh is a CPU mesh
    # even on a machine with GPUs.
    if dist.get_backend() == "gloo":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_sizes(axes: Mapping[str, int]) -> dict[str, int]:
    """Copy ``axes`` and infer a single ``-1`` size from the world size."""
    sizes = dict(axes)
    if not sizes:
        raise ValueError("axes must name at least one mesh dimension")

    placeholders = [name for name, size in sizes.items() if size == -1]
    if len(placeholders) > 1:
        raise ValueError(f"at most one axis may be -1, got {placeholders}")

    world_size = dist.get_world_size()
    if placeholders:
        known = math.prod(s for s in sizes.values() if s != -1)
        if known == 0 or world_size % known != 0:
            raise ValueError(
                f"cannot infer -1 axis: world size {world_size} is not divisible "
                f"by the product {known} of the fixed axes"
            )
        sizes[placeholders[0]] = world_size // known

    bad = {name: size for name, size in sizes.items() if size < 1}
    if bad:
        raise ValueError(f"every axis size must be >= 1, got {bad}")
    return sizes


def _resolve_model_axes(
    names: list[str], model_parallel_axes: Sequence[str] | None
) -> list[str]:
    """Validate that the model-parallel axes are the trailing axes."""
    if model_parallel_axes is None:
        return list(names)

    model_axes = list(model_parallel_axes)
    if not model_axes:
        raise ValueError("model_parallel_axes must be non-empty when given")
    unknown = [a for a in model_axes if a not in names]
    if unknown:
        raise ValueError(f"model_parallel_axes {unknown} are not in axes {names}")
    if model_axes != names[len(names) - len(model_axes) :]:
        raise ValueError(
            f"model_parallel_axes must be the trailing axes of {names} in order; "
            f"got {model_axes}"
        )
    return model_axes


def make_process_mesh(
    axes: Mapping[str, int],
    *,
    device_type: str | None = None,
    model_parallel_axes: Sequence[str] | None = None,
    model_key: str = "model",
) -> ProcessMesh:
    """Build a jigsaw ``process_mesh_dict`` from named, ordered mesh axes.

    Parameters
    ----------
    axes : Mapping[str, int]
        Ordered mapping of axis name to size, slowest-varying first. At most one
        size may be ``-1``, which is inferred from the world size.
    device_type : str, optional
        Device type of the mesh. Defaults to ``"cpu"`` if the default process
        group uses Gloo, else ``"cuda"`` if available, else ``"cpu"``.
    model_parallel_axes : Sequence[str], optional
        Trailing axes that form one model replica. Defaults to all axes.
    model_key : str
        Key of the group spanning the model-parallel axes.

    Returns
    -------
    ProcessMesh
        One group per axis plus the ``model_key`` group, in a ``dict`` that
        also carries the n-d ``DeviceMesh`` as ``.device_mesh``.

    Notes
    -----
    Requires an initialized default process group on ``nccl`` (NCCL or RCCL),
    ``gloo``, or a mix such as ``"cpu:gloo,cuda:nccl"``; any other backend
    raises ``RuntimeError``.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "make_process_mesh needs an initialized default process group; call "
            "torch.distributed.init_process_group first"
        )
    # Anything but nccl/gloo would silently take the NCCL comm paths and hang.
    check_backend()

    device = _resolve_device_type(device_type)
    sizes = _resolve_sizes(axes)
    names = list(sizes)
    model_axes = _resolve_model_axes(names, model_parallel_axes)
    if model_key in names:
        raise ValueError(f"model_key {model_key!r} collides with an axis name")

    # Structural argument checks are done; now hold the mesh to the live world.
    # ``init_device_mesh`` silently builds a mesh over only the first prod(sizes)
    # ranks otherwise, and the excluded ranks then hang or crash in the first
    # collective. Fail identically on every rank instead.
    world_size = dist.get_world_size()
    numel = math.prod(sizes.values())
    if numel != world_size:
        raise ValueError(
            f"mesh {dict(sizes)} needs exactly {numel} ranks but the process "
            f"group has {world_size}; launch with {numel} ranks or change the "
            f"axis sizes"
        )

    nd_mesh = dist.device_mesh.init_device_mesh(
        device_type=device,
        mesh_shape=tuple(sizes.values()),
        mesh_dim_names=tuple(names),
    )
    mesh = ProcessMesh({name: nd_mesh.get_group(name) for name in names}, nd_mesh)

    model_numel = math.prod(sizes[a] for a in model_axes)
    if len(model_axes) == len(names):
        flat_mesh = dist.device_mesh.init_device_mesh(
            device_type=device,
            mesh_shape=(model_numel,),
            mesh_dim_names=(model_key,),
        )
    else:
        replica_numel = math.prod(sizes.values()) // model_numel
        flat_mesh = dist.device_mesh.init_device_mesh(
            device_type=device,
            mesh_shape=(replica_numel, model_numel),
            mesh_dim_names=("replica", model_key),
        )
    mesh[model_key] = flat_mesh.get_group(model_key)
    return mesh
