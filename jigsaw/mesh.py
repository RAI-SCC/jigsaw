"""Build the ``process_mesh_dict`` that every jigsaw module expects.

A jigsaw module is configured with a mapping ``{axis name -> ProcessGroup}``.
The final two entries are the domain- / tensor-parallel axes (``row`` and
``col`` by default); any leading entries are data-parallel *replica* axes. On
top of the per-axis groups the mapping carries a ``model`` group that spans one
full parameter replica -- the flattened product of the model-parallel axes --
which the parameter-free matmul, the weight-merging converters and seed setup
all read.

``torch.distributed.device_mesh.init_device_mesh`` builds the per-axis groups
but not that flattened ``model`` group, and it silently succeeds when the mesh
covers only some of the world. :func:`make_process_mesh` fills the first gap and
turns the second into a loud, identical error on every rank.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist

__all__ = ["make_process_mesh"]


def _resolve_device_type(device_type: str | None) -> str:
    if device_type is not None:
        return device_type
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_sizes(axes: Mapping[str, int]) -> dict[str, int]:
    """Copy ``axes`` and fill a single ``-1`` from the world size."""
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
    """Validate that the model-parallel axes are a trailing, contiguous run."""
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
) -> dict[str, dist.ProcessGroup]:
    """Build a jigsaw ``process_mesh_dict`` from named, ordered mesh axes.

    Parameters
    ----------
    axes : Mapping[str, int]
        Ordered mapping of axis name to size, slowest-varying first. Exactly one
        size may be ``-1``, inferred so the product matches the world size. The
        trailing axes are the model-parallel (domain- / tensor-parallel) axes;
        for a fully model-parallel run that is every axis, e.g.
        ``{"row": 2, "col": 2}`` or ``{"sh": 2, "sw": 2, "col": 2}``. Leading
        axes are data-parallel replicas, e.g. ``{"dp": -1, "row": 2, "col": 2}``.
    device_type : str, optional
        Device type passed to ``init_device_mesh``. Defaults to ``"cuda"`` when a
        CUDA device is visible, else ``"cpu"``.
    model_parallel_axes : Sequence[str], optional
        Which trailing axes form one parameter replica and are flattened into
        the ``model_key`` group. Must be a contiguous run at the end of ``axes``.
        Defaults to every axis.
    model_key : str, default "model"
        Key for the flattened model-parallel group.

    Returns
    -------
    dict[str, torch.distributed.ProcessGroup]
        One group per entry of ``axes`` plus the ``model_key`` group. Pass it
        straight to a jigsaw module, or to :func:`jigsaw.convert`.

    Notes
    -----
    Requires an initialised default process group. Groups are laid out row-major
    over global ranks, matching ``init_device_mesh``.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "make_process_mesh needs an initialised default process group; call "
            "torch.distributed.init_process_group first"
        )

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
    mesh: dict[str, dist.ProcessGroup] = {
        name: nd_mesh.get_group(name) for name in names
    }

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
