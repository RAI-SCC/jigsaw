"""Wrapping a jigsaw model in DDP, and checking that the two compose."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .._core.comm import all_reduce_max, get_device_of_process_group
from ._joint import register_joint_comm_hook
from ._replicas import broadcast_replicas, replica_divergence, replica_groups

# DDP attribute set once check_ddp has passed
_CHECKED = "_jigsaw_ddp_checked"


def wrap_ddp(
    module: torch.nn.Module,
    mesh: Mapping[str, dist.ProcessGroup],
    dp_key: str = "dp",
    joint_hook: bool = False,
    **ddp_kwargs: Any,
) -> DistributedDataParallel:
    """Wrap a jigsaw model in ``DistributedDataParallel`` over the data-parallel axis.

    First makes the marked parameters identical across their replica groups
    (see :func:`broadcast_replicas`; jigsaw modules do this when built, but
    parameters marked by hand may need it). DDP then broadcasts the model over
    ``mesh[dp_key]`` and averages gradients over it. The result is checked with
    :func:`check_ddp`.

    A ``DistributedDataParallel(module, process_group=mesh[dp_key])`` built by
    hand is checked the same way, in its first forward.

    Parameters
    ----------
    module : torch.nn.Module
        Model to wrap.
    mesh : Mapping[str, dist.ProcessGroup]
        Process mesh, e.g. from :func:`jigsaw.make_process_mesh`.
    dp_key : str
        Key of the data-parallel axis in ``mesh``.
    joint_hook : bool
        Also call :func:`register_joint_comm_hook` (without compression).
    **ddp_kwargs : Any
        Passed on to ``DistributedDataParallel``; ``process_group`` is not
        accepted.

    Returns
    -------
    DistributedDataParallel
        The wrapped model.
    """
    if "process_group" in ddp_kwargs:
        raise TypeError("wrap_ddp sets process_group to mesh[dp_key] itself")
    if dp_key not in mesh:
        raise KeyError(
            f"mesh has no data-parallel axis {dp_key!r}; its axes are {list(mesh)}"
        )
    broadcast_replicas(module)
    ddp = DistributedDataParallel(module, process_group=mesh[dp_key], **ddp_kwargs)
    check_ddp(ddp)
    if joint_hook:
        register_joint_comm_hook(ddp)
    return ddp


def _model_groups(module: torch.nn.Module) -> list[dist.ProcessGroup]:
    """Return the groups ``module``'s parameters are sharded or replicated over.

    These are the replica groups of the marked parameters plus every process
    group held by a jigsaw module that owns parameters, in module order (the
    same on every rank) and of size > 1.
    """
    # Ordered set keyed by identity; insertion order is the same on every rank.
    out: dict[int, dist.ProcessGroup] = {}

    def add(g: Any) -> None:
        if isinstance(g, dist.ProcessGroup) and g.size() > 1:
            out.setdefault(id(g), g)

    for m in module.modules():
        # Sharded parameters carry no mark; find their groups by scanning the
        # attributes (and list/tuple attributes) of jigsaw modules.
        if type(m).__module__.startswith("jigsaw.") and any(
            True for _ in m.parameters(recurse=False)
        ):
            for v in vars(m).values():
                for g in v if isinstance(v, (list, tuple)) else (v,):
                    add(g)
        for p in m.parameters(recurse=False):
            for g in replica_groups(p):
                add(g)
    return list(out.values())


def check_ddp(ddp: DistributedDataParallel) -> None:
    """Check that ``ddp`` composes with the jigsaw model it wraps.

    Raises if DDP's process group shares a rank other than the caller with any
    group the model is sharded or replicated over (DDP would average different
    shards), or if a marked parameter differs across its replica groups (DDP
    only broadcasts over its own group).

    Collective over DDP's group and the model's groups: call it on every rank.
    The verdict is reduced over all of them, so every rank raises or none does.

    Parameters
    ----------
    ddp : DistributedDataParallel
        Wrapped jigsaw model.

    Raises
    ------
    RuntimeError
        On a wrong process group or diverged replicas.
    """
    module = ddp.module
    dp = ddp.process_group
    me = dist.get_rank()
    dp_ranks = set(dist.get_process_group_ranks(dp))
    groups = _model_groups(module)
    # On a proper mesh, dp and every model group intersect only in this rank.
    wrong_group = any(
        dp_ranks & set(dist.get_process_group_ranks(g)) - {me} for g in groups
    )

    named = [(n, p, replica_groups(p)) for n, p in module.named_parameters()]
    named = [x for x in named if x[2]]
    params = [(p, g) for _, p, g in named]
    first = next(module.parameters(), None)
    device = first.device if first is not None else get_device_of_process_group(dp)
    # Pack both checks into one vector, [wrong_group, divergence per marked
    # param], and MAX-reduce it so every rank reaches the same verdict.
    flags = torch.cat(
        [
            torch.tensor([float(wrong_group)], dtype=torch.float64, device=device),
            replica_divergence(params).to(device),
        ]
    )
    all_reduce_max(flags, [dp, *groups])

    if flags[0] > 0:
        raise RuntimeError(
            "DistributedDataParallel's process group spans model-parallel ranks "
            f"(ranks {dist.get_process_group_ranks(dp)}); it would average sharded "
            "parameters with other shards. Build it over the data-parallel axis: "
            'DistributedDataParallel(model, process_group=mesh["dp"]), or '
            "jigsaw.parallel.wrap_ddp(model, mesh)."
        )
    bad = [n for (n, _, _), d in zip(named, flags[1:].tolist(), strict=True) if d > 0]
    if bad:
        raise RuntimeError(
            f"{len(bad)} replicated parameter(s) differ across their replica groups "
            f"before training, e.g. {bad[:5]}. DistributedDataParallel only "
            "broadcasts over the data-parallel axis; call "
            "jigsaw.parallel.broadcast_replicas(model) before wrapping, or build "
            "it with jigsaw.parallel.wrap_ddp(model, mesh)."
        )
    setattr(ddp, _CHECKED, True)


def check_active_ddp() -> None:
    """Run :func:`check_ddp` once on a hand-built DDP, from inside its forward.

    ``DistributedDataParallel`` records itself while its forward runs; the first
    jigsaw forward inside checks it. Skipped under ``torch.compile`` (only
    :func:`wrap_ddp` checks then).
    """
    ddp = getattr(DistributedDataParallel, "_active_ddp_module", None)
    if ddp is None or getattr(ddp, _CHECKED, False):
        return
    if torch.compiler.is_compiling():
        return
    check_ddp(ddp)
