"""Debugging aids for distributed training."""

from collections.abc import Iterator, Mapping

import torch
import torch.distributed as dist

from . import parallel

__all__ = ["assert_replica_consistency"]


def assert_replica_consistency(
    model: torch.nn.Module,
    mesh: Mapping[str, dist.ProcessGroup],
    dp_key: str = "dp",
    atol: float = 0.0,
    unmarked: bool = False,
) -> None:
    """Assert that every replicated parameter is identical across its replicas.

    Each parameter marked with :func:`jigsaw.parallel.mark_replicated` is compared
    with its copy on rank 0 of each of its replica groups and, if ``mesh`` has a
    ``dp_key`` axis, every parameter with its copy on data-parallel rank 0. Call
    it after ``optimizer.step()`` to find the first step at which replicas
    diverge.

    A parameter that is replicated over a model-parallel axis but carries no mark
    keeps only its local gradient share, so its copies drift apart; the check
    above cannot see that, because it only knows the groups a mark names. With
    ``unmarked`` every unmarked parameter of a module from outside jigsaw (a
    plain ``torch.nn.Linear`` in a jigsaw model, say) is compared over *every*
    axis of ``mesh`` too. Leave it off for such modules whose parameters you
    shard yourself.

    Collective over every group in ``mesh``: call it on every rank. The verdict
    is reduced over the whole mesh, so every rank raises the same error or none
    does.

    Parameters
    ----------
    model : torch.nn.Module
        Model to check (unwrapped, or wrapped in ``DistributedDataParallel``).
    mesh : Mapping[str, dist.ProcessGroup]
        Process mesh the model lives on.
    dp_key : str
        Key of the data-parallel axis in ``mesh``; ignored if absent.
    atol : float
        Largest absolute difference tolerated; ``0`` asks for bitwise identity.
    unmarked : bool
        Also compare the unmarked parameters of non-jigsaw modules over every
        axis of ``mesh``.

    Raises
    ------
    AssertionError
        Naming the parameters that differ by more than ``atol``.
    """
    dp = mesh.get(dp_key)
    every_axis = list({id(g): g for g in mesh.values()}.values())
    entries = []
    for name, p, foreign in _named_parameters(model):
        groups = list(parallel.replica_groups(p))
        missing_mark = unmarked and foreign and not groups
        if missing_mark:
            groups = every_axis
        elif dp is not None:
            groups.append(dp)
        entries.append((name, p, groups, missing_mark))
    div = parallel.replica_divergence([(p, g) for _, p, g, _ in entries])
    if not entries:
        div = torch.zeros(1, dtype=torch.float64, device=_device(mesh))
    parallel.all_reduce_max(div, mesh.values())
    bad = [
        (f"{name} ({d:.3g})", missing_mark)
        for (name, _, _, missing_mark), d in zip(entries, div.tolist())
        if d > atol
    ]
    if bad:
        msg = (
            f"{len(bad)} parameter(s) differ across their replicas by more than "
            f"{atol} (max |p - p_rank0|): " + ", ".join(b for b, _ in bad)
        )
        no_mark = [b for b, missing_mark in bad if missing_mark]
        if no_mark:
            msg += (
                f". {len(no_mark)} of them carry no replica mark, so their "
                "gradient is never summed over the model-parallel axes: "
                + ", ".join(no_mark)
                + ". Mark them with jigsaw.parallel.mark_replicated and run their "
                "module through jigsaw.parallel.synced_parameters."
            )
        raise AssertionError(msg)


def _named_parameters(
    model: torch.nn.Module,
) -> Iterator[tuple[str, torch.nn.Parameter, bool]]:
    """Yield ``(name, param, foreign)``; ``foreign`` if no jigsaw module owns it.

    A parameter of a plain submodule *inside* a jigsaw module (the
    ``torch.nn.GroupNorm`` of ``DistributedGroupNorm``, say) is jigsaw's: the
    module shards or syncs it itself. Same parameters, names and order as
    ``model.named_parameters()``.
    """
    # Every module at or below a jigsaw module.
    owned = {
        id(sub)
        for m in model.modules()
        if type(m).__module__.startswith("jigsaw.")
        for sub in m.modules()
    }
    seen: set[int] = set()
    for mod_name, m in model.named_modules():
        foreign = id(m) not in owned
        for name, p in m.named_parameters(prefix=mod_name, recurse=False):
            if id(p) not in seen:
                seen.add(id(p))
                yield name, p, foreign


def _device(mesh: Mapping[str, dist.ProcessGroup]) -> torch.device:
    group = next(iter(mesh.values()))
    if dist.get_backend(group) == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")
