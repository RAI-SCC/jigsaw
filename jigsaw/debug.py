"""Debugging and inspection aids for distributed training."""

from collections.abc import Iterator, Mapping
from typing import Optional

import torch
import torch.distributed as dist

from . import parallel
from ._core.comm import all_reduce_max, get_device_of_process_group
from .parallel._replicas import replica_divergence

__all__ = ["assert_replica_consistency", "count_parameters"]


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

    An unmarked parameter replicated over a model-parallel axis keeps only its
    local gradient share, so its copies drift apart without the check above
    noticing. ``unmarked`` catches this for modules from outside jigsaw (a
    plain ``torch.nn.Linear`` in a jigsaw model, say) by comparing their
    unmarked parameters over every axis of ``mesh``. Leave it off if you shard
    such parameters yourself.

    Collective over every group in ``mesh``: call it on every rank. Every rank
    raises the same error or none does.

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
        If a parameter differs by more than ``atol``; the message names them.
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
    div = replica_divergence([(p, g) for _, p, g, _ in entries])
    if not entries:
        div = torch.zeros(1, dtype=torch.float64, device=_device(mesh))
    all_reduce_max(div, mesh.values())
    # Not strict: with no entries ``div`` still holds the one placeholder above.
    bad = [
        (f"{name} ({d:.3g})", missing_mark)
        for (name, _, _, missing_mark), d in zip(entries, div.tolist(), strict=False)
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
                "module through jigsaw.parallel.synced_call."
            )
        raise AssertionError(msg)


def count_parameters(
    model: torch.nn.Module,
    mesh: Optional[Mapping[str, dist.ProcessGroup]] = None,
    model_key: str = "model",
) -> int:
    """Count the trainable parameters of a model.

    Without ``mesh``, counts the parameters held by the calling rank. With
    ``mesh``, counts the whole model: local counts are summed over
    ``mesh[model_key]``, and a parameter marked by
    :func:`jigsaw.parallel.mark_replicated` is counted only on rank 0 of each
    of its replica groups. Data-parallel copies are not counted. An unmarked
    parameter is taken to be a shard, so an unmarked replica is counted once
    per copy.

    With ``mesh``, collective over ``mesh[model_key]``: call it on every rank.

    Parameters
    ----------
    model : torch.nn.Module
        Model to inspect (unwrapped, or wrapped in ``DistributedDataParallel``).
    mesh : Mapping[str, dist.ProcessGroup], optional
        Process mesh the model lives on.
    model_key : str
        Key of the group spanning one model copy in ``mesh``.

    Returns
    -------
    int
        Number of parameters with ``requires_grad``, local or of the whole model.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if mesh is None:
        return sum(p.numel() for p in params)
    local = sum(
        p.numel()
        for p in params
        if all(g.rank() == 0 for g in parallel.replica_groups(p))
    )
    total = torch.tensor(local, dtype=torch.int64, device=_device(mesh))
    dist.all_reduce(total, group=mesh[model_key])
    return int(total.item())


def _named_parameters(
    model: torch.nn.Module,
) -> Iterator[tuple[str, torch.nn.Parameter, bool]]:
    """Yield ``(name, param, foreign)``; ``foreign`` if no jigsaw module owns it.

    A parameter of a plain submodule *inside* a jigsaw module (the
    ``torch.nn.Linear`` projections of an unsharded-channel
    ``DistributedAttention``, say) is jigsaw's: the module shards or syncs it
    itself. Same parameters, names and order as
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
    return get_device_of_process_group(next(iter(mesh.values())))
