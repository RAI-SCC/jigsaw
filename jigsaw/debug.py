"""Debugging aids for distributed training."""

from collections.abc import Mapping

import torch
import torch.distributed as dist

from . import parallel

__all__ = ["assert_replica_consistency"]


def assert_replica_consistency(
    model: torch.nn.Module,
    mesh: Mapping[str, dist.ProcessGroup],
    dp_key: str = "dp",
    atol: float = 0.0,
) -> None:
    """Assert that every replicated parameter is identical across its replicas.

    Each parameter marked with :func:`jigsaw.parallel.mark_replicated` is compared
    with its copy on rank 0 of each of its replica groups and, if ``mesh`` has a
    ``dp_key`` axis, every parameter with its copy on data-parallel rank 0. Call
    it after ``optimizer.step()`` to find the first step at which replicas
    diverge.

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

    Raises
    ------
    AssertionError
        Naming the parameters that differ by more than ``atol``.
    """
    dp = mesh.get(dp_key)
    entries = []
    for name, p in model.named_parameters():
        groups = list(parallel.replica_groups(p))
        if dp is not None:
            groups.append(dp)
        entries.append((name, p, groups))
    div = parallel.replica_divergence([(p, g) for _, p, g in entries])
    if not entries:
        div = torch.zeros(1, dtype=torch.float64, device=_device(mesh))
    parallel.all_reduce_max(div, mesh.values())
    bad = [
        f"{name} ({d:.3g})"
        for (name, _, _), d in zip(entries, div.tolist())
        if d > atol
    ]
    if bad:
        raise AssertionError(
            f"{len(bad)} parameter(s) differ across their replicas by more than "
            f"{atol} (max |p - p_rank0|): " + ", ".join(bad)
        )


def _device(mesh: Mapping[str, dist.ProcessGroup]) -> torch.device:
    group = next(iter(mesh.values()))
    if dist.get_backend(group) == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")
