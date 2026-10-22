"""Gradient clipping by the norm of the whole (unsharded) model's gradient."""

import math
from collections.abc import Iterable
from typing import Union

import torch
import torch.distributed as dist

from .._core.comm import get_device_of_process_group
from ._replicas import Groups, replica_groups


def clip_grad_norm_(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float,
    group: dist.ProcessGroup,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """Clip gradients by the norm of the whole model's gradient.

    Same semantics as ``torch.nn.utils.clip_grad_norm_`` on the unsharded
    model: shards and replicated parameters are each counted once, so the
    result is identical on every rank and independent of the process mesh.

    Collective over ``group``: call it on every rank after the backward, also
    on ranks without gradients. Replicated parameters must be marked with
    :func:`mark_replicated`; an unmarked one is counted once per rank holding
    it, which inflates the norm and over-clips.

    Parameters
    ----------
    parameters : torch.Tensor | Iterable[torch.Tensor]
        This rank's parameters; those without a gradient are skipped.
    max_norm : float
        Maximum norm of the gradients.
    group : dist.ProcessGroup
        The model-parallel group, over which the model is sharded and
        replicated (e.g. ``mesh["model"]``).
    norm_type : float
        Type of the p-norm; ``inf`` for the max norm.
    error_if_nonfinite : bool
        Raise if the total norm is ``nan`` or ``inf``.

    Returns
    -------
    torch.Tensor
        Total norm before clipping, in the promoted dtype of all ``parameters``
        (at least float32).

    Raises
    ------
    ValueError
        If a parameter is marked replicated over a group with ranks outside
        ``group``.
    RuntimeError
        If ``error_if_nonfinite`` is set and the total norm is not finite.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads: list[torch.Tensor] = []
    copies: list[int] = []
    # Visit every parameter, gradient or not, so all ranks agree on the dtype and
    # raise together instead of one hanging in the all-reduce.
    dtype = torch.float32
    cache: dict[Groups, int] = {}
    for p in parameters:
        dtype = torch.promote_types(dtype, p.dtype)
        g = replica_groups(p)
        if g not in cache:
            cache[g] = _copies(g, group)
        if p.grad is not None:
            grads.append(p.grad)
            copies.append(cache[g])
    norm_type = float(norm_type)
    device = grads[0].device if grads else get_device_of_process_group(group)

    with torch.no_grad():
        # Sum in float64 so many large shards neither overflow nor drown small ones.
        total = torch.zeros((), dtype=torch.float64, device=device)
        if math.isinf(norm_type):
            if grads:
                # Replicas hold the same values; the max needs no correction.
                total = _norms(grads, norm_type, device).max()
            dist.all_reduce(total, op=dist.ReduceOp.MAX, group=group)
        else:
            # Bucket by copy count to divide by a Python int (no host-to-device copy).
            by_copies: dict[int, list[torch.Tensor]] = {}
            for grad, n in zip(grads, copies, strict=True):
                by_copies.setdefault(n, []).append(grad)
            for n, gs in by_copies.items():
                total = total + _norms(gs, norm_type, device).pow(norm_type).sum() / n
            dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
            total = total.pow(1.0 / norm_type)

        if error_if_nonfinite and not torch.isfinite(total):
            raise RuntimeError(
                f"The total norm of order {norm_type} for gradients from "
                "`parameters` is non-finite, so it cannot be clipped. To disable "
                "this error and scale the gradients by the non-finite norm anyway, "
                "set `error_if_nonfinite=False`"
            )
        # Same coefficient as torch.nn.utils.clip_grad_norm_.
        clip_coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
        # foreach needs one device and dtype per call.
        buckets: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
        for grad in grads:
            buckets.setdefault((grad.device, grad.dtype), []).append(grad)
        for (dev, dt), gs in buckets.items():
            torch._foreach_mul_(gs, clip_coef.to(dev, dt))
    return total.to(dtype)


def _norms(
    grads: list[torch.Tensor], norm_type: float, device: torch.device
) -> torch.Tensor:
    """Return the per-gradient norms as one float64 vector on ``device``.

    Each norm is accumulated in at least float32, so fp16/bf16 norms neither
    overflow nor lose precision.

    Parameters
    ----------
    grads : list[torch.Tensor]
        Gradients, possibly on different devices.
    norm_type : float
        Type of the p-norm.
    device : torch.device
        Device of the result.

    Returns
    -------
    torch.Tensor
        Norm of each gradient, grouped by dtype (callers only sum or max it).
    """
    # foreach and stack need one (device, dtype) per call.
    buckets: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for grad in grads:
        buckets.setdefault((grad.device, grad.dtype), []).append(grad)
    norms: list[torch.Tensor] = []
    for (_, dt), gs in buckets.items():
        acc = torch.promote_types(dt, torch.float32)
        stacked = torch.stack(torch._foreach_norm(gs, norm_type, dtype=acc))
        norms.append(stacked.to(device, torch.float64))
    return torch.cat(norms)


def _copies(groups: Groups, group: dist.ProcessGroup) -> int:
    """Return the number of ranks of ``group`` holding one copy of a parameter.

    Parameters
    ----------
    groups : Groups
        The parameter's replica groups.
    group : dist.ProcessGroup
        Clipping group, which must contain every rank of ``groups``.

    Returns
    -------
    int
        Product of the sizes of ``groups``; 1 for an unmarked parameter.
    """
    if groups:
        ranks = set(dist.get_process_group_ranks(group))
        for g in groups:
            if not ranks.issuperset(dist.get_process_group_ranks(g)):
                raise ValueError(
                    f"a parameter is replicated over ranks "
                    f"{dist.get_process_group_ranks(g)}, not all in the clipping "
                    f"group {sorted(ranks)}; pass the model-parallel group, e.g. "
                    'mesh["model"].'
                )
    return math.prod(g.size() for g in groups)
