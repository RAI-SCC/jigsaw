"""Replica marks on parameters, and keeping the marked replicas identical."""

from collections.abc import Sequence
from typing import Optional

import torch
import torch.distributed as dist

# parameter attribute holding its replica groups
_ATTR = "_jigsaw_replica_groups"


Groups = tuple[dist.ProcessGroup, ...]


def _as_groups(
    groups: "dist.ProcessGroup | Sequence[Optional[dist.ProcessGroup]]",
) -> Groups:
    """Normalize to a tuple of the groups with a real (>1) split."""
    seq = [groups] if isinstance(groups, dist.ProcessGroup) else list(groups)
    return tuple(g for g in seq if g is not None and g.size() > 1)


def mark_replicated(
    param: torch.Tensor,
    groups: "dist.ProcessGroup | Sequence[Optional[dist.ProcessGroup]]",
    extend: bool = False,
) -> None:
    """Mark a parameter as replicated over ``groups``.

    Its gradient is then summed over each group in turn by :func:`sync_grad`.
    Groups of size 1 (and ``None``) are dropped; marking with only such groups
    is a no-op.

    Parameters
    ----------
    param : torch.Tensor
        Parameter to mark.
    groups : dist.ProcessGroup | Sequence[dist.ProcessGroup | None]
        Process group(s) the parameter is replicated over.
    extend : bool
        Add ``groups`` to an existing mark instead of raising, e.g. an ensemble
        axis on top of a jigsaw module's spatial group. Groups with the same
        ranks as one already marked are skipped.

    Raises
    ------
    RuntimeError
        If ``param`` is already marked and ``extend`` is false; a second SUM
        over the same group would scale its gradient by the group size.
    """
    old = replica_groups(param)
    if old and not extend:
        raise RuntimeError(
            "parameter is already marked replicated; a second mark would sum "
            "its gradient twice. Pass extend=True to add groups to the mark."
        )
    # Deduplicate by rank set, not group identity: two group objects over the
    # same ranks would still sum the gradient twice.
    seen = {tuple(dist.get_process_group_ranks(g)) for g in old}
    new = list(old)
    for g in _as_groups(groups):
        ranks = tuple(dist.get_process_group_ranks(g))
        if ranks not in seen:
            seen.add(ranks)
            new.append(g)
    if new:
        setattr(param, _ATTR, tuple(new))


def replica_groups(param: Optional[torch.Tensor]) -> Groups:
    """Return the groups ``param`` is marked replicated over, or ``()``."""
    return getattr(param, _ATTR, ())


def broadcast_replicas(module: torch.nn.Module, src: int = 0) -> None:
    """Make every marked parameter of ``module`` identical over its groups.

    Each parameter is broadcast from group rank ``src`` over each of its replica
    groups in turn, which leaves it equal across their whole product. Collective
    over those groups: call it on every rank.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose marked parameters to broadcast.
    src : int
        Group rank to broadcast from.
    """
    # Bucket by replica-group tuple; parameters() order is the same on every
    # rank, so all ranks issue the broadcasts in the same order.
    buckets: dict[Groups, list[torch.Tensor]] = {}
    for p in module.parameters():
        if replica_groups(p):
            buckets.setdefault(replica_groups(p), []).append(p)
    with torch.no_grad():
        for groups, params in buckets.items():
            for group in groups:
                _broadcast_coalesced(params, group, src)


def _broadcast_coalesced(
    tensors: list[torch.Tensor], group: dist.ProcessGroup, src: int
) -> None:
    """Broadcast ``tensors`` in place from group rank ``src``, one call per dtype."""
    # A flat buffer needs a single dtype, so split the tensors by dtype first.
    by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
    for t in tensors:
        by_dtype.setdefault(t.dtype, []).append(t)
    for ts in by_dtype.values():
        # Pack into one contiguous buffer (a copy) so that each dtype needs a
        # single collective instead of one per tensor.
        flat = torch.cat([t.detach().reshape(-1) for t in ts])
        # On ``src`` the buffer is sent; everywhere else it is overwritten.
        dist.broadcast(flat, group_src=src, group=group)
        # Unpack: copy each slice back into its tensor. ``detach`` lets this
        # write into leaf parameters that require grad.
        for t, chunk in zip(ts, flat.split([t.numel() for t in ts])):
            t.detach().copy_(chunk.view_as(t))


def replica_divergence(
    params: Sequence[tuple[torch.Tensor, Sequence[dist.ProcessGroup]]],
) -> torch.Tensor:
    """Return ``max|p - p_0|`` per parameter, over each of its groups.

    ``p_0`` is the copy on group rank 0. The result is local; reduce it over the
    mesh before acting on it. Collective over every group listed, in order.

    Parameters
    ----------
    params : Sequence[tuple[torch.Tensor, Sequence[dist.ProcessGroup]]]
        Parameters, each with the groups it should be identical over.

    Returns
    -------
    torch.Tensor
        One float64 entry per parameter; ``inf`` where they differ in a NaN.
    """
    device = params[0][0].device if params else torch.device("cpu")
    div = torch.zeros(len(params), dtype=torch.float64, device=device)
    # Bucket by group tuple so each bucket needs one broadcast per group.
    buckets: dict[tuple[dist.ProcessGroup, ...], list[int]] = {}
    for i, (_, groups) in enumerate(params):
        groups = tuple(g for g in groups if g is not None and g.size() > 1)
        if groups:
            buckets.setdefault(groups, []).append(i)
    with torch.no_grad():
        for groups, idx in buckets.items():
            # float64 holds fp16 / bf16 / fp32 exactly, so the check is bitwise.
            mine = torch.cat([params[i][0].detach().reshape(-1).double() for i in idx])
            sizes = [params[i][0].numel() for i in idx]
            for group in groups:
                # Fetch group rank 0's copy and compare elementwise.
                ref = mine.clone()
                dist.broadcast(ref, group_src=0, group=group)
                d = (mine - ref).abs()
                # NaN on one rank only is a divergence; NaN on both is not.
                d = torch.where(mine.isnan() != ref.isnan(), torch.inf, d.nan_to_num())
                for i, chunk in zip(idx, d.split(sizes)):
                    if chunk.numel():
                        div[i] = torch.maximum(div[i], chunk.max())
    return div
