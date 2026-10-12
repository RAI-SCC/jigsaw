"""DDP comm hook that averages replicated gradients over their joint groups."""

from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fcol
from torch.nn.parallel import DistributedDataParallel

from ._replicas import Groups, replica_groups

# DDP attribute holding its JointCommHookState
_JOINT = "_jigsaw_joint_hook_state"
# Under DDP's Python reducer, parameter attributes holding the JointCommHookState
# that reduces it and the group it is averaged over
_JOINT_PARAM = "_jigsaw_joint_state"
_JOINT_PARAM_GROUP = "_jigsaw_joint_hook_group"
# (id(WORLD), partition as sorted rank tuples) -> this rank's group in it
_JOINT_GROUPS: dict[tuple[int, tuple[tuple[int, ...], ...]], dist.ProcessGroup] = {}


class JointCommHookState:
    """State of :func:`joint_comm_hook` for one ``DistributedDataParallel``.

    For each distinct replica-group tuple ``R`` among the parameters DDP
    reduces, builds the *joint* group: all ranks connected to this one through
    DDP's group and the groups in ``R``, i.e. every rank holding a copy of the
    parameter. Unmarked parameters use DDP's group.

    Construction is collective over the default group (it gathers group layouts
    and may create process groups; joint groups are cached). It also switches
    :func:`sync_grad` to scaling inside ``ddp``'s forward, so the hook must then
    be registered, or :func:`sync_grad` raises. :func:`register_joint_comm_hook`
    does both. Under DDP's Python reducer (compiled autograd), which reduces
    after every backward, :func:`sync_grad` scales the reduced parameters
    wherever they are used.

    Parameters
    ----------
    ddp : DistributedDataParallel
        Wrapped jigsaw model; its process group is the data-parallel axis.

    Raises
    ------
    RuntimeError
        If the ranks disagree on the model's replica groups.
    """

    def __init__(self, ddp: DistributedDataParallel) -> None:
        self.python_reducer: bool = getattr(ddp, "_use_python_reducer", False)
        self.ddp = ddp
        self.dp: dist.ProcessGroup = ddp.process_group
        # The parameters that end up in DDP's buckets.
        ignored: set[str] = getattr(ddp, "parameters_to_ignore", set())
        reduced = [
            p
            for n, p in ddp.module.named_parameters()
            if n not in ignored and p.requires_grad
        ]
        # Distinct replica-group tuples, numbered in parameter order (the same
        # on every rank) so that index k means the same tuple everywhere.
        keys: dict[Groups, int] = {}
        for p in reduced:
            r = replica_groups(p)
            if r and r not in keys:
                keys[r] = len(keys)

        # Each rank only knows the groups it is in; gather everyone's rank
        # lists so every rank can build the full partition.
        mine = (
            dist.get_process_group_ranks(self.dp),
            [[dist.get_process_group_ranks(g) for g in r] for r in keys],
        )
        everyone: list = [None] * dist.get_world_size()
        dist.all_gather_object(everyone, mine)
        if len({len(e[1]) for e in everyone}) != 1:
            raise RuntimeError(
                "ranks disagree on the number of distinct replica-group tuples; "
                "the jigsaw model must be built the same way on every rank"
            )

        self._group: dict[Groups, dist.ProcessGroup] = {}
        self._factor: dict[Groups, float] = {}
        for r, k in keys.items():
            # Connect all dp groups and all groups of tuple k across ranks.
            joint = _joint_group(
                [e[0] for e in everyone] + [g for e in everyone for g in e[1][k]]
            )
            self._group[r] = joint
            # The hook divides by |joint| instead of |dp|; the in-graph scale
            # by this factor turns that average back into AVG_dp(SUM_replica).
            self._factor[r] = joint.size() / self.dp.size()
        # parameter -> joint group (DDP's group when unmarked)
        self._of: dict[int, dist.ProcessGroup] = {
            id(p): self._group.get(replica_groups(p), self.dp) for p in reduced
        }
        setattr(ddp, _JOINT, self)
        if self.python_reducer:
            for p in reduced:
                setattr(p, _JOINT_PARAM, self)
                setattr(p, _JOINT_PARAM_GROUP, self._of[id(p)])

    def reduces(self, param: torch.Tensor) -> bool:
        """Return whether DDP's buckets, hence the hook, reduce ``param``."""
        return id(param) in self._of

    def factor(self, groups: Groups) -> float:
        """Return ``|joint group| / |dp|`` for replica groups ``groups``."""
        return self._factor[groups]

    def group_of(self, param: torch.Tensor) -> dist.ProcessGroup:
        """Return the group the hook averages ``param``'s gradient over."""
        return self._of[id(param)]


def _joint_group(edges: Sequence[Sequence[int]]) -> dist.ProcessGroup:
    """Return this rank's group in the partition the rank lists ``edges`` connect.

    Collective over the default group the first time a partition is seen. The
    same ``edges`` on every rank give the same partition, so all ranks agree on
    whether to create it.
    """
    # Union-find over all ranks: each rank list in ``edges`` joins its members.
    world = dist.get_world_size()
    parent = list(range(world))

    def find(a: int) -> int:
        # Path halving keeps the trees shallow.
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ranks in edges:
        for r in ranks[1:]:
            parent[find(r)] = find(ranks[0])
    # Connected components, in a canonical (sorted) form usable as a cache key.
    comps: dict[int, list[int]] = {}
    for r in range(world):
        comps.setdefault(find(r), []).append(r)
    partition = tuple(sorted(tuple(c) for c in comps.values()))
    if len(partition) == 1:
        return dist.group.WORLD  # type: ignore[return-value]
    # Keyed by WORLD too, so a re-initialized process group does not hit stale
    # entries.
    key = (id(dist.group.WORLD), partition)
    if key not in _JOINT_GROUPS:
        # Collective: every rank creates all parts and keeps its own.
        group, _ = dist.new_subgroups_by_enumeration([list(c) for c in partition])
        assert isinstance(group, dist.ProcessGroup)  # every rank is in a part
        _JOINT_GROUPS[key] = group
    return _JOINT_GROUPS[key]


def joint_comm_hook(
    state: JointCommHookState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """DDP comm hook: average each gradient over its parameter's joint group.

    The bucket is split into runs of consecutive parameters sharing a joint
    group, each averaged in one async all-reduce. With the in-graph scaling of
    :func:`sync_grad`, a replicated parameter ends up with
    ``AVG_dp(SUM_replica(local))``, as with the in-graph sum plus plain DDP.
    Gradient accumulation with ``no_sync`` stays correct, and the hook works
    under ``fp16_compress_wrapper`` / ``bf16_compress_wrapper``.

    DDP's Python reducer (compiled autograd) instead calls the hook once per
    parameter with ``(grad, param)``, reduces ``grad`` in place and ignores the
    return value.

    Parameters
    ----------
    state : JointCommHookState
        Built for the DDP this hook is registered on.
    bucket : dist.GradBucket | tuple[torch.Tensor, torch.Tensor]
        Bucket of flattened gradients, or ``(grad, param)``.

    Returns
    -------
    torch.futures.Future[torch.Tensor]
        Resolves to the reduced bucket buffer; ``None`` for ``(grad, param)``.
    """
    if isinstance(bucket, tuple):
        _reduce_param_grad(*bucket)
        return None  # type: ignore[return-value]
    buf = bucket.buffer()
    # Parameters lie back to back in the buffer; merge neighbours with the same
    # joint group into (group, start, end) runs.
    runs: list[tuple[dist.ProcessGroup, int, int]] = []
    off = 0
    for p in bucket.parameters():
        g, n = state.group_of(p), p.numel()
        if runs and runs[-1][0] is g:
            runs[-1] = (g, runs[-1][1], off + n)
        else:
            runs.append((g, off, off + n))
        off += n
    if off != buf.numel():
        raise RuntimeError(
            f"bucket buffer has {buf.numel()} elements but its parameters {off}"
        )
    # Average each run in place over its group: pre-divide, then SUM.
    works = []
    for g, a, b in runs:
        seg = buf[a:b]
        seg.div_(g.size())
        works.append(dist.all_reduce(seg, group=g, async_op=True))
    # DDP expects a single future; chain the others onto the last one.
    last = works[-1].get_future()

    def join(_: Any) -> torch.Tensor:
        # Wait on each process group's own future: it carries NCCL's stream
        # dependency, which ``torch.futures.collect_all`` would drop.
        for w in works[:-1]:
            w.get_future().wait()
        return buf

    return last.then(join)


def _reduce_param_grad(grad: torch.Tensor, param: torch.Tensor) -> None:
    """Average ``grad`` over ``param``'s joint group in place (Python reducer)."""
    group = getattr(param, _JOINT_PARAM_GROUP)
    if replica_groups(param):
        # Inductor's DDP fusion merges every ``div -> all_reduce -> copy_`` of
        # the backward into one all-reduce over the last one's group. The joint
        # groups differ per parameter, so pre-scale with ``mul`` to stay out of it.
        grad_in = grad * (1.0 / group.size())
    else:
        # Over DDP's own group, as DDP does it, so these still fuse.
        grad_in = grad / group.size()
    grad.copy_(fcol.all_reduce(grad_in, "sum", group))


def register_joint_comm_hook(
    ddp: DistributedDataParallel,
    wrapper: Optional[Callable[[Callable], Callable]] = None,
) -> JointCommHookState:
    """Reduce replicated gradients over their joint groups in DDP's buckets.

    Replaces the in-graph sum of :func:`sync_grad` (for the parameters ``ddp``
    reduces) and DDP's own ``dp`` average by one average over the joint group
    ``dp x replica groups``, overlapped with the backward pass. Collective over
    the default group; call it on every rank, before the first forward.

    Parameters
    ----------
    ddp : DistributedDataParallel
        Wrapped jigsaw model, built over the data-parallel axis.
    wrapper : Callable, optional
        Wraps the hook before registering, e.g.
        ``torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper``.
        Not with DDP's Python reducer, which has no buckets to compress.

    Returns
    -------
    JointCommHookState
        The hook's state.

    Raises
    ------
    ValueError
        If ``wrapper`` is given and ``ddp`` uses the Python reducer.
    """
    if wrapper is not None and getattr(ddp, "_use_python_reducer", False):
        raise ValueError(
            "comm hook wrappers need DDP's gradient buckets, which DDP's Python "
            "reducer (compiled autograd) does not have"
        )
    state = JointCommHookState(ddp)
    hook = joint_comm_hook if wrapper is None else wrapper(joint_comm_hook)
    ddp.register_comm_hook(state, hook)
    return state


def active_joint_state() -> Optional[JointCommHookState]:
    """Return the joint-hook state of the DDP whose forward is running, if any."""
    ddp = getattr(DistributedDataParallel, "_active_ddp_module", None)
    state = getattr(ddp, _JOINT, None) if ddp is not None else None
    if not isinstance(state, JointCommHookState):
        return None
    _require_hook(state)
    return state


def joint_state_of(
    param: torch.Tensor, active: Optional[JointCommHookState]
) -> Optional[JointCommHookState]:
    """Return the joint-hook state whose hook reduces ``param``, if any."""
    if active is not None:
        return active if active.reduces(param) else None
    # DDP's Python reducer runs the forward outside ``_inside_ddp_forward``, so
    # there is no active DDP; it reduces each of its parameters in a
    # post-accumulate hook after any backward instead, which the mark follows.
    state = getattr(param, _JOINT_PARAM, None)
    if state is not None:
        _require_hook(state)
    return state


def _require_hook(state: JointCommHookState) -> None:
    """Raise unless ``joint_comm_hook`` is registered with ``state``."""
    # A state without its hook would make sync_grad only scale, and nothing sum.
    if not any(s is state for _, s in getattr(state.ddp, "_comm_hooks", ())):
        raise RuntimeError(
            "a JointCommHookState was built for this DDP but joint_comm_hook is not "
            "registered on it; without the hook the replicated gradients are never "
            "summed. Use jigsaw.parallel.register_joint_comm_hook(ddp)."
        )
