"""In-graph gradient synchronization of replicated parameters.

A parameter that is replicated over a process group (e.g. a conv weight over the
spatial grid, or a linear bias over the row group) sees only its local share of
the gradient on each rank; the shares have to be summed over the group. This
module does that *inside the autograd graph*: :func:`sync_grad` routes the
parameters through an identity :class:`torch.autograd.Function` whose backward
all-reduces the incoming gradients. The reduction therefore happens upstream of
the parameter's ``AccumulateGrad`` node, which gives three properties a
``register_post_accumulate_grad_hook`` cannot:

* It completes before anything that observes ``p.grad`` runs -- in particular
  ``DistributedDataParallel``'s reducer hook on the same ``AccumulateGrad``
  node. An asynchronous post-accumulate hook races that reducer, which then
  overwrites ``p.grad`` with the un-summed local share.
* Every backward pass reduces its own gradient, not the running ``p.grad`` sum,
  so gradient accumulation over microbatches stays correct.
* Several parameters sharing a group tuple and dtype are reduced in one flat
  collective.

Usage: mark a parameter once with :func:`mark_replicated` where it is created,
and route it through :func:`sync_grad` (or :func:`synced_parameters` for
submodules whose forward jigsaw does not own) in the forward.

Data parallelism composes with this through a plain ``DistributedDataParallel``
over the mesh's data-parallel axis, ``DistributedDataParallel(model,
process_group=mesh["dp"])``; :func:`wrap_ddp` is a shorthand that also
broadcasts the replicas first. A DDP whose process group reaches across the
model-parallel axes (the default group, say) averages different shards together
without any error, so :func:`check_ddp` rejects it. It runs from the first jigsaw
forward inside any ``DistributedDataParallel``, and from :func:`wrap_ddp`.

With plain DDP, a replicated gradient is reduced twice: summed over its replica
groups in the graph, then averaged over ``dp`` by DDP. After the first step all
replicas hold the same gradient, so the second all-reduce sends it over the
(typically inter-node) ``dp`` links once per replica. :func:`register_joint_comm_hook`
replaces both with one DDP comm hook: in the graph the gradient is only scaled,
and DDP averages it over the *joint* group ``dp x replica groups`` in its
buckets. That moves the replica sum off the critical path and lets NCCL reduce
within a node before crossing nodes.
"""

import os
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

_ATTR = "_jigsaw_replica_groups"
_CHECKED = "_jigsaw_ddp_checked"
#: set on the hooks of :func:`jigsaw.hooks.make_grad_sync_hook`
HOOK_TAG = "_jigsaw_grad_sync_hook"
#: Mutation-testing switch: with ``JIGSAW_DISABLE_GRAD_SYNC=1`` in the environment
#: :func:`sync_grad` passes everything through, so replicated parameters keep only
#: their local gradient share. Training is then wrong; the test suite must notice.
SYNC_DISABLED = os.environ.get("JIGSAW_DISABLE_GRAD_SYNC", "") == "1"

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
        Append ``groups`` to an existing mark instead of raising, e.g. to add
        an ensemble axis to a parameter a jigsaw module already marks over its
        spatial group. A group spanning the same ranks as one already in the
        mark is dropped, so the gradient is still summed over it only once.

    Raises
    ------
    RuntimeError
        If ``param`` is already marked and ``extend`` is false. A second SUM
        over the same group would scale its gradient by the group size.
    """
    old = replica_groups(param)
    if old and not extend:
        raise RuntimeError(
            "parameter is already marked replicated; a second mark would sum "
            "its gradient twice. Pass extend=True to add groups to the mark."
        )
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


class _SyncGrad(torch.autograd.Function):
    """Identity in forward; SUM all-reduce of the gradients in backward."""

    @staticmethod
    def forward(groups: Groups, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return views of ``tensors``."""
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
        """Save the groups for the backward pass."""
        ctx.groups = inputs[0]

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> tuple:
        """Sum the gradients over each group, in one flat collective per group.

        Autograd materializes a zero gradient for an unused output, so every
        rank reaching this node sends the same-sized buffer.
        """
        # ``cat`` copies, so the incoming gradients are never written in place.
        flat = torch.cat([g.reshape(-1) for g in grads])
        for group in ctx.groups:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        chunks = flat.split([g.numel() for g in grads])
        return (None, *(c.view_as(g) for c, g in zip(chunks, grads)))

    @staticmethod
    def vmap(
        info: Any, in_dims: tuple[Optional[int], ...], groups: Groups, *tensors: Any
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Optional[int], ...]]:
        """Batching rule: the reduction is elementwise, so reduce the whole batch."""
        return _SyncGrad.apply(groups, *tensors), tuple(in_dims[1:])


class _ScaleGrad(torch.autograd.Function):
    """Identity in forward; multiplies the gradients by ``factor`` in backward.

    Stands in for :class:`_SyncGrad` under the joint comm hook, which averages
    over ``factor`` times as many ranks as the plain ``dp`` mean.
    """

    @staticmethod
    def forward(factor: float, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return views of ``tensors``."""
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
        """Save the factor for the backward pass."""
        ctx.factor = inputs[0]

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> tuple:
        """Scale every gradient by ``factor``."""
        return (None, *(g * ctx.factor for g in grads))

    @staticmethod
    def vmap(
        info: Any, in_dims: tuple[Optional[int], ...], factor: float, *tensors: Any
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Optional[int], ...]]:
        """Batching rule: the scaling is elementwise, so scale the whole batch."""
        return _ScaleGrad.apply(factor, *tensors), tuple(in_dims[1:])


def sync_grad(*tensors: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], ...]:
    """Route marked parameters through an in-graph gradient all-reduce.

    Unmarked tensors, ``None`` and tensors that do not require a gradient are
    returned unchanged, as is everything when grad mode is off. The marked ones
    are grouped by ``(groups, dtype)`` and each group is reduced in one flat
    collective per process group. Inside the forward of a DDP with
    :func:`register_joint_comm_hook`, the parameters that DDP reduces are only
    scaled instead; the hook does the sum. :data:`SYNC_DISABLED` turns this into
    a pass-through, for mutation testing only.

    Parameters
    ----------
    *tensors : torch.Tensor | None
        Parameters, typically a module's weight and bias.

    Returns
    -------
    tuple[torch.Tensor | None, ...]
        The tensors to use in the forward, in the same order.
    """
    _check_active_ddp()
    out = list(tensors)
    if SYNC_DISABLED or not torch.is_grad_enabled():
        return tuple(out)
    joint = _active_joint_state()
    # (groups, dtype, scale only) -> indices
    buckets: dict[tuple[Groups, torch.dtype, bool], list[int]] = {}
    for i, t in enumerate(tensors):
        groups = replica_groups(t)
        if groups and t is not None and t.requires_grad:
            scaled = joint is not None and joint.reduces(t)
            buckets.setdefault((groups, t.dtype, scaled), []).append(i)
    for (groups, _, scaled), idx in buckets.items():
        ts = (tensors[i] for i in idx)
        if scaled:
            assert joint is not None
            synced = _ScaleGrad.apply(joint.factor(groups), *ts)
        else:
            synced = _SyncGrad.apply(groups, *ts)
        for i, s in zip(idx, synced):
            out[i] = s
    return tuple(out)


def synced_parameters(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return the marked parameters of ``module`` routed through :func:`sync_grad`.

    Meant for ``torch.func.functional_call(module, synced_parameters(module),
    ...)`` around submodules whose forward jigsaw does not own (a plain
    ``torch.nn.Linear`` replicated over the spatial group, say). Inside that
    call the module sees the synced tensors, which carry no mark, so a jigsaw
    submodule that calls :func:`sync_grad` itself does not reduce twice.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters to sync.

    Returns
    -------
    dict[str, torch.Tensor]
        Synced tensor per marked parameter name; empty if none is marked or
        grad mode is off.
    """
    marked = {n: p for n, p in module.named_parameters() if replica_groups(p)}
    if not marked:
        return {}
    synced = sync_grad(*marked.values())
    return {
        n: s
        for (n, p), s in zip(marked.items(), synced)
        if s is not p and s is not None
    }


def broadcast_replicas(module: torch.nn.Module, src: int = 0) -> None:
    """Make every marked parameter of ``module`` identical over its groups.

    Each parameter is broadcast from group rank ``src`` over each of its replica
    groups in turn, which leaves it equal across their whole product.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose marked parameters to broadcast.
    src : int
        Group rank to broadcast from.
    """
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
    by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
    for t in tensors:
        by_dtype.setdefault(t.dtype, []).append(t)
    for ts in by_dtype.values():
        flat = torch.cat([t.detach().reshape(-1) for t in ts])
        dist.broadcast(flat, group_src=src, group=group)
        for t, chunk in zip(ts, flat.split([t.numel() for t in ts])):
            t.detach().copy_(chunk.view_as(t))


def wrap_ddp(
    module: torch.nn.Module,
    mesh: Mapping[str, dist.ProcessGroup],
    dp_key: str = "dp",
    joint_hook: bool = False,
    **ddp_kwargs: Any,
) -> DistributedDataParallel:
    """Wrap a jigsaw model in ``DistributedDataParallel`` over the data-parallel axis.

    Optional: ``DistributedDataParallel(module, process_group=mesh[dp_key])`` is
    checked just the same, in its first forward. This also broadcasts the
    replicas, which jigsaw modules already do when they are built, but
    parameters marked by hand may need. The marked parameters are first made identical across their replica groups
    (from group rank 0), then DDP broadcasts the whole model over ``mesh[dp_key]``
    from data-parallel rank 0 and averages the gradients over it. The result is
    checked with :func:`check_ddp`.

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


_JOINT = "_jigsaw_joint_hook_state"
#: partition of the world (tuple of sorted rank tuples) -> this rank's group in it
_JOINT_GROUPS: dict[tuple[int, tuple[tuple[int, ...], ...]], dist.ProcessGroup] = {}


class JointCommHookState:
    """State of :func:`joint_comm_hook` for one ``DistributedDataParallel``.

    For every distinct replica-group tuple ``R`` of the marked parameters DDP
    reduces, this builds the *joint* group: the ranks reachable from this one
    through DDP's group and the groups in ``R``, i.e. every rank holding a copy
    of the parameter. Unmarked parameters use DDP's group.

    Creating it is collective over the default group (it gathers the group
    layouts and may create process groups): call it on every rank. Joint groups
    are cached, so a second state over the same layout creates none.
    Constructing it switches :func:`sync_grad` to scaling inside ``ddp``'s
    forward, so the hook must then be registered on ``ddp``; :func:`sync_grad`
    raises otherwise. :func:`register_joint_comm_hook` does both.

    Parameters
    ----------
    ddp : DistributedDataParallel
        Wrapped jigsaw model; its process group is the data-parallel axis.

    Raises
    ------
    RuntimeError
        If ``ddp`` uses the Python reducer (compiled autograd), whose forward
        :func:`sync_grad` cannot detect, or if the ranks disagree on the model's
        replica groups.
    """

    def __init__(self, ddp: DistributedDataParallel) -> None:
        if getattr(ddp, "_use_python_reducer", False):
            raise RuntimeError(
                "the joint comm hook does not support DDP's Python reducer"
            )
        self.ddp = ddp
        self.dp: dist.ProcessGroup = ddp.process_group
        ignored: set[str] = getattr(ddp, "parameters_to_ignore", set())
        reduced = [
            p
            for n, p in ddp.module.named_parameters()
            if n not in ignored and p.requires_grad
        ]
        keys: dict[Groups, int] = {}
        for p in reduced:
            r = replica_groups(p)
            if r and r not in keys:
                keys[r] = len(keys)

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
            joint = _joint_group(
                [e[0] for e in everyone] + [g for e in everyone for g in e[1][k]]
            )
            self._group[r] = joint
            self._factor[r] = joint.size() / self.dp.size()
        # parameter -> joint group (DDP's group when unmarked)
        self._of: dict[int, dist.ProcessGroup] = {
            id(p): self._group.get(replica_groups(p), self.dp) for p in reduced
        }
        setattr(ddp, _JOINT, self)

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

    Collective over the default group the first time a partition is seen; the
    same ``edges`` on every rank give the same partition, so all ranks agree on
    whether to create it.
    """
    world = dist.get_world_size()
    parent = list(range(world))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ranks in edges:
        for r in ranks[1:]:
            parent[find(r)] = find(ranks[0])
    comps: dict[int, list[int]] = {}
    for r in range(world):
        comps.setdefault(find(r), []).append(r)
    partition = tuple(sorted(tuple(c) for c in comps.values()))
    if len(partition) == 1:
        return dist.group.WORLD  # type: ignore[return-value]
    key = (id(dist.group.WORLD), partition)
    if key not in _JOINT_GROUPS:
        group, _ = dist.new_subgroups_by_enumeration([list(c) for c in partition])
        assert isinstance(group, dist.ProcessGroup)  # every rank is in a part
        _JOINT_GROUPS[key] = group
    return _JOINT_GROUPS[key]


def joint_comm_hook(
    state: JointCommHookState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """DDP comm hook: average each gradient over its parameter's joint group.

    The bucket is split into runs of parameters that share a joint group, and
    each run is averaged in one all-reduce. Combined with the in-graph scaling
    that :func:`sync_grad` applies under this hook, a replicated parameter ends
    with ``AVG_dp(SUM_replica(local))``, as with the in-graph sum plus plain
    DDP. Like plain DDP it averages, so gradients accumulated with or without
    ``no_sync`` stay correct. Works under ``fp16_compress_wrapper`` and
    ``bf16_compress_wrapper``, which swap the bucket's buffer.

    Parameters
    ----------
    state : JointCommHookState
        Built for the DDP this hook is registered on.
    bucket : dist.GradBucket
        Bucket of flattened gradients.

    Returns
    -------
    torch.futures.Future[torch.Tensor]
        Resolves to the reduced bucket buffer.
    """
    buf = bucket.buffer()
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
    works = []
    for g, a, b in runs:
        seg = buf[a:b]
        seg.div_(g.size())
        works.append(dist.all_reduce(seg, group=g, async_op=True))
    last = works[-1].get_future()

    def join(_: Any) -> torch.Tensor:
        # Wait on each process group's own future: it carries NCCL's stream
        # dependency, which ``torch.futures.collect_all`` would drop.
        for w in works[:-1]:
            w.get_future().wait()
        return buf

    return last.then(join)


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

    Returns
    -------
    JointCommHookState
        The hook's state.
    """
    state = JointCommHookState(ddp)
    hook = joint_comm_hook if wrapper is None else wrapper(joint_comm_hook)
    ddp.register_comm_hook(state, hook)
    return state


def _model_groups(module: torch.nn.Module) -> list[dist.ProcessGroup]:
    """Return the groups ``module``'s parameters are sharded or replicated over.

    These are the replica groups of the marked parameters plus every process
    group held by a jigsaw module that owns parameters, in module order (the
    same on every rank) and of size > 1.
    """
    out: dict[int, dist.ProcessGroup] = {}

    def add(g: Any) -> None:
        if isinstance(g, dist.ProcessGroup) and g.size() > 1:
            out.setdefault(id(g), g)

    for m in module.modules():
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


def replica_divergence(
    params: Sequence[tuple[torch.Tensor, Sequence[dist.ProcessGroup]]],
) -> torch.Tensor:
    """Return ``max|p - p_0|`` per parameter, over each of its groups.

    ``p_0`` is the copy on group rank 0 of the respective group. The result is
    local: reduce it over the mesh before acting on it. Collective over every
    group listed, in order.

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
                ref = mine.clone()
                dist.broadcast(ref, group_src=0, group=group)
                d = (mine - ref).abs()
                # NaN on one rank only is a divergence; NaN on both is not.
                d = torch.where(mine.isnan() != ref.isnan(), torch.inf, d.nan_to_num())
                for i, chunk in zip(idx, d.split(sizes)):
                    if chunk.numel():
                        div[i] = torch.maximum(div[i], chunk.max())
    return div


def all_reduce_max(t: torch.Tensor, groups: Iterable[dist.ProcessGroup]) -> None:
    """All-reduce ``t`` with MAX over each group in turn, in place."""
    for g in groups:
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=g)


def check_ddp(ddp: DistributedDataParallel) -> None:
    """Check that ``ddp`` composes with the jigsaw model it wraps.

    Raises if DDP's process group overlaps any group the model's parameters are
    sharded or replicated over in more than the calling rank (the default group
    on a model-parallel mesh, say: DDP would then average different shards), or
    if a marked parameter differs across its replica groups (DDP only broadcasts
    over its own group, so replicas initialized differently stay different).
    Warns if a parameter still carries a gradient hook from
    :func:`jigsaw.hooks.make_grad_sync_hook`.

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
    wrong_group = any(
        dp_ranks & set(dist.get_process_group_ranks(g)) - {me} for g in groups
    )

    if any(
        any(getattr(h, HOOK_TAG, False) for h in _post_accumulate_hooks(p))
        for p in module.parameters()
    ):
        warnings.warn(
            "a DDP-wrapped jigsaw model has parameters with a make_grad_sync_hook; "
            "that hook races DDP's reducer and breaks gradient accumulation. Use "
            "jigsaw.parallel.mark_replicated and sync_grad instead.",
            stacklevel=2,
        )

    named = [(n, p, replica_groups(p)) for n, p in module.named_parameters()]
    named = [x for x in named if x[2]]
    params = [(p, g) for _, p, g in named]
    first = next(module.parameters(), None)
    device = first.device if first is not None else _device_of(dp)
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
    bad = [n for (n, _, _), d in zip(named, flags[1:].tolist()) if d > 0]
    if bad:
        raise RuntimeError(
            f"{len(bad)} replicated parameter(s) differ across their replica groups "
            f"before training, e.g. {bad[:5]}. DistributedDataParallel only "
            "broadcasts over the data-parallel axis; call "
            "jigsaw.parallel.broadcast_replicas(model) before wrapping, or build "
            "it with jigsaw.parallel.wrap_ddp(model, mesh)."
        )
    setattr(ddp, _CHECKED, True)


def _post_accumulate_hooks(p: torch.Tensor) -> Iterable[Any]:
    hooks = getattr(p, "_post_accumulate_grad_hooks", None)
    return hooks.values() if hooks else ()


def _device_of(group: dist.ProcessGroup) -> torch.device:
    if dist.get_backend(group) == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _check_active_ddp() -> None:
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


def _active_joint_state() -> Optional[JointCommHookState]:
    """Return the joint-hook state of the DDP whose forward is running, if any."""
    ddp = getattr(DistributedDataParallel, "_active_ddp_module", None)
    state = getattr(ddp, _JOINT, None) if ddp is not None else None
    if not isinstance(state, JointCommHookState):
        return None
    if not any(s is state for _, s in getattr(ddp, "_comm_hooks", ())):
        raise RuntimeError(
            "a JointCommHookState was built for this DDP but joint_comm_hook is not "
            "registered on it; without the hook the replicated gradients are never "
            "summed. Use jigsaw.parallel.register_joint_comm_hook(ddp)."
        )
    return state
