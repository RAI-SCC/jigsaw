"""In-graph gradient synchronization of replicated parameters.

A parameter replicated over a process group gets only its local share of the
gradient on each rank. Mark it with :func:`mark_replicated` and route it through
:func:`sync_grad` (or :func:`synced_call`) in the forward; the backward then sums
the shares over the group inside the autograd graph, before DDP reads ``p.grad``.

For data parallelism, wrap the model in ``DistributedDataParallel`` over
``mesh["dp"]``, or use :func:`wrap_ddp`. :func:`check_ddp` rejects a DDP group
spanning model-parallel ranks. :func:`register_joint_comm_hook` fuses the
replica sum into DDP's bucketed all-reduce, taking it off the backward's
critical path. :func:`clip_grad_norm_` clips by the global gradient norm,
counting shards and replicas once.
"""

from jigsaw._core.public import set_module as _set_module

from ._clip import clip_grad_norm_
from ._ddp import check_ddp, wrap_ddp
from ._joint import JointCommHookState, joint_comm_hook, register_joint_comm_hook
from ._replicas import broadcast_replicas, mark_replicated, replica_groups
from ._sync import sync_grad, synced_call, synced_parameters

__all__ = [
    "mark_replicated",
    "replica_groups",
    "sync_grad",
    "synced_parameters",
    "synced_call",
    "broadcast_replicas",
    "wrap_ddp",
    "JointCommHookState",
    "joint_comm_hook",
    "register_joint_comm_hook",
    "check_ddp",
    "clip_grad_norm_",
]

_set_module(globals(), __all__)
