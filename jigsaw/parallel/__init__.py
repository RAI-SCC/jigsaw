"""In-graph gradient synchronization of replicated parameters.

A parameter replicated over a process group (e.g. a conv weight over the spatial
grid) receives only its local share of the gradient on each rank; the shares
must be summed over the group. :func:`sync_grad` does this *inside the autograd
graph*, by routing the parameters through an identity
:class:`torch.autograd.Function` whose backward all-reduces the gradients.
Because the reduction runs upstream of the parameter's ``AccumulateGrad`` node,
unlike a ``register_post_accumulate_grad_hook``:

* it finishes before anything reads ``p.grad``, in particular DDP's reducer
  hook on the same node (an asynchronous post-accumulate hook races it);
* each backward reduces its own gradient, not the running ``p.grad`` sum, so
  gradient accumulation stays correct;
* parameters sharing a group tuple and dtype are reduced in one collective.

Usage: mark a parameter once with :func:`mark_replicated` where it is created,
and route it through :func:`sync_grad` (or :func:`synced_call` for
submodules jigsaw does not own) in the forward.

Data parallelism: wrap the model in a plain ``DistributedDataParallel`` over
the mesh's data-parallel axis (``process_group=mesh["dp"]``), or use
:func:`wrap_ddp`, which also broadcasts the replicas first. :func:`check_ddp`
rejects a DDP whose group spans model-parallel ranks, which would silently
average different shards; it runs from :func:`wrap_ddp` and from the first
jigsaw forward inside any DDP.

With plain DDP a replicated gradient is reduced twice: summed over its replica
groups in the graph, then averaged over ``dp`` by DDP.
:func:`register_joint_comm_hook` merges both into one average over the *joint*
group ``dp x replica groups`` inside DDP's buckets, with only a scaling left in
the graph. This takes the replica sum off the critical path and lets NCCL reduce
within a node before crossing nodes.
"""

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
]
