# Data parallelism (DDP) with jigsaw

This page explains how to train a jigsaw model with PyTorch's `DistributedDataParallel`
(DDP), and the two ways jigsaw reduces gradients when you do:

- **In-graph** (the default): jigsaw sums replicated gradients inside the backward pass, and
  plain DDP averages over the data-parallel axis.
- **Joint comm hook** (opt-in, faster): one DDP comm hook does both reductions at once, in
  DDP's buckets.

Both give the same gradients. Start with in-graph; switch to the joint hook when gradient
communication shows up in your step time.

## Examples

Runnable scripts in [`examples/`](../examples/README.md), each on a multiple of 4 ranks,
with 4 ranks per model copy:

| Command | Shows |
| --- | --- |
| `torchrun --nproc-per-node 8 examples/vit.py --ddp` | A ViT split over `row` x `col` (tensor and sequence parallel), plus `dp`, through `wrap_ddp` |
| `torchrun --nproc-per-node 8 examples/swin.py --ddp` | A Swin Transformer split over the image, whose plain PyTorch layers are marked by hand (see [Replicating your own layers](#replicating-your-own-layers)) |
| `torchrun --nproc-per-node 8 examples/domain_parallel.py --ddp` | A CNN with a hand-built `DistributedDataParallel(model, process_group=mesh["dp"])` |
| `torchrun --nproc-per-node 8 examples/ddp.py` | The rest of the training loop: the global batch split over `dp`, gradient accumulation with `no_sync()`, the loss to log, periodic replica checks |

Replace `--ddp` with `--joint-hook` for mode 2 (`ddp.py` uses it by default; `--plain`
switches it off). `tests/test_examples_ddp.py` checks every one of these, in every mode,
against training the same model on the whole batch without DDP.

## What has to be reduced

On a mesh like `{"dp": 2, "row": 2, "col": 2}`, a parameter is either:

- **sharded** over the model-parallel axes (`row`, `col`): each rank holds a different
  piece, e.g. a tensor-parallel weight. Its gradient only needs DDP's average over `dp`.
- **replicated** over some model-parallel axes: several ranks hold the same copy, e.g. a
  conv weight over the spatial axes, or a bias over `row`. Each rank sees only the gradient
  of its own part of the domain, so the copies' gradients must be **summed** over those
  replica groups, and then averaged over `dp`.

So the gradient of a replicated parameter `p` with replica groups `R` must end up as

```
AVG_dp( SUM_R( local gradient ) )
```

jigsaw modules mark their replicated parameters themselves with
`jigsaw.parallel.mark_replicated(p, groups)`, and route them through
`jigsaw.parallel.sync_grad` in their forward. That call is where the two modes differ.
`DistributedLayerNorm` marks its weight and bias over its `row_group`. On a mesh with
several spatial axes, add the others with `mark_replicated(p, group, extend=True)`.

## Common setup

Both modes start the same way: build a mesh with a data-parallel axis, and wrap the model in
DDP over **that axis only**.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import jigsaw

dist.init_process_group("nccl")
mesh = jigsaw.make_process_mesh(
    {"dp": -1, "row": 2, "col": 2}, model_parallel_axes=("row", "col")
)
model = build_model(mesh)  # jigsaw layers on mesh["row"] / mesh["col"]
```

jigsaw raises RuntimeError, on every rank and in the first forward (under `torch.compile`,
only in `wrap_ddp`), for the two mistakes that would otherwise train silently into a wrong
model:

- **DDP over the default group** (`DistributedDataParallel(model)`): it would average
  sharded parameters with other shards.
- **Replicas that differ at the start**: DDP only broadcasts over its own group. jigsaw
  modules make their replicas identical when they are built. For parameters you mark
  yourself, call `jigsaw.parallel.broadcast_replicas(model)` before wrapping.

`jigsaw.parallel.wrap_ddp(model, mesh)` broadcasts, wraps over `mesh["dp"]` and runs these
checks right away.

## Mode 1: in-graph sum + plain DDP (default)

```python
model = DistributedDataParallel(model, process_group=mesh["dp"])
# or: model = jigsaw.parallel.wrap_ddp(model, mesh)
```

**How it works.** `sync_grad` passes each replicated parameter through an identity autograd
function whose backward all-reduces (SUM) the gradient over the parameter's replica groups.
This happens *upstream* of the parameter's gradient accumulation, so the sum is complete
before DDP's reducer reads `p.grad`. DDP then averages over `dp` as usual.

**Properties:**

- It's ordinary DDP: `no_sync()`, gradient accumulation with or without `no_sync()`,
  `gradient_as_bucket_view`, `static_graph`, `find_unused_parameters`, and any comm hook of
  your own (fp16 compression, PowerSGD) all work.
- It works without DDP too (pure model parallelism).
- **Cost:** a replicated gradient is reduced twice: once over its replica groups in the
  backward, and again over `dp` by DDP. The first sum is a blocking collective on the
  backward's critical path. After it, every replica holds the same gradient, so DDP sends
  that gradient over `dp` once per replica.

## Mode 2: joint comm hook (faster)

```python
model = DistributedDataParallel(model, process_group=mesh["dp"])
jigsaw.parallel.register_joint_comm_hook(model)  # collective: every rank, before the first forward

# or, in one call:
model = jigsaw.parallel.wrap_ddp(model, mesh, joint_hook=True)

# with fp16 (or bf16) compression:
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
jigsaw.parallel.register_joint_comm_hook(model, default_hooks.fp16_compress_wrapper)
```

**How it works.** For each distinct set of replica groups `R`, the hook builds the *joint
group* `dp x R`: every rank that holds a copy of the parameter. Then:

1. Inside DDP's forward, `sync_grad` no longer communicates. It multiplies the gradient by
   `|R|` in the backward (no collective).
2. The DDP comm hook averages each bucket over each parameter's joint group, overlapped with
   the backward like any DDP bucket. Unmarked parameters are averaged over `dp` as usual.

Averaging `|R|` times the local gradient over `dp x R` gives
`AVG_dp(SUM_R(local))`, the same as mode 1, but in one collective that overlaps the
backward instead of blocking it.

**Properties:**

- Same gradients as mode 1. `no_sync()`, accumulation without `no_sync()`,
  `gradient_as_bucket_view` and `static_graph` keep working.
- The hook takes DDP's comm-hook slot, so it can't be combined with another hook such as
  PowerSGD. fp16/bf16 compression works through the `wrapper` argument.
- It creates at most one extra process group per distinct replica-group set, collectively
  when you register the hook. Sets that join the same ranks share a group, and none is
  created when the joint group spans every rank. With NCCL each group is a communicator that costs GPU memory,
  so check peak memory on your configuration.
- Works with DDP's Python reducer (compiled autograd,
  `torch._dynamo.config.optimize_ddp = "python_reducer"`). That reducer has no buckets: the
  hook averages each parameter on its own, compression wrappers are rejected, and the
  replicated gradients are not fused by inductor's DDP bucketing (it ignores process
  groups, so they are kept out of it). Since that reducer reduces after every backward,
  `sync_grad` scales the reduced parameters wherever they are used, not only inside DDP's
  forward.
- Every rank must build the model the same way. Registration raises RuntimeError if the
  ranks disagree on the number of distinct replica-group sets; other mismatches are not
  detected.
- Scaling only replaces the sum where it is safe: for `sync_grad` calls *inside* this DDP's
  forward (anywhere, under the Python reducer), on parameters DDP actually reduces.
  Everywhere else (parameters that were frozen or ignored when the hook was registered,
  calls outside the forward), `sync_grad` falls back to the in-graph sum, which is still
  correct.

**When it pays off.** The gain grows with the share of parameters that are replicated and
with the cost of the `dp` traffic. For a model dominated by sharded parameters, expect
little difference.

## Choosing a mode

| | In-graph (mode 1) | Joint comm hook (mode 2) |
| --- | --- | --- |
| Setup | plain DDP over `mesh["dp"]` | plus `register_joint_comm_hook` |
| Collectives per replicated gradient | sum over `R` in backward, then DDP average over `dp` | one average over `dp x R`, in DDP's buckets |
| Other DDP comm hooks (PowerSGD, ...) | yes | only compression wrappers |
| DDP's Python reducer (compiled autograd) | yes | yes, without compression wrappers |
| Extra process groups | none | at most one per distinct replica-group set |
| Gradient accumulation / `no_sync()` | yes | yes |

## Replicating your own layers

For plain PyTorch layers inside a jigsaw model whose parameters are replicated (a
`torch.nn.LayerNorm` applied to every spatial shard, say):

1. Mark the parameters where they are created:
   `jigsaw.parallel.mark_replicated(p, mesh["row"])`. A list of groups is fine; size-1
   groups are dropped.
2. In the forward, run the layer through the synced parameters:
   ```python
   out = jigsaw.parallel.synced_call(self.norm, x)
   ```
   To sync the parameters of a whole module from its own forward, use
   `jigsaw.parallel.synced_parameters(self)` with `torch.func.functional_call` and re-enter the
   forward only if it returned anything. `examples/swin.py` does this.
3. To add a replica axis of your own to parameters jigsaw modules already mark (an ensemble
   axis, say), pass `extend=True`. A group over the same ranks as one already in the mark is
   not added again, so nothing is summed twice. Marking an already marked parameter without
   `extend=True` raises RuntimeError.
4. If you mark parameters after the model is built, call
   `jigsaw.parallel.broadcast_replicas(model)` before wrapping (or use `wrap_ddp`).

Both modes work with these steps. Under the joint hook, `synced_call` scales just like
the jigsaw modules do.

## Gradient terms that are already complete on every replica

`sync_grad` assumes each replica holds a *partial* gradient that must be summed. Some loss
terms instead compute the *full* gradient identically on every replica, e.g. a regularizer
or a KL term evaluated on the parameters themselves rather than on the rank's data shard.
Summing those over `R` would scale them by `|R|`.

Read the raw parameter for such terms, not the output of `sync_grad`. Only the path through
`sync_grad` is summed (mode 1) or scaled (mode 2). The raw path is left alone, and the final
average (over `dp` in mode 1, over `dp x R` in mode 2) leaves an identical gradient unchanged.
For example, a layer can compute its forward pass from the synced parameters, while a
regularizer on the parameters reads the unsynced ones.

## Loss scaling

The replica sync sums over the shards, and DDP averages over `dp`. So the loss each rank
backpropagates must be its replica's mean loss *as seen from its shard*: either a loss that
reduces over the shards with an identity backward (`jigsaw.modules.DistributedMSELoss`), or the local mean
divided by the number of shards. `tests/test_ddp_serial.py` pins this down against serial
training.

## Checking a setup

- `check_ddp` runs automatically in the first jigsaw forward inside any DDP (except under
  `torch.compile`), and in `wrap_ddp`. It catches a wrong DDP group and replicas that differ before training.
- `jigsaw.debug.assert_replica_consistency(model, mesh)` after `optimizer.step()` finds the
  step at which replicas drift apart, e.g. from a parameter you forgot to mark. Pass
  `unmarked=True` to also compare unmarked parameters of non-jigsaw modules over every mesh
  axis.
- Replica consistency **can't** catch a wrong gradient *scale*: if every replica gets the
  same wrong gradient, they still agree. To check the scale, compare the gradient against a
  serial run of the same model on the whole batch, in float64 if you can, with SGD rather
  than Adam (Adam normalizes the gradient and hides a constant factor).
  `tests/test_ddp_serial.py` and `tests/test_ddp.py` do this for jigsaw itself, and
  `tests/test_examples_ddp.py` for the models of the examples.
