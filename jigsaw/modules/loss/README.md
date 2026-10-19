# Distributed loss functions

`jigsaw.modules.loss` provides drop-in replacements for the common
`torch.nn` / `torch.nn.functional` losses that stay correct when the input
tensor is **sharded across a process mesh**. Each loss all-reduces exactly the
terms that cross a shard boundary, so the value (and the gradient) it returns on
every rank matches what the stock PyTorch loss would produce on the fully
gathered tensor.

With no groups passed, every function/module collapses onto its `torch`
equivalent and issues **no collectives** — you can use them unconditionally.

## What's here

| Kind | Functional | Module | Mirrors |
|---|---|---|---|
| Classification | `distributed_cross_entropy` | `DistributedCrossEntropyLoss` | `F.cross_entropy` |
| | `distributed_nll_loss` | `DistributedNLLLoss` | `F.nll_loss` |
| | `distributed_binary_cross_entropy` | `DistributedBCELoss` | `F.binary_cross_entropy` |
| | `distributed_binary_cross_entropy_with_logits` | `DistributedBCEWithLogitsLoss` | `F.binary_cross_entropy_with_logits` |
| Regression | `distributed_mse_loss` | `DistributedMSELoss` | `F.mse_loss` |
| | `distributed_l1_loss` | `DistributedL1Loss` | `F.l1_loss` |
| | `distributed_smooth_l1_loss` | `DistributedSmoothL1Loss` | `F.smooth_l1_loss` |
| Shared primitives | `redundant_all_reduce` | — | — |

### Importing
```python
import jigsaw

loss_fn = jigsaw.modules.DistributedCrossEntropyLoss(...)
# or
from jigsaw.modules.functional import distributed_cross_entropy
```

## The two kinds of sharding

A loss evaluated on a process mesh sees at most two kinds of partitioning described with two arguments:

- **`channel_group`** — the process group across which the **class dimension**
  (`dim`, `1` by convention) of a classification input is split. Any reduction
  *over classes* (`log_softmax`, gathering the true-class log-probability, the
  label-smoothing term) is all-reduced over this group. Pass `None` / a size-1
  group when the class dimension is replicated. *Classification losses only.*

- **`reduce_groups`** — every process group across which the **remaining**
  (batch and spatial) dimensions are split. Only the final `"mean"` / `"sum"`
  reduction needs these; `"mean"` divides by the *global* element count.
  Accepts `None`, a single `ProcessGroup`, or a sequence of them.

> **BCE and the regression losses have no distinguished class dimension.** They
> treat every axis as an ordinary element axis, so `reduce_groups` must list
> **every** group the input is sharded over — the channel group included — and
> there is no `channel_group` argument.


## Classification: worked example

Layout: logits of shape `[N, C, S]` with the class dimension `C` sharded across
the `col` group and the spatial dimension `S` sharded across the `row` group;
`N` is not sharded. Each rank holds only its `[N, C/col, S/row]` shard.

```python
import torch
import jigsaw

mesh = ...  # {"row": ProcessGroup, "col": ProcessGroup}

ce = jigsaw.modules.DistributedCrossEntropyLoss(
    channel_group=mesh["col"],  # class dim is split here
    reduce_groups=(mesh["row"],),  # batch/spatial dims are split here
    label_smoothing=0.1,
    reduction="mean",
)

logits = ...  # [N, C_local, S_local], this rank's shard
target = ...  # [N, S_local]  class indices in [0, C_global)

loss = ce(logits, target)  # identical scalar on every rank
loss.backward()  # grads flow back into logits
```

Functional form, same semantics:

```python
from jigsaw.modules.functional import distributed_cross_entropy

loss = distributed_cross_entropy(
    logits,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    label_smoothing=0.1,
    dim=1,
    channel_group=mesh["col"],
    reduce_groups=(mesh["row"],),
)
```

### Targets

- **Class indices** — shape = `logits.shape` with `dim` removed. Supports
  `weight` (length `C_global`, replicated on every rank), `ignore_index`, and
  `label_smoothing`.
- **Class probabilities** (`cross_entropy` only) — same shape as `logits`, class
  dimension sharded the same way. Supports **neither** `weight`,
  `ignore_index`, nor `label_smoothing` (raises `NotImplementedError`).
- `label_smoothing` together with `weight` is not supported.

### NLL

`DistributedNLLLoss` / `distributed_nll_loss` take **log-probabilities**, not
logits — feed them `jigsaw.modules.functional.distributed_log_softmax(logits, dim=1,
group=channel_group)`. Same `channel_group` / `reduce_groups` / `weight` / `ignore_index`
contract as cross-entropy.

### Binary cross-entropy

```python
bce = jigsaw.modules.DistributedBCEWithLogitsLoss(
    reduce_groups=(mesh["row"], mesh["col"]),  # EVERY group the tensor is split on
    pos_weight=pos_weight,  # broadcastable to the class dim
    reduction="mean",
)
loss = bce(logits, target)  # target: soft/binary labels in [0, 1], same shard
```

`DistributedBCELoss` is the same but expects probabilities already in `[0, 1]`;
`DistributedBCEWithLogitsLoss` is the numerically stable logits form and also
accepts `pos_weight`.

## Regression: worked example

Elementwise, no class dimension — `reduce_groups` lists every group the tensor
is split on.

```python
mse = jigsaw.modules.DistributedMSELoss(reduce_groups=(mesh["row"], mesh["col"]))
loss = mse(pred, target)  # pred, target: matching shards

huber = jigsaw.modules.DistributedSmoothL1Loss(
    reduce_groups=(mesh["row"], mesh["col"]),
    beta=1.0,  # quadratic↔linear transition point
)
```

Functional forms: `distributed_mse_loss`, `distributed_l1_loss`,
`distributed_smooth_l1_loss(input, target, *, beta=1.0, reduction=..., reduce_groups=...)`.

## `reduction`

All losses take `reduction="none" | "sum" | "mean"`:

- `"none"` — returns **this rank's local per-element shard**, no collective. You
  own any further reduction.
- `"sum"` — global sum across `reduce_groups` (and, for classification, the
  class collective where relevant).
- `"mean"` — global sum divided by the global normalizer: the element count for
  the regression / BCE losses, or `Σ weight` (valid, non-ignored elements) for
  NLL / cross-entropy — matching how the corresponding `torch` loss averages.

The backward of `"sum"` / `"mean"` is an identity pass-through over
`reduce_groups` (each rank independently seeds the same upstream gradient), so
**do not** wrap the loss in any extra manual all-reduce.

## Modules

Every module subclasses `torch.nn.Module`, holds no trainable parameters
(`get_parameter_count()` returns `0`), and takes the groups plus the loss-shape
options in `__init__`; `forward(input_or_logits, target)` returns the scalar (or
the `"none"` shard). Constructor signatures:

```python
DistributedNLLLoss(channel_group=None, reduce_groups=None, *,
                   weight=None, ignore_index=-100, reduction="mean", dim=1)

DistributedCrossEntropyLoss(channel_group=None, reduce_groups=None, *,
                            weight=None, ignore_index=-100, reduction="mean",
                            label_smoothing=0.0, dim=1)

DistributedBCELoss(reduce_groups=None, *, weight=None, reduction="mean")

DistributedBCEWithLogitsLoss(reduce_groups=None, *, weight=None,
                             pos_weight=None, reduction="mean")

DistributedMSELoss(reduce_groups=None, *, reduction="mean")
DistributedL1Loss(reduce_groups=None, *, reduction="mean")
DistributedSmoothL1Loss(reduce_groups=None, *, beta=1.0, reduction="mean")
```

## Shared primitives

Exported for building your own distributed objectives:

- **`jigsaw.modules.functional.distributed_log_softmax(x, dim=-1, group=None)`** and
  **`jigsaw.modules.functional.distributed_argmax(x, dim, group=None)`** (they live in
  `jigsaw.modules.functional`) — `log_softmax` and arg-max along a dimension sharded
  across `group`. Use the first to turn sharded logits into the
  log-probabilities that `DistributedNLLLoss` expects; the second returns
  **global** indices, breaks ties to the lowest global index, and runs under
  `@torch.no_grad`.
- **`redundant_all_reduce(x, groups)`** — sum all-reduce whose result every rank
  recomputes identically, with an **identity gradient** (a second all-reduce
  would scale it by the group size). This is the primitive behind the `"sum"` /
  `"mean"` reductions.

## Mixed precision

These losses are ordinary `torch` ops plus a couple of custom autograd
functions; run them **outside** the `torch.autocast` region, exactly as with
stock PyTorch:

```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    logits = model(x)
loss = loss_fn(logits, target)  # outside autocast
loss.backward()
```
