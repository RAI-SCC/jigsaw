# Jigsaw
Code for performing domain- and tensor-parallel training optimized for large image-based data.

## Structure of repository
- jigsaw/modules contains the implementations of the individual modules. Currently supported operations are ```nn.Linear, nn.Conv2d, nn.ConvTranspose2d, F.roll, Attention (with or without ring), and  non-parameterized matrix multiplications```.

## Environment
### Option A: bare metal
The library is published on PyPI as `torch-jigsaw`; the import name is `jigsaw`.
```
pip install torch-jigsaw
```
To work on the source instead, install it in editable mode:
```
pip install -e .
```
This installs `torch` and `einops`, the library's only runtime dependencies. The dev environment (necessary for running tests) can be installed with
```
pip install -e .[dev]
```

### Option B: containerized (enables TransformerEngine)
Running inside an NGC PyTorch container is optional. It gives a matched
CUDA/NCCL stack and a working `transformer_engine`, which unlocks the `te`
attention backend (`DistributedAttention(attn_backend="te")`); outside a
container that choice warns and falls back to `sdpa`.

`attn_backend` selects the attention core — `te`, `sdpa` (default, PyTorch
flash/​math SDPA) or `math` (unfused reference). `ring` (sequence-parallel
attention) is a **separate** switch that composes with any backend: `ring=True`
still runs its inner per-chunk attention through the chosen core. Ring attention
splits the sequence over the mesh rows and by default needs an unsharded channel
dimension, so the mesh columns must have size 1; without `ring`, columns split the
heads. To combine the two, pass a `RingAttentionConfig` that sends and saves the
keys and values (`send_in_forward=SendForward.KV`, `save_for_backward=SaveBackward.KV`),
as [`examples/vit.py`](examples/vit.py) does.

The container is also useful without TE — a pinned CUDA/NCCL stack for
reproducible scaling and parameter studies — in which case the defaults
(`attn_backend="sdpa"`, `ring=False`) need no flags.

The repo ships the portable parts — the image recipe
([`containers/README.md`](containers/README.md)) and a cluster-agnostic launcher
([`scripts/README.md`](scripts/README.md)). You supply the per-cluster pieces:
the image, a `scripts/sites/<cluster>.env` preset, and your batch script.

#### Launching

`scripts/run.sh <command>` wraps the command in `apptainer exec --nv`, sourcing
`scripts/sites/$JIGSAW_SITE.env` for the cluster-specific paths. The same
invocation works from a login shell, from inside an `salloc` allocation, and
from an `sbatch` script — `run.sh` derives `MASTER_ADDR` / `MASTER_PORT` from the
SLURM allocation, and jigsaw reads `SLURM_PROCID` / `SLURM_NTASKS` /
`SLURM_LOCALID` (which `srun` sets) via `resolve_launch_env()` in
[`jigsaw/utils.py`](jigsaw/utils.py). `torchrun`'s `RANK` / `WORLD_SIZE` /
`LOCAL_RANK` are accepted too.

```bash
# single process — login node or interactive allocation
JIGSAW_SITE=horeka scripts/run.sh python -c "import transformer_engine; print('ok')"

# distributed — one task per GPU. Identical whether run directly from a login
# shell or as a step inside `salloc` / `sbatch`.
JIGSAW_SITE=horeka srun --account=hk-project-XXXX --partition=accelerated-h100 \
    --nodes=2 --ntasks-per-node=4 --gres=gpu:4 --mpi=pmix \
    scripts/run.sh python -m your_script --attn-backend te

# batch — copy the template, fill in the #SBATCH directives, submit
cp scripts/train.sbatch.example experiments/train.sbatch   # gitignored
sbatch experiments/train.sbatch
```

See [`scripts/README.md`](scripts/README.md) for the full launch contract and
every `JIGSAW_*` knob.

### Sequence parallelism: ring vs. Ulysses

`DistributedAttention` offers two mutually exclusive ways to parallelise
attention over `spatial_group`, on top of `attn_backend`:

- `ring=True` — K/V circulate around `spatial_group` in a P2P ring, with a
  per-step online-softmax merge. Works for any head count.
- `ulysses=True` — a single all-to-all trades the local sequence shard for a
  head shard, runs one dense attention over the full sequence, then an
  all-to-all brings it back. Needs `heads // channel_group.size()` divisible
  by `spatial_group.size()`. Set `ulysses_gpus_per_node > 0` to run the
  *hierarchical* scheme instead — Ulysses over an intra-node sub-group nested
  in a P2P ring over the inter-node one — which keeps the all-to-all on the
  fast intra-node fabric.

Both schemes can overlap their K/V communication with the attention core on a
dedicated CUDA stream: `overlap_comm` on `RingAttentionConfig` for the plain
ring, `ulysses_inter_overlap` for the hierarchical scheme's inter-node ring.
CUDA only; inert elsewhere. See `RingAttentionConfig.__init__` and
`DistributedAttention.__init__` for the full parameter reference.

## Usage
_jigsaw_ follows the interface of PyTorch: build the model from _jigsaw_ layers, then train it as usual. Each layer takes the *global* (unsharded) dimensions and the process groups of a [process mesh](#process-mesh), and allocates only its local shard.

```python
import torch
import jigsaw
from jigsaw.utils import init_cuda, set_all_seeds

device, _, rank, world_size = init_cuda()  # NCCL init from the SLURM / torchrun env
mesh = jigsaw.make_process_mesh({"row": 2, "col": 2})  # 4 ranks, one model replica


class Linear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, mesh, device):
        super().__init__()
        self.lin = jigsaw.linear.DistributedXWT(
            in_dim=in_dim,
            out_dim=out_dim,
            bias=True,
            local_rank=mesh["model"].rank(),
            device=device,
            model_group=mesh["model"],
            column_group=mesh["col"],
            row_group=mesh["row"],
        )

    def forward(self, x):
        return self.lin(x)  # x is this rank's shard, split over "col" in its last dim


# Give every model-parallel rank its own seed so the shards are initialized independently.
set_all_seeds(mesh["model"].rank())
model = Linear(1024, 1024, mesh, device)
```

Run it with one process per GPU, e.g. `torchrun --nproc-per-node 4 train.py` or `srun` (see [Launching](#launching)).

Complete, runnable scripts -- a CNN, a Vision Transformer, a Swin Transformer, a dataloader, converting an existing model, re-sharding a trained one -- are in [`examples/`](examples/README.md). Each runs with `torchrun --nproc-per-node 4`, on GPUs or on the CPU.

### Mixed precision (autocast)

Wrap the forward in `torch.autocast` and run `backward()` outside it, as with plain PyTorch:

```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    y = model(x)
loss = loss_fn(y, target)
loss.backward()  # outside the autocast region
```

_jigsaw_'s layers are custom `torch.autograd.Function`s, which `torch.autocast` does **not**
descend into automatically, so each opts in by hand. Behaviour:

- **Matmul family** — `DistributedXWT`, attention (plain + ring), `DistributedConv*` /
  `DistributedConv*Transpose`: the forward runs in the autocast dtype and the output carries
  it (bias included); gradients are reconciled to the **fp32 (parameter) dtype** in backward,
  i.e. you keep fp32 master gradients for free.
- **`DistributedLayerNorm` (with or without `elementwise_affine`) / `DistributedSoftMax`**
  follow stock-AMP "fp32 list" semantics: computed and emitted in fp32 even when fed bf16/fp16.
- **Attention core (SDPA / TransformerEngine)**: under autocast it uses the autocast dtype
  (so `autocast(float16)` stays fp16); with no autocast it uses bf16 on CUDA — keeping the
  flash kernel eligible, which rejects fp32 — and the input dtype on CPU (SDPA math backend).

Only `float16` and `bfloat16` are autocast dtypes. `float64` params work for the matmul,
conv and norm layers but not attention (it force-casts q/k/v). `float8` is not supported by
any hand-rolled layer — it needs TransformerEngine's own `fp8_autocast` path.

### Process mesh
Each module accepts a ```process_mesh_dict``` -- a mapping ``{axis name -> ProcessGroup}``. The last two axes represent the DTP (default kw ``row`` and ``col``); any leading axes are data-parallel replicas. On top of the per-axis groups the dict carries a ``model`` group spanning one full parameter replica, which the parameter-free matmul, the weight-merging converters and seed setup all read.

Build one with `jigsaw.make_process_mesh`, which takes an ordered ``{name: size}`` mapping (one axis may be `-1`, inferred from the world size) and adds the flattened ``model`` group:

```python
import torch.distributed as dist
import jigsaw

dist.init_process_group("nccl")

# fully domain--tensor parallel: (4, 2) shards params across all 8 ranks
mesh = jigsaw.make_process_mesh({"row": 4, "col": 2})

# data parallel over `dp`, DTP over row x col; `model` spans one row x col replica
mesh = jigsaw.make_process_mesh(
    {"dp": -1, "row": 2, "col": 2}, model_parallel_axes=("row", "col")
)

# convolution: split the domain over several spatial axes plus a channel axis
mesh = jigsaw.make_process_mesh({"sh": 2, "sw": 2, "col": 2})
```

There are two options
1) Training the model in a fully domain--tensor parallel manner. This means that a process mesh of size (4, 2) will have the parameters sharded across 8 processes.
2) Training the model with domain-parallelism across the ``rows`` axis, and with domain--tensor parallelism across the ``col`` axis. See the individual modules for detailed usage.

### Data parallelism (DDP)

The full guide, including when to use the joint comm hook instead of the default in-graph
sync, is in [docs/ddp.md](docs/ddp.md).

Use plain `DistributedDataParallel` over the `dp` axis:

```python
from torch.nn.parallel import DistributedDataParallel

mesh = jigsaw.make_process_mesh(
    {"dp": -1, "row": 2, "col": 2}, model_parallel_axes=("row", "col")
)
model = build_model(mesh)  # jigsaw layers on mesh["row"] / mesh["col"]
model = DistributedDataParallel(model, process_group=mesh["dp"])
```

Everything after that is ordinary DDP: `no_sync()`, `gradient_as_bucket_view`,
`find_unused_parameters`, `static_graph` and your own comm hook (fp16 compression, say) all
work. `make_process_mesh` also returns the n-d `DeviceMesh` it builds, so
`DistributedDataParallel(model, device_mesh=mesh.device_mesh["dp"])` works too; it goes
through DDP's DTensor code path, so `process_group=` is the default we recommend.

Naming the `dp` axis is the one thing you have to do. Two mistakes would otherwise train
without an error and give a wrong model, so jigsaw raises on them, on every rank, in the
first forward pass:

- **DDP over the default group** (`DistributedDataParallel(model)`): on a model-parallel
  mesh it averages sharded parameters with other shards.
- **Replicas that differ** at the start: DDP only broadcasts over its own group. jigsaw
  modules make their replicas identical when they are built; for parameters you mark
  yourself (below), call `jigsaw.parallel.broadcast_replicas(model)` before wrapping.

`jigsaw.parallel.wrap_ddp(model, mesh)` does both steps, `broadcast_replicas` and
`DistributedDataParallel(model, process_group=mesh["dp"])`, and runs the check right away.

Parameters that are replicated over model-parallel ranks (biases over `row`, convolution
weights over the spatial axes, ...) have their gradients summed over those ranks inside the
backward pass (`jigsaw.parallel`). That sum is done before DDP's reducer reads the gradient,
so it composes with DDP, `no_sync()` and gradient accumulation. To replicate the parameters of
your own plain PyTorch layers, mark them with `jigsaw.parallel.mark_replicated(p, group)` and
run them through `jigsaw.parallel.synced_parameters` (see `examples/swin.py`). Don't register
`jigsaw.hooks.make_grad_sync_hook` on them for this: it races DDP and breaks gradient
accumulation. To add a replica axis of your own (an ensemble axis, say) to parameters that
jigsaw modules already mark, pass `extend=True`; a group over the same ranks as one already in
the mark is not added again.

#### Joint comm hook (faster gradient sync for replicated parameters)

With plain DDP a replicated gradient is reduced twice: summed over its model-parallel
replicas inside the backward pass, then averaged over `dp` by DDP. Since every replica then
holds the same gradient, the `dp` all-reduce sends it over the inter-node links once per
replica. The joint comm hook replaces both with a single DDP comm hook that averages each
gradient over the *joint* group (`dp` x its replica groups) in DDP's buckets, overlapped with
the backward pass:

```python
model = DistributedDataParallel(model, process_group=mesh["dp"])
jigsaw.parallel.register_joint_comm_hook(model)  # collective: call on every rank

# with fp16 (or bf16) compression on top:
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
jigsaw.parallel.register_joint_comm_hook(model, default_hooks.fp16_compress_wrapper)
```

or `jigsaw.parallel.wrap_ddp(model, mesh, joint_hook=True)`. Gradients are the same as
without it, and `no_sync()`, accumulating without `no_sync()`, `gradient_as_bucket_view` and
`static_graph` all keep working. The hook takes DDP's comm-hook slot, so it cannot be combined
with PowerSGD. It creates one extra process group per distinct set of replica groups. The gain
grows with the share of parameters that are replicated: on 8 A100s over two nodes (`dp=2`,
2x2 spatial split), a 14M-parameter conv net trained about 8% faster.

To find the step at which replicas drift apart, call
`jigsaw.debug.assert_replica_consistency(model, mesh)` after `optimizer.step()`. It is a
collective and raises on every rank, naming the parameters that differ. It only knows the
groups a mark names, so a plain layer you forgot to mark is invisible to it; pass
`unmarked=True` to also compare every unmarked parameter of a non-jigsaw module over every
mesh axis.

### Distributed logging

`jigsaw.log` is a rank-aware wrapper over the stdlib `logging` module. `init_cuda()`
calls `log.configure_logging()` for you; call it yourself (once, after the launcher
env is known) only if you want to log before `init_cuda` or override the defaults.

```python
from jigsaw import log

log.info("world_size=%d", world)  # emitted by the print rank only (default: rank 0)
log.info_all("process group ready")  # every rank
log.at(2).debug("q shard %s", q.shape)  # only rank 2 (accepts an int, a list, or "all")
```

Every record is stamped with `[rank/world]` and an elapsed-time marker. Other helpers:

- `log.assert_all_agree("seed", seed)` — collective check that a value matches across ranks.
- `with log.timed("forward", barrier=True, cuda_sync=True): ...` — time a block, log it from the print rank.
- `log.log_ranks("loss=%.4f", loss)` — gather one line per rank and print them in order from rank 0.
- `log.die("bad shard %s", shape)` — log on the calling rank, tear down the process group, exit (no hang).
- `configure_logging()` also installs `faulthandler` (`kill -USR1 <pid>` dumps stacks) and a rank-aware
  `excepthook` so a crashing rank fails the job instead of deadlocking its peers.

Configured via arguments to `configure_logging()` or environment variables (arguments win):
`JIGSAW_LOG_LEVEL` (default `INFO`), `JIGSAW_LOG_RANK` (default print rank `0`),
`JIGSAW_LOG_DIR` (write per-rank `rank_<NNNN>.log` files; default off),
`JIGSAW_LOG_FORMAT` (`json` for one JSON object per line).

### Dataloader
As domain parallelism is applied throughout the model training process, the dataloader needs to be written according.
For a model-parallel group:
```
set_all_seeds(0) # constant number for all instances of the model-parallel group
dataloader = torch.utils.data.DataLoader(...) # Define the dataloader here
# The Dataset must return the correct shard of the data samples.
```
