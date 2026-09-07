# Jigsaw
Code for performing domain- and tensor-parallel training optimized for large image-based data.

## Structure of repository
- jigsaw/modules contains the implementations of the individual modules. Currently supported operations are ```nn.Linear, nn.Conv2d, nn.ConvTranspose2d, F.roll, Attention (with or without ring), and  non-parameterized matrix multiplications```.

## Environment
This is an installable library. Requirements can be installed with:
```
pip install -e .
```
The dev environment (necessary for running tests) can be installed with
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
still runs its inner per-chunk attention through the chosen core.

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

## Usage
_jigsaw_ is designed to follow the interface of PyTorch. The model must be defined exclusively via _jigsaw_ layers. i.e.
```
class Linear(torch.nn.Module):
   def __init__(self, global_in_dim, global_out_dim, process_mesh_dict):
      self.lin = jigsaw.DistributedXWT(global_in_dim, global_out_dim, process_mesh_dict)
   def forward(self, x):
      return self.lin(x)
```
When initializing the models, ensure that each model-parallel rank has a different random seed, thus insuring true randomness in the initialization. The DP wrapper will handle weight initialization synchronization across data-parallel ranks.
```
local_rank = model_group.rank()
set_all_seeds(local_rank)
model = Linear(global_in_dim, global_out_dim, process_mesh_dict)
```

### Mixed precision (autocast)

Wrap the forward in `torch.autocast` and run `backward()` outside it, as with plain PyTorch:

```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    y = model(x)
loss = loss_fn(y, target)
loss.backward()   # outside the autocast region
```

_jigsaw_'s layers are custom `torch.autograd.Function`s, which `torch.autocast` does **not**
descend into automatically, so each opts in by hand. Behaviour:

- **Matmul family** — `DistributedXWT`, attention (plain + ring), `DistributedConv*` /
  `DistributedConv*Transpose`: the forward runs in the autocast dtype and the output carries
  it (bias included); gradients are reconciled to the **fp32 (parameter) dtype** in backward,
  i.e. you keep fp32 master gradients for free.
- **`DistributedLayerNorm` / `DistributedNoParamLayerNorm` / `DistributedSoftMax`** follow
  stock-AMP "fp32 list" semantics: computed and emitted in fp32 even when fed bf16/fp16.
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

### Distributed logging

`jigsaw.log` is a rank-aware wrapper over the stdlib `logging` module. `init_cuda()`
calls `log.configure_logging()` for you; call it yourself (once, after the launcher
env is known) only if you want to log before `init_cuda` or override the defaults.

```python
from jigsaw import log

log.info("world_size=%d", world)   # emitted by the print rank only (default: rank 0)
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
dataloader = torch.nn.DataLoader(...) # Define the dataloader here
# The Dataset must return the correct shard of the data samples.
```
