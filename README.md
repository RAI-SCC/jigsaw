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

### Process mesh
Each module accepts a ```process_mesh_dict```, where the last two dimensions represent the DTP. Default kw for the dictionary are ``row`` and ``col``. There are two options
1) Training the model in a fully domain--tensor parallel manner. This means that a process mesh of size (4, 2) will have the parameters sharded across 8 processes.
2) Training the model with domain-parallelism across the ``rows`` axis, and with domain--tensor parallelism across the ``col`` axis. See the individual modules for detailed usage.

### Dataloader
As domain parallelism is applied throughout the model training process, the dataloader needs to be written according.
For a model-parallel group:
```
set_all_seeds(0) # constant number for all instances of the model-parallel group
dataloader = torch.nn.DataLoader(...) # Define the dataloader here
# The Dataset must return the correct shard of the data samples.
```
