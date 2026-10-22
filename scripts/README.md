# scripts/: container launcher

Running jigsaw in a container is **optional**. It provides a matching
CUDA/NCCL/cuDNN stack and a working
[TransformerEngine](../jigsaw/modules/attention/_attention.py), which unlocks the
`te` attention backend (`attn_backend="te"` on `DistributedAttention`, or
`--attn-backend te` in scripts that expose it). With neither `transformer_engine` nor
`flash_attn` importable, that choice warns and falls back to `sdpa`. `ring` attention is a separate switch and works
with every backend.

## What the repo ships vs. what the cluster supplies

| In the repo (portable) | Supplied per cluster |
| --- | --- |
| `run.sh`: the invariant `apptainer exec --nv` wrapper and the launch env-var contract | The image itself (`apptainer pull` / `build`) and where it lives |
| `sites/example.env`: every knob, documented | `sites/<cluster>.env`: the values |
| `train.sbatch.example`: batch structure | Account, partition, walltime, node counts; the real batch script |
| | mpi4py built against the image's MPI; the host venv overlay |

See [`containers/README.md`](../containers/README.md) for the image provenance
and the venv-overlay / mpi4py build steps.

## Usage

```bash
# interactive, one process (login shell or an salloc allocation)
JIGSAW_SITE=cluster_name scripts/run.sh python -c "import transformer_engine; print('ok')"

# distributed, under srun (one task per GPU): works both as a direct
# `srun ...` from a login shell and as an `srun ...` step inside an salloc
# allocation or an sbatch script; run.sh derives MASTER_ADDR from the
# allocation either way
JIGSAW_SITE=cluster_name srun --account=<account> --partition=<partition> \
    --nodes=1 --ntasks-per-node=4 --gres=gpu:4 \
    scripts/run.sh python -m your_script --attn-backend te

# batch
cp scripts/train.sbatch.example experiments/train.sbatch  # then edit + sbatch
```

Point at a preset with `JIGSAW_SITE=<name>` (→ `sites/<name>.env`) or an explicit
file with `JIGSAW_SITE_FILE=/abs/path.env`. With neither, export the `JIGSAW_*`
vars directly. `run.sh` sources the preset after the caller's environment, so a plain
`export` in the preset wins over a value exported before calling `run.sh`. To let such
a value override the preset, write that line as `export JIGSAW_SIF="${JIGSAW_SIF:-...}"`.

## The launch contract

`run.sh` only guarantees the wrapper. jigsaw's distributed init
([`jigsaw/launch.py`](../jigsaw/launch.py)) still expects the process manager to
provide:

- `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_LOCALID`: set by `srun` (or `RANK`,
  `WORLD_SIZE`, `LOCAL_RANK` from `torchrun`)
- `MASTER_ADDR`, `MASTER_PORT`: when unset, `run.sh` takes the first host of
  `SLURM_JOB_NODELIST` (`localhost` outside SLURM) and port 29500

`run.sh` also sets `NVTE_FRAMEWORK=pytorch` and
`CUDA_DEVICE_MAX_CONNECTIONS=1` unless they are already set.

NOTE: NGC images predefine `MASTER_PORT` (and sometimes others) in the container
environment, which overrides plain host-env passthrough. `run.sh` works around
this by forcing the launch-contract vars with `apptainer --env`, which takes
precedence over the image-defined values.
