# scripts/ — container launcher

Running jigsaw in a container is **optional**. It buys you a matching
CUDA/NCCL/cuDNN stack and a working
[TransformerEngine](../jigsaw/modules/attention/attention.py), which unlocks the
`te` attention backend (`attn_backend="te"` on `DistributedAttention`, or
`--attn-backend te` in scripts that expose it). Outside a container that choice
warns and falls back to `sdpa`. `ring` attention is a separate switch and works
with every backend.

## What lives here vs. what you supply

| In the repo (portable) | You supply (per cluster) |
| --- | --- |
| `run.sh` — the invariant `apptainer exec --nv` wrapper and the launch env-var contract | The image itself (`apptainer pull` / `build`) and where it lives |
| `sites/example.env` — every knob, documented | `sites/<yourcluster>.env` — the values |
| `train.sbatch.example` — batch structure | Account, partition, walltime, node counts; the real batch script |
| `sites/horeka.env` — a working example (best-effort, not CI-guaranteed) | mpi4py built against the image's MPI; the host venv overlay |

See [`containers/README.md`](../containers/README.md) for the image provenance
and the venv-overlay / mpi4py build steps.

## Usage

```bash
# interactive, one process (login shell or an salloc allocation)
JIGSAW_SITE=horeka scripts/run.sh python -c "import transformer_engine; print('ok')"

# distributed, under srun (one task per GPU) - works both as a direct
# `srun ...` from a login shell and as an `srun ...` step inside an salloc
# allocation or an sbatch script; run.sh derives MASTER_ADDR/PORT from the
# allocation either way
JIGSAW_SITE=horeka srun --account=... --partition=accelerated-h100 \
    --nodes=1 --ntasks-per-node=4 --gres=gpu:4 \
    scripts/run.sh python -m your_script --attn-backend te

# batch
cp scripts/train.sbatch.example experiments/train.sbatch  # then edit + sbatch
```

Point at a preset with `JIGSAW_SITE=<name>` (→ `sites/<name>.env`) or an explicit
file with `JIGSAW_SITE_FILE=/abs/path.env`. With neither, export the `JIGSAW_*`
vars yourself. Presets use `${VAR:-default}`, so anything you export before
calling `run.sh` overrides the preset (e.g. `JIGSAW_SIF=/other.sif JIGSAW_SITE=horeka ...`).

## The launch contract

`run.sh` only guarantees the wrapper. jigsaw's distributed init
([`jigsaw/utils.py`](../jigsaw/utils.py)) still expects the process manager to
provide:

- `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_LOCALID` — set by `srun`
- `MASTER_ADDR`, `MASTER_PORT` — `run.sh` fills these from `SLURM_JOB_NODELIST`
  when unset

`run.sh` also sets `NVTE_FRAMEWORK=pytorch` and
`CUDA_DEVICE_MAX_CONNECTIONS=1` unless you already chose values.

Note: NGC images bake `MASTER_PORT` (and sometimes others) into the container
environment, which overrides plain host-env passthrough. `run.sh` works around
this by forcing the launch-contract vars with `apptainer --env`, which wins over
the baked values.
