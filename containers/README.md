# Container image

jigsaw runs on top of NVIDIA's PyTorch container. The `.sif` file lives externally. This directory documents how to rebuild it, set up the venv with jigsaw, and run code inside it.

## Base image

Base image is `nvcr.io/nvidia/pytorch:26.02-py3`
(`sha256:bbc2b67e2533edd63ff1496bb1ed00a00338cdc1478af6c1a0bf9f4b369977e7`), which ships
- Python 3.12
- torch 2.11.0a0
- transformer-engine 2.12.0
- CUDA 13
- NCCL 2.29, and
- HPC-X OpenMPI under `/opt/hpcx/ompi`.

`zarr`, `timm`, `xarray`, `h5netcdf`, `mpi4py`, and jigsaw are excluded.

Pull the container with:

```bash
apptainer pull pytorch26.02.sif docker://nvcr.io/nvidia/pytorch:26.02-py3
```

When bumping the tag, re-pull, update the digest above, and confirm
transformer-engine still exposes
`transformer_engine.pytorch.DotProductAttention`.

## Installing jigsaw with a venv overlay

Day-to-day work does not need a full image rebuild. Create a venv from inside the
container with `--system-site-packages`, on a filesystem the compute nodes can
reach:

```bash
SIF=/path/to/pytorch26.02.sif
ENV=$HOME/envs/jigsaw-te

apptainer exec --nv "$SIF" python -m venv --system-site-packages "$ENV"

# jigsaw plus the pure-Python deps the image is missing, without touching
# torch / TE / mpi4py
apptainer exec --nv "$SIF" "$ENV/bin/pip" install --no-build-isolation --no-deps -e ..
apptainer exec --nv "$SIF" "$ENV/bin/pip" install --no-build-isolation \
    zarr timm xarray h5netcdf

# mpi4py, built against the image's OpenMPI
apptainer exec --nv "$SIF" bash -c \
    "MPICC=/opt/hpcx/ompi/bin/mpicc '$ENV/bin/pip' install --no-binary=mpi4py mpi4py"
```

Then set `JIGSAW_VENV=$ENV` in the site preset.

The `--no-deps` flag on the jigsaw install stops pip from pulling a second `torch` / `torchvision` on top of the image's build.
NOTE: `requirements.txt` is only for bare-metal installs and should not be used here.

## Running code in the container

For a one-off command, call `apptainer exec` directly:

```bash
SIF=/path/to/pytorch26.02.sif
ENV=$HOME/envs/jigsaw-te

apptainer exec --nv --bind /scratch "$SIF" \
    "$ENV/bin/python" train.py --config run.yaml
```

- `--nv` exposes the GPUs
- Add `--bind` for any filesystem outside `$HOME` and `$PWD`.
- Without a venv, drop the `$ENV/bin/` prefix and pass
`--env PYTHONPATH=/path/to/jigsaw` so the checkout is importable.

For anything distributed, use `scripts/run.sh` instead of a raw `apptainer exec`.
It wraps the same call, routes `python` through `JIGSAW_VENV`, and fills in
`MASTER_ADDR` / `MASTER_PORT` from the SLURM allocation. See
[scripts/README.md](../scripts/README.md).

### Submitting a Python script with SLURM

Launch one task per GPU with `srun` inside an `sbatch` script:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=jigsaw
#SBATCH --account=hk-project-XXXX
#SBATCH --partition=accelerated-h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00

export JIGSAW_SITE=horeka          # scripts/sites/<name>.env

srun --mpi=pmix scripts/run.sh python -m your_training_module \
    --config configs/run.yaml
```

Submit it with `sbatch train.sbatch`. `run.sh` derives the rendezvous address
from the allocation; jigsaw reads `SLURM_PROCID` / `SLURM_NTASKS` /
`SLURM_LOCALID` from `srun`. A copy-ready template is at
[scripts/train.sbatch.example](../scripts/train.sbatch.example).

NOTE: keep run-specific batch scripts out of the repo, under `experiments/`
(gitignored) or elsewhere.

## Building a self-contained image

The venv overlay above is the default. For a self-contained image instead
(production runs, or sharing with people who won't run the overlay steps), see
[prebuilt-image.md](prebuilt-image.md).
