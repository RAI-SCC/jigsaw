# Container image

The `.sif` is **not** in the repo (multi-GB, per-cluster storage). This file is
the recipe for reproducing it and layering jigsaw on top.

## Upstream image

| | |
| --- | --- |
| Base | `nvcr.io/nvidia/pytorch:26.02-py3` |
| Digest | `sha256:bbc2b67e2533edd63ff1496bb1ed00a00338cdc1478af6c1a0bf9f4b369977e7` |
| Provides | Python 3.12, torch 2.11.0a0, **transformer-engine 2.12.0**, CUDA 13, NCCL 2.29, HPC-X OpenMPI at `/opt/hpcx/ompi` |
| Missing for jigsaw | `zarr`, `timm`, `xarray`, `h5netcdf`, `mpi4py`, and jigsaw itself |

```bash
apptainer pull pytorch26.02.sif docker://nvcr.io/nvidia/pytorch:26.02-py3
```

Pin a new version deliberately: bump the tag here, re-pull, update the digest,
and check TE still exposes `transformer_engine.pytorch.DotProductAttention`.

## Layering jigsaw: host venv overlay

Don't rebuild the image for iteration. Create a venv *from inside the image*
with `--system-site-packages`, on a filesystem the compute nodes can see:

```bash
SIF=/path/to/pytorch26.02.sif
ENV=$HOME/envs/jigsaw-te

apptainer exec --nv "$SIF" python -m venv --system-site-packages "$ENV"

# jigsaw + the pure-Python deps the image lacks, WITHOUT touching torch/TE/mpi4py
apptainer exec --nv "$SIF" "$ENV/bin/pip" install --no-build-isolation --no-deps -e ..
apptainer exec --nv "$SIF" "$ENV/bin/pip" install --no-build-isolation \
    zarr timm xarray h5netcdf

# mpi4py, built against the image's OpenMPI
apptainer exec --nv "$SIF" bash -c \
    "MPICC=/opt/hpcx/ompi/bin/mpicc '$ENV/bin/pip' install --no-binary=mpi4py mpi4py"
```

Then set `JIGSAW_VENV=$ENV` in your site preset. `--no-deps` on the jigsaw
install is what keeps pip from pulling a second `torch` / `torchvision` over the
image's build — `requirements.txt` is for bare-metal installs only, do not use
it here.

Rebuild the venv whenever you change the image.

## When to bake instead of overlay

If you want a frozen artifact (production runs, sharing with people who won't run
the overlay steps), write an `apptainer.def`:

```
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:26.02-py3

%post
    pip install --no-build-isolation zarr timm xarray h5netcdf
    MPICC=/opt/hpcx/ompi/bin/mpicc pip install --no-binary=mpi4py mpi4py
```

Build with `apptainer build jigsaw.sif apptainer.def` and still install jigsaw
itself via the venv (or add `pip install -e /opt/jigsaw` with the repo bind-mounted).
