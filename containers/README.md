# Container image

jigsaw runs on top of NVIDIA's PyTorch container. The `.sif` file lives externally. This directory documents how to rebuild it and how to set up the venv with jigsaw.

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

## Baking a frozen image

For a self-contained artifact instead (production runs, or sharing with people
who won't run the overlay steps), write an `apptainer.def`:

```
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:26.02-py3

%post
    pip install --no-build-isolation zarr timm xarray h5netcdf
    MPICC=/opt/hpcx/ompi/bin/mpicc pip install --no-binary=mpi4py mpi4py
```

Build it with `apptainer build jigsaw.sif apptainer.def`. Install jigsaw itself
through the venv as above, or add `pip install -e /opt/jigsaw` to `%post` with
the repo bind-mounted.
