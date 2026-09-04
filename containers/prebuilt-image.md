# Building a self-contained image

The venv overlay (see [README.md](README.md)) is the default. Build the deps
into the image itself for production runs, or to share with people who won't run
the overlay steps.

Write an `apptainer.def`:

```
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:26.02-py3

%post
    pip install --no-build-isolation zarr timm xarray h5netcdf
    MPICC=/opt/hpcx/ompi/bin/mpicc pip install --no-binary=mpi4py mpi4py
```

Build with:

```bash
apptainer build jigsaw.sif apptainer.def
```

jigsaw itself still goes through the venv. Alternatively, add
`pip install -e /opt/jigsaw` to `%post` with the repo bind-mounted.
