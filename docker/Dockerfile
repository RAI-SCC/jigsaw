# Container for running the jigsaw test suite, in particular the distributed tests,
# which need multiple ranks, NCCL and an MPI launcher.
#
# The NGC PyTorch image already ships torch, NCCL and OpenMPI built against the local
# CUDA, so none of those are reinstalled here: pulling a different torch on top is the
# usual way these images get broken.
#
# Build (from the repository root, so the build context contains the source):
#   docker build -f docker/Dockerfile -t jigsaw:test .
#
# Run the distributed tests (4 ranks on 4 GPUs):
#   docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
#       jigsaw:test ./tests/run_distributed_tests.sh -row 2 -col 2
#
# Iterate on the tests without rebuilding by mounting the working tree over /workspace:
#   docker run --rm --gpus all --ipc=host -v "$PWD":/workspace/jigsaw jigsaw:test \
#       ./tests/run_distributed_tests.sh -row 1 -col 4
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.06-py3
FROM ${BASE_IMAGE}

# mpi4py is built against whichever OpenMPI the image ships (HPC-X on recent NGC
# images), picked up via mpicc on PATH. tests/conftest.py imports mpi4py at collection
# time to derive rank and world size, so it is needed even for the serial tests.
WORKDIR /workspace/jigsaw

# Dependencies first, so edits to the source do not invalidate the cached pip layer.
#
# requirements.txt is deliberately not installed. It describes the full training and
# data stack (zarr, xarray, timm, PyYAML, ...), but jigsaw and its tests only import
# einops, mpi4py, numpy, pytest and torch. Installing it here does active harm: PyYAML
# is a Debian-managed package in the image with no RECORD file, so pip cannot uninstall
# it and the build fails outright, and numpy gets queued for replacement, which would
# swap out the build torch was compiled against. Add packages here individually if a
# test grows a new dependency.
COPY pyproject.toml README.md ./
RUN python3 -m pip install --no-cache-dir \
        "mpi4py>=4.1.1" \
        pytest \
        pytest-cov \
        einops

COPY . .

# Editable so a mounted working tree is picked up without reinstalling. --no-deps keeps
# pip from resolving the torch pin in pyproject.toml and replacing the image's build.
RUN python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -e .
