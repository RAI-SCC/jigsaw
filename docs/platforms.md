# Supported platforms

- **Process-group backends:** `nccl` (NVIDIA NCCL, or AMD RCCL — ROCm builds of
  PyTorch register it under the same name), `gloo` (CPU tensors), or a mix such
  as `"cpu:gloo,cuda:nccl"`. `make_process_mesh` raises on any other backend
  (`mpi`, `ucc`, `xccl`, ...).
- **Devices:** `cuda` (NVIDIA, or AMD via ROCm/HIP) and `cpu`. Other device types
  raise when a module picks its kernels.
- **ROCm:** expected to work for all comm paths and the `sdpa`/`math` attention
  backends, but not covered by the test suite. For `attn_backend="te"` see
  [Attention](attention.md).
- `init_cuda()` always initialises `nccl`; on CPU call
  `dist.init_process_group("gloo")` yourself.

## Running on a cluster

An NGC PyTorch container is optional. It gives a matched CUDA/NCCL stack and a
working `transformer_engine` (for `attn_backend="te"`). The repo ships the image
recipe ([containers/README.md](../containers/README.md)) and a cluster-agnostic
SLURM/Apptainer launcher ([scripts/README.md](../scripts/README.md)). jigsaw reads
the rank layout from `srun` (`SLURM_PROCID` / `SLURM_NTASKS` / `SLURM_LOCALID`) or
`torchrun` (`RANK` / `WORLD_SIZE` / `LOCAL_RANK`) via
`jigsaw.launch.resolve_launch_env()`.
