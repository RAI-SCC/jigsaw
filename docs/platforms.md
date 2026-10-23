# Platforms and limitations

## Supported platforms

- **Process-group backends:** `nccl` (NVIDIA NCCL, or AMD RCCL, which ROCm builds of
  PyTorch register under the same name), `gloo` (CPU tensors), or a mix such
  as `"cpu:gloo,cuda:nccl"`. `make_process_mesh` raises on any other backend
  (`mpi`, `ucc`, `xccl`, ...).
- **Devices:** `cuda` (NVIDIA, or AMD via ROCm/HIP) and `cpu`. Other device types
  (TPU/XLA, NPU, XPU, MPS) are not supported and raise when a module picks its
  kernels.
- **ROCm:** tested on AMD MI250X with RCCL (see below), using the `sdpa`/`math`
  attention backends. For `attn_backend="te"` see [Attention](attention.md).
- **Versions:** Python 3.11 or newer, PyTorch 2.8 or newer.
- `init_cuda()` always initializes `nccl`; on CPU call
  `dist.init_process_group("gloo")` yourself.

## Tested configurations

- **Architectures:** Vision Transformers (ring attention, distributed MLP, layer norm
  and patch embedding), Swin Transformers (local window attention, shifted windows),
  and CNNs with domain parallelism. See [examples/](../examples/README.md). At scale:
  BEAST, a Bayesian Swin Transformer for global weather forecasting on ERA5 at 0.25°
  (84 input variables, attention windows of up to 64K tokens), with models of 0.7B
  to 2.4B parameters ([arXiv:2609.12815](https://arxiv.org/abs/2609.12815)).
- **Scale:** up to 5,120 nodes (20,480 NVIDIA GH200 GPUs) on JUPITER and 1,024 nodes
  (4,096 AMD MI250X, 8,192 GCDs) on LUMI. jigsaw ran the domain-tensor mesh of 8 GPUs
  (`[4, 2]`); the rest of the scale came from data- and uncertainty-parallel copies
  over DDP.
- **Software stacks:** NCCL 2.29.7, PyTorch 2.12.0, CUDA 13.2 (JUPITER); RCCL 2.22.5,
  PyTorch 2.7.1, ROCm 6.2.4 (LUMI); Python 3.12, in Apptainer / SingularityCE
  containers.
- **Test suite:** CI runs the CPU suite over `gloo` on 1, 4 and 8 ranks. GPU tests
  (NCCL, RCCL) and the TransformerEngine backend are run manually, not in CI.
- **Launchers:** large runs have used SLURM (`srun`) with Apptainer, so the cluster
  documentation and scripts are SLURM-specific. `torchrun` is supported and used for
  the examples, but has not been exercised at scale.

## Known limitations

- **Layer coverage.** Distributed equivalents exist for linear and matmul, convolution
  and transposed convolution (1D to 3D), attention, layer and group norm, softmax, roll,
  and the losses and metrics in `jigsaw.modules`. There is none for batch, instance or
  RMS norm, pooling, padding, upsampling, flatten and pixel shuffle, embeddings,
  recurrent layers, or `torch.nn.MultiheadAttention` / `torch.nn.Transformer*`.
  `convert_to_jigsaw` raises `ValueError` on these unless `non_safe=True`.
- **Attention.** No attention bias, GQA, or FP8. See [Attention](attention.md) for the
  per-backend limits.
- **Precision.** Only `float16` and `bfloat16` autocast. `float64` works for matmul,
  convolution and normalization layers but not attention; FP8 is not supported. See
  [Mixed precision](mixed_precision.md).

## Running on a cluster

An NGC PyTorch container is optional. It gives a matched CUDA/NCCL stack and a
working `transformer_engine` (for `attn_backend="te"`). The repo ships the image
recipe ([containers/README.md](../containers/README.md)) and a cluster-agnostic
SLURM/Apptainer launcher ([scripts/README.md](../scripts/README.md)). jigsaw reads
the rank layout from `srun` (`SLURM_PROCID` / `SLURM_NTASKS` / `SLURM_LOCALID`) or
`torchrun` (`RANK` / `WORLD_SIZE` / `LOCAL_RANK`) via
`jigsaw.launch.resolve_launch_env()`.
