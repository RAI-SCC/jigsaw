# Jigsaw

Domain- and tensor-parallel training in PyTorch, built for large image-based data.
Swap your layers for _jigsaw_ layers, give them a process mesh, and train as usual:
each rank holds only its tile of the data and its shard of the weights.

## Installation

```bash
pip install torch-jigsaw        # import name: jigsaw
pip install -e .[dev]           # from a checkout, with the test dependencies
```

The only runtime dependencies are `torch` and `einops`.

## Quickstart

Each layer takes the *global* (unsharded) sizes and the process groups of a mesh, and
works on this rank's shard of the input. Training is plain PyTorch:

```python
import torch
import jigsaw
from jigsaw.launch import init_cuda

device, _, rank, world_size = init_cuda()  # NCCL, from the SLURM / torchrun env
mesh = jigsaw.make_process_mesh({"sh": 2, "sw": 2, "col": 1})  # 4 ranks: a 2x2 grid of tiles
spatial = (mesh["sh"], mesh["sw"])  # groups splitting (H, W)


def conv(cin, cout):
    return jigsaw.modules.DistributedConv2d(
        cin, cout, kernel_size=3, padding=1, device=device,
        spatial_group=spatial,       # exchanges halos with the neighbouring tiles
        channel_group=mesh["col"],   # size 1: channels not split
    )


model = torch.nn.Sequential(conv(3, 16), torch.nn.GELU(), conv(16, 3))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = jigsaw.modules.DistributedMSELoss(reduce_groups=spatial)  # loss over the full image

for x, target in loader:  # each rank loads its 256x256 tile of a 512x512 image
    loss = loss_fn(model(x), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Launch one process per GPU: `torchrun --nproc-per-node 4 train.py`, or `srun`.

### Converting an existing model

`jigsaw.convert.convert_to_jigsaw` swaps every layer it knows (`nn.Linear`, `nn.Conv*`,
`nn.ConvTranspose*`, `nn.LayerNorm`, `jigsaw.modules.Attention`, `jigsaw.modules.Roll`)
for its distributed version and keeps each rank's shard of the weights. For example, a
small transformer with the tokens split over `row` and the heads/channels over `col`:

```python
from torch import nn
from jigsaw.convert import convert_to_jigsaw


class Block(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = jigsaw.modules.Attention(
            in_dim=dim, embed_dim=dim, out_dim=dim, heads=heads, device=device
        )
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):  # x: (batch, tokens, dim)
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


mesh = jigsaw.make_process_mesh({"row": 2, "col": 2})
torch.manual_seed(0)  # every rank converts the same full model
model = convert_to_jigsaw(nn.Sequential(Block(), Block()), mesh, device)  # ring attention over "row"

x = torch.randn(8, 1024 // 2, 64 // 2, device=device)  # this rank's (tokens, channels) shard
y = model(x)
```

[`examples/autoconvert.py`](examples/autoconvert.py) also checks the converted model
against the original.

## Tutorials

[`examples/`](examples/README.md) contains short, runnable scripts. Each needs 4 ranks
and runs on GPUs or on the CPU (gloo) with the same command:

| Script | Shows |
|---|---|
| [`domain_parallel.py`](examples/domain_parallel.py) | Domain parallelism only: a CNN on an image split over a 2x2 grid, weights replicated. **Start here.** |
| [`vit.py`](examples/vit.py) | A Vision Transformer with domain *and* tensor parallelism. |
| [`swin.py`](examples/swin.py) | A Swin Transformer: window attention, shifted windows via `DistributedRoll`. |
| [`ddp.py`](examples/ddp.py) | A data-parallel training loop on top of the model-parallel ViT. |
| [`dataloader.py`](examples/dataloader.py) | A dataloader where each rank reads only its tile. |
| [`autoconvert.py`](examples/autoconvert.py) | Converting a plain PyTorch model with `convert_to_jigsaw`. |
| [`reshard.py`](examples/reshard.py) | Moving a trained model to another mesh, or gathering it into plain PyTorch. |

```bash
torchrun --nproc-per-node 4 examples/domain_parallel.py
torchrun --nproc-per-node 8 examples/vit.py --ddp   # 2 data-parallel copies
```

## Layers

Linear (`DistributedLinear`, parameter-free matmuls), convolution and transposed
convolution (1d/2d/3d), attention (head-parallel over the channel group, optionally
combined with ring or Ulysses sequence parallelism),
`LayerNorm`, `GroupNorm`, `Softmax`, `Roll`, losses (cross-entropy, NLL, BCE, MSE, L1,
smooth L1) and segmentation metrics. All live in `jigsaw.modules`.

## Process mesh

`jigsaw.make_process_mesh` takes an ordered `{axis name: size}` mapping (one size may be
`-1`) and returns a `{name: ProcessGroup}` dict, plus a `model` group spanning one full
model copy. The trailing axes split the model; any leading axes hold data-parallel copies.

```python
mesh = jigsaw.make_process_mesh({"row": 4, "col": 2})                    # 8 ranks, one model
mesh = jigsaw.make_process_mesh({"dp": -1, "row": 2, "col": 2},
                                model_parallel_axes=("row", "col"))      # + data parallelism
```

For data parallelism, wrap the model with `jigsaw.parallel.wrap_ddp(model, mesh)`, which
applies DDP over `mesh["dp"]`. See [docs/ddp.md](docs/ddp.md).

## Further reading

- [Data parallelism](docs/ddp.md): DDP over a model-parallel mesh, the joint comm hook, replica checks.
- [Attention](docs/attention.md): attention backends (`sdpa`, `math`, TransformerEngine), ring vs. Ulysses.
- [Mixed precision](docs/mixed_precision.md): `torch.autocast` support.
- [Logging](docs/logging.md): rank-aware logging and debugging helpers (`jigsaw.log`).
- [Platforms](docs/platforms.md): supported backends and devices, running on a cluster.
- [Containers](containers/README.md) and the [SLURM launcher](scripts/README.md).

## Running tests

Tests run under MPI. A test marked `@pytest.mark.dist(N)` only runs at exactly `N` ranks:

```bash
mpirun -n 4 pytest tests --device cpu    # gloo
mpirun -n 4 pytest tests --device cuda   # nccl, one rank per GPU
```

CI runs the CPU suite on 1, 4 and 8 ranks. GPU tests are not run in CI.

## Citation

If you use _jigsaw_, please cite it as described in [CITATION.cff](CITATION.cff).
