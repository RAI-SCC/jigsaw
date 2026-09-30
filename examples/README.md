# Tutorials

Small, runnable scripts that each show one way of using _jigsaw_. Every script is a
few dozen lines of code with the explanation in its docstring, and every one needs
exactly **4 ranks**.

| Script | Shows |
|---|---|
| [`domain_parallel.py`](domain_parallel.py) | Domain parallelism only: a CNN on an image split over a 2x2 grid, weights replicated. |
| [`vit.py`](vit.py) | A Vision Transformer with domain *and* tensor parallelism: ring attention, distributed MLP, layer norm and patch embedding. |
| [`swin.py`](swin.py) | A Swin Transformer with domain parallelism: local window attention, shifted windows via `DistributedRoll`, patch merging without communication. |
| [`dataloader.py`](dataloader.py) | A dataloader for a domain-parallel model: each rank reads only its tile, and the sampler is keyed by the data-parallel index. |
| [`autoconvert.py`](autoconvert.py) | Converting a plain PyTorch model with `convert_to_jigsaw`, and checking it against the original. |
| [`reshard.py`](reshard.py) | Moving a trained model onto a different process mesh, or gathering it into plain PyTorch. |

Start with `domain_parallel.py`; the others build on the same pieces (a process mesh,
sharded inputs, a distributed loss).

## Running

With GPUs, one process per GPU, exactly as described in the [main README](../README.md):

```bash
torchrun --nproc-per-node 4 examples/vit.py
# or, on a SLURM cluster
srun --nodes=1 --ntasks-per-node=4 --gres=gpu:4 python examples/vit.py
```

Without a GPU the scripts fall back to gloo on the CPU, so you can try them on a laptop:

```bash
torchrun --nproc-per-node 4 examples/vit.py
```

They are all small enough to finish in seconds to a minute. Only rank 0 prints.

`_common.py` holds the two helpers they share: `setup()` (process-group bring-up) and
`shard()` (this rank's slice of a tensor).
