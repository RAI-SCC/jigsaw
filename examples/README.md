# Tutorials

Small, runnable scripts that each show one way of using _jigsaw_. Every script is a
few dozen lines of code with the explanation in its docstring, and every one needs
exactly **4 ranks** (`domain_parallel.py --ddp`: a multiple of 4).

| Script | Shows |
|---|---|
| [`domain_parallel.py`](domain_parallel.py) | Domain parallelism only: a CNN on an image split over a 2x2 grid, weights replicated. With `--ddp` (8 ranks), data parallelism on top via `DistributedDataParallel`. |
| [`vit.py`](vit.py) | A Vision Transformer with domain *and* tensor parallelism: ring attention, distributed MLP, layer norm and patch embedding. |
| [`swin.py`](swin.py) | A Swin Transformer with domain parallelism: local window attention, shifted windows via `DistributedRoll`, patch merging without communication. |
| [`dataloader.py`](dataloader.py) | A dataloader for a domain-parallel model: each rank reads only its tile, and the sampler is keyed by the data-parallel index. |
| [`autoconvert.py`](autoconvert.py) | Converting a plain PyTorch model with `convert_to_jigsaw`, and checking it against the original. |
| [`reshard.py`](reshard.py) | Moving a trained model onto a different process mesh, or gathering it into plain PyTorch. |

Start with `domain_parallel.py`; the others build on the same pieces (a process mesh,
sharded inputs, a distributed loss).

## Running

Install the package (`pip install torch-jigsaw`, or `pip install -e .` from a checkout) and
start each script with **4 processes**:

```bash
torchrun --nproc-per-node 4 examples/vit.py
```

With one GPU per process this is what you want; on a SLURM cluster use `srun` instead, as
described in the [main README](../README.md):

```bash
srun --nodes=1 --ntasks-per-node=4 --gres=gpu:4 python examples/vit.py
```

If the node has fewer GPUs than processes -- or none, as on a laptop -- the scripts
run on the CPU with gloo, with the very same command. They are small enough to finish in
seconds to a minute either way.

A few things that are good to know:

* Running `python examples/vit.py` on its own stops with
  `needs exactly 4 ranks but the process group has 1`: launch it as above.
* Only rank 0 prints. (On the CPU, gloo also prints a few `[Gloo] Rank ... is connected`
  lines from every rank; they are harmless.)
* jigsaw looks at `SLURM_PROCID` before `RANK`. If it is set in your shell -- as it is
  inside an `sbatch` script -- start the scripts with `srun`, not with `torchrun`.

### What success looks like

| Script | Last lines of output |
|---|---|
| `domain_parallel.py`, `vit.py`, `swin.py` | the loss falls over the 100 steps |
| `autoconvert.py` | `forward max \|converted - original\|` and the gradient error around `1e-6` or below |
| `reshard.py` | `max diff` around `1e-6` or below, twice |
| `dataloader.py` | `all checks passed` |

The last three check themselves with `assert`s, so a wrong result is a traceback, not just a
number. `tests/test_examples.py` runs all six on every commit, and compares the models of the
first three with their undistributed equivalents (values, loss and gradients).

`_common.py` holds the two helpers they share: `setup()` (process-group bring-up) and
`shard()` (this rank's slice of a tensor).
