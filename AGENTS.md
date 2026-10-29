# AGENTS.md

## Project

Jigsaw (`torch-jigsaw`, import name `jigsaw`) is a domain- and tensor-parallel
training library for PyTorch, aimed at large image-based data. Users swap standard
layers for `Distributed*` layers (or run `jigsaw.convert.convert_to_jigsaw`), pass the
process groups of a mesh, and each rank holds only its tile of the data and its shard
of the weights. Results must match the non-distributed torch model.

## Layout

```
jigsaw/              the library
  _core/             comm helpers, autocast, buffers, vmap, offload, public-API labeling
  modules/           Distributed* layers: linear, convolution, attention, normalization,
                     activation, loss, metrics, roll, permute; each with _-prefixed internals
  parallel/          data-parallel wrappers (DDP, grad clipping, replica sync)
  convert.py         torch model -> jigsaw model
  mesh.py            process mesh construction
  init.py            shard-aware torch.nn.init
  launch.py, log.py, gather.py, debug.py
tests/               pytest suite, run under mpirun; parity.py holds the comparison helpers
examples/            runnable example scripts (vit, swin, ddp, reshard, ...)
docs/                Sphinx docs
containers/          GPU container (NVIDIA PyTorch + TransformerEngine) setup
scripts/             container launcher and SLURM templates
```

## Setup and tests

CPU:

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev]"            # needs a system OpenMPI for mpi4py
.venv/bin/pre-commit install
mpirun -n 4 .venv/bin/pytest tests --device cpu
```

GPU: install a CUDA build of torch instead of the CPU wheel, or use the container venv
overlay described in `containers/README.md` (needed for the `te` attention backend).
Run with `--device cuda`.

Tests marked `@pytest.mark.dist(N)` run only under `mpirun -n N`, so check the rank
counts that cover your change (CI runs 1, 4 and 8). `--runslow` adds the slow tests.

### Common commands

```bash
mpirun -n 4 .venv/bin/pytest tests/modules/linear/test_linear.py --device cpu   # one file
mpirun -n 4 .venv/bin/pytest "tests/test_mesh.py::<test_name>" --device cpu       # one test
.venv/bin/pytest tests --device cpu                                             # 1-rank tests only
.venv/bin/mypy --config-file pyproject.toml --ignore-missing-imports
.venv/bin/pre-commit run --all-files
```

Run from the repo root with the `.venv` binaries; otherwise an installed `jigsaw` can
shadow the checkout.

### Debugging hangs

A test that never finishes is usually a collective or p2p call that some ranks reached
and others did not, not a slow test. Dump every rank's stack:

```bash
mpirun -n 4 .venv/bin/python -m pytest <test_id> --device cpu \
    -o faulthandler_timeout=20 --print-all-ranks
```

Compare which ranks sit in a collective and which wait in `handle.wait()`. Check
`pgrep -fa "[m]pirun"` for leftover jobs before rerunning.

The pre-commit hooks run ruff, mypy, numpydoc, codespell and others on every commit,
and the full test suite on push. Do not skip or bypass them (`--no-verify`, `SKIP=`);
fix the failure instead.

## Rules

- **Do not commit, push, or write commit messages.** Leave changes in the working tree
  for the user to review and commit.
- **Private by default.** Anything not part of the public API gets a leading `_`
  (modules, functions, classes, attributes). Public names are re-exported through each
  package's `__init__.py` `__all__`, which calls `set_module(globals(), __all__)`.
- **Docstrings** follow numpydoc (checked by ruff and numpydoc-validation). Keep them
  brief: the one-line why behind non-obvious code, hard limits, and the exception class
  each error case raises. No benchmark claims, history, comparisons to other libraries,
  or prose that repeats the Parameters section.
- **No first person** (we/our/us) anywhere, including comments and examples. Say
  "this rank", "the test", "this example".
- **American spelling** (e.g. "behavior", "normalize", "neighboring").
- **Type hints** on every function in `jigsaw/`; mypy runs with `disallow_untyped_defs`.
- **Distributed tests** compare through `tests/parity.py` (`assert_close`,
  `all_ranks_ok`, `check_parity`), never a bare `assert torch.allclose`: an assert that
  fails on one rank hangs the others in the next collective.
- **Parity with torch.** Distributed layers must match the non-distributed torch result,
  including initialization (`jigsaw.init`). Add a parity test for new layers.
- Commit messages, when the user writes them, follow Conventional Commits.
