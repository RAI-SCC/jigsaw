# Contributing to Jigsaw

Contributions are welcome. This guide covers setting up a development install,
testing a change and opening a pull request. If you are unsure whether a change
fits, open an issue first and describe what you have in mind.

## How to contribute

1. **Fork the repository** on GitHub. Members of the Jigsaw team can create a
   branch in the repository directly.

2. **Install Jigsaw for development** in a venv and set up the pre-commit hooks.
   `mpi4py` needs a system OpenMPI (`libopenmpi-dev openmpi-bin` on Ubuntu).

   ```console
   git clone https://github.com/<your-account>/jigsaw.git
   cd jigsaw
   python -m venv .venv
   .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
   .venv/bin/pip install -e ".[dev]"
   .venv/bin/pre-commit install
   ```

   For GPU work, install a CUDA build of torch instead, or use the container setup
   in [containers/README.md](containers/README.md), which the `te` attention
   backend requires. Run commands from the repo root with the `.venv` binaries;
   otherwise an installed `jigsaw` can shadow the checkout.

3. **Create a branch** with a prefix for the kind of change: `feat/`, `fix/`,
   `refactor/`, `chore/` or `docs/`, for example `feat/grouped-conv`.

4. **Make your changes**, following the conventions below.

5. **Test your changes** under `mpirun`:

   ```console
   mpirun -n 4 .venv/bin/pytest tests --device cpu
   ```

   Tests marked `@pytest.mark.dist(N)` only run under exactly `N` ranks, so run
   each rank count your change affects. CI runs 1, 4 and 8 ranks on CPU; 2 and 6
   ranks and the slow tests (`--runslow`) only run in a full CI run. GPU tests
   (`--device cuda`) and larger rank counts are not in CI, so run them yourself if
   you change GPU-only code such as NCCL communication or the sdpa/TE attention
   cores.

   A test that hangs is usually a collective that only some ranks reached. To see
   where each rank is stuck:

   ```console
   mpirun -n 4 .venv/bin/python -m pytest <test_id> --device cpu \
       -o faulthandler_timeout=20 --print-all-ranks
   ```

6. **Update the documentation** in `docs/` if your change affects how Jigsaw is
   used, and add new public members to `docs/api.rst`. Check that the docs build
   without new warnings:

   ```console
   .venv/bin/pip install -r docs/requirements.txt
   make -C docs html
   ```

7. **Commit** following [Conventional Commits](https://www.conventionalcommits.org/),
   for example `fix: reduce GroupNorm stats over spatial groups`. The pre-commit
   hooks run ruff, mypy, numpydoc, codespell, TorchFix and commitlint on every
   commit, and the test suite on push. If a hook fails, fix the cause instead of
   skipping it.

8. **Open a pull request** against `main` and fill in the template's checklist.

## Conventions

- **Parity with torch.** A distributed layer must give the same result as the
  non-distributed torch layer, including initialization (`jigsaw.init`). New
  layers need a parity test.
- **Distributed tests** compare results with the helpers in
  [tests/parity.py](tests/parity.py) (`assert_close`, `all_ranks_ok`,
  `check_parity`), not a bare `assert torch.allclose`. An assert that fails on one
  rank leaves the others hanging in the next collective.
- **Private by default.** Anything outside the public API gets a leading `_`.
  Public names are listed in the `__all__` of their package's `__init__.py`.
- **Type hints** are required on every function in `jigsaw/`. Pre-commit only
  checks the files in a commit, so run mypy on everything before you push:
  `.venv/bin/mypy --config-file pyproject.toml --ignore-missing-imports`.
- **Docstrings** follow [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).
  Constructor arguments go in the class docstring; `__init__` has none of its
  own. Keep them short: a one-line summary, anything non-obvious, hard limits, and
  the exception each error case raises. Leave out history and benchmarks.
- **Wording.** American spelling, and no first person ("we", "our") anywhere,
  including comments and examples.

## Questions and bug reports

Please open an [issue](https://github.com/RAI-SCC/jigsaw/issues). For bugs,
include the mesh shape, rank count, device, backend (Gloo or NCCL), torch version
and a minimal script that reproduces the problem.
