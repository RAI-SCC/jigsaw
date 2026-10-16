## Summary
<!-- What does this PR change, and why? Link related issues (e.g. "Closes #123"). -->

## Type of change
- [ ] Bug fix
- [ ] New feature / module
- [ ] Refactor / cleanup (no behavior change)
- [ ] Docs / CI / tooling

## Testing
<!-- Which rank counts / meshes / devices did you run? CI only covers CPU with 1/2/4/6/8 ranks. -->
- [ ] New or updated tests cover the change (jigsaw-related distributed comparisons go through `tests/parity.py`)
- [ ] Tests pass locally: `mpirun -n <N> pytest tests --device cpu`
- [ ] GPU / NCCL tested, if the change touches GPU-only paths (e.g. `distributed_matmul`, sdpa/TE attention)
- [ ] Larger rank counts (10/12/24) run on the cluster, if relevant

## Checklist
- [ ] I have reviewed this code
- [ ] `pre-commit run --all-files` passes (ruff, ruff-format, mypy, pytest, whitespace/YAML checks)
- [ ] `mypy` passes
- [ ] Public APIs have numpy-style docstrings
- [ ] Docs build without new warnings (`make -C docs html`), and `docs/api.rst` is updated for new public members
- [ ] README / examples updated, if the change is user-facing

## Notes for reviewers
<!-- Anything non-obvious: known limitations, follow-ups, unsupported mesh shapes, etc. -->
