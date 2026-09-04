# benchmarks/ — reproducible fwd+backward timing

Times the forward+backward cost of jigsaw modules on **dummy data** (no dataset,
no I/O), records a hardware fingerprint next to every number, and diffs a run
against a committed per-hardware baseline so regressions are visible in review.

Not shipped in the wheel — it is not in `[tool.setuptools] packages`. Run it from
a checkout (`PYTHONPATH=<repo>`, which `scripts/run.sh` sets inside the
container, or an editable install).

## Layout

| path | what |
| --- | --- |
| `harness/timing.py` | blocked-autorange measurement loop; CUDA events; cross-rank `MAX` aggregation + per-rank imbalance |
| `harness/fingerprint.py` | environment capture + `hw_key` (names the baseline file) + live clock sampling |
| `harness/launch.py` | `init()` / `cleanup()` — resolve the launch env, bring up the process group |
| `harness/mesh.py` | `"RxC"` spec → `{row, col, model}` process groups |
| `harness/driver.py` | `run_grid()` — iterate a param grid, time each combo, rank-0 writes the run file |
| `harness/result.py` | JSONL run-file read/write + markdown summary |
| `harness/compare.py` | `compare` / `promote` against `baselines/<hw_key>.jsonl` |
| `cases/calibration_matmul.py` | calibration only — dense square matmul, not a jigsaw benchmark |
| `cases/distributed_matmul.py` | `DistributedMatMul{XW,XWT,XTW}` on a mesh |
| `cases/distributed_conv.py` | `DistributedConv{2,3}D` — grouped vs full-channel |
| `baselines/` | committed reference numbers, one file per hardware class |
| `results/` | run output — gitignored |

## Cases

```bash
# calibration — machine/harness sanity check (auto cuda, falls back to cpu)
python -m benchmarks run calibration_matmul --sizes 4096 8192 --dtypes bfloat16 float32

# distributed matmul — one task per rank, 2x2 mesh (world 4)
srun -n4 scripts/run.sh python -m benchmarks run distributed_matmul \
    --mesh 2x2 --variants xw xwt --sizes 4096 8192

# distributed conv — groups=1 (full channel gather) vs groups=4 (channel-local),
# on a 1x4 channel-parallel mesh
srun -n4 scripts/run.sh python -m benchmarks run distributed_conv \
    --mesh 1x4 --dim 3 --groups 1 4 --channel-comm all_gather p2p
```

`distributed_conv --groups` that is neither `1` nor a multiple of `col.size()`
is rejected by jigsaw today; the harness records it as **skipped** (with the
reason) instead of failing, so the row starts producing numbers the moment
non-divisible groups land. `imbalance = max/mean` of the per-rank medians is the
metric for that comparison.

## Regression tracking

```bash
python -m benchmarks compare benchmarks/results/<run>.jsonl   # diff vs baseline
python -m benchmarks promote benchmarks/results/<run>.jsonl   # make it the baseline
```

`compare` matches rows by `(case, params)`, prints base/new/ratio, and exits
non-zero when any row is slower than the baseline by more than `--tol`
(default 8 %) — wire into CI later if wanted. Baselines are keyed by `hw_key`
(GPU name, driver major, CUDA major, torch version, **world size**), so a
container run and a bare-metal run get different baselines and are never
cross-compared.

See `bench.sbatch.example` for a full batch run (calibration + both distributed
cases + compare).

## Measurement method

- **Warmup** (default 25 steps) discarded; lets cuDNN autotune settle.
- **Inner loop** sized so one timed block ≥ `--min-block-ms` (default 50 ms).
- **Blocks** repeated until IQR/median ≤ 5 % (min 20 blocks) or the time budget
  / max blocks is hit; a `*` in the summary marks non-convergence.
- Between blocks: null `.grad`, flush a buffer larger than L2. No `empty_cache`
  inside the loop.
- CUDA-event timing with `synchronize`; `perf_counter` on CPU.
- GC disabled during a case, `gc.collect()` between cases.
- Distributed: `dist.barrier` per block, per-step time reduced with `MAX` across
  the model group (reported = slowest rank), each rank's median gathered for the
  `imbalance` column.
- Reported TFLOP/s is the **global** FLOP count over the slowest-rank wall time
  (an approximation for conv).

## Reproducibility notes

Every row records the knobs that move timing: seed, `allow_tf32`,
`cudnn.benchmark`, device, backend, mesh, block/budget settings, plus GPU SM/mem
clocks, temperature and throttle reasons sampled before and after each case.
GPUs boost and thermally throttle, so for tight cross-run agreement lock the
clocks for the job (`nvidia-smi -pm 1` then `nvidia-smi -lgc <freq>` /
`-lmc <freq>`, needs permission) and check `clocks_start` / `clocks_end` agree in
the run file.

In a minimal container `git` may be absent — the fingerprint then records
`jigsaw_commit: null` rather than failing. `nvidia-smi` is normally injected by
`apptainer --nv`.

## Adding a case

Create `cases/<name>.py` exposing `main(argv)`, register it in
`benchmarks/__main__.py::_CASES`. For a single-device case, build inputs and hand
`driver.run_grid` a `build(params) -> (step, reset, flops, comm_bytes)` (last
three optional). For a distributed case, `launch.init()` + `mesh.make_mesh()`
first, pass `group=md["model"]` and `world_size=ctx.world_size` to `run_grid`,
and raise `NotImplementedError` from `build` for unsupported combos.
