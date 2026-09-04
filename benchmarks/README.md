# benchmarks/ — reproducible fwd+backward timing

Times the forward+backward cost of jigsaw modules on **dummy data** (no dataset,
no I/O), records a hardware fingerprint next to every number, and diffs a run
against a committed per-hardware baseline so regressions are visible in review.

Not shipped in the wheel — it is not in `[tool.setuptools] packages`. Run it from
a checkout (`PYTHONPATH=<repo>` or an editable install).

## Layout

| path | what |
| --- | --- |
| `harness/timing.py` | blocked-autorange measurement loop; CUDA events; cross-rank `MAX` aggregation |
| `harness/fingerprint.py` | environment capture + `hw_key` (names the baseline file) + live clock sampling |
| `harness/result.py` | JSONL run-file read/write + markdown summary |
| `harness/compare.py` | `compare` / `promote` against `baselines/<hw_key>.jsonl` |
| `cases/calibration_matmul.py` | the one calibration case (dense square matmul) |
| `baselines/` | committed reference numbers, one file per hardware class |
| `results/` | run output — gitignored |

## Usage

```bash
# run the calibration (auto-detects cuda, falls back to cpu)
python -m benchmarks run calibration_matmul --sizes 2048 4096 8192 --dtypes bfloat16 float32

# on the cluster, through the container launcher
JIGSAW_SITE=horeka scripts/run.sh python -m benchmarks run calibration_matmul

# diff a fresh run against the committed baseline for this hardware
python -m benchmarks compare benchmarks/results/2026...-calibration_matmul.jsonl

# make a run the baseline (commit the result in a normal PR)
python -m benchmarks promote benchmarks/results/2026...-calibration_matmul.jsonl
```

`compare` exits non-zero when any row is slower than the baseline by more than
`--tol` (default 8%), so it can be wired into CI later — nothing forces that now.

## Measurement method

- **Warmup** (default 25 steps) discarded; lets cuDNN autotune settle.
- **Inner loop** sized so one timed block ≥ `--min-block-ms` (default 50 ms).
- **Blocks** repeated until IQR/median ≤ 5 % (min 20 blocks) or the time budget
  / max blocks is hit; a `*` in the summary marks non-convergence.
- Between blocks: null `.grad`, flush a buffer larger than L2. No `empty_cache`
  inside the loop.
- CUDA-event timing with `synchronize`; `perf_counter` on CPU.
- GC disabled during a case, `gc.collect()` between cases.
- Distributed (future cases): barrier per block, per-step time reduced with `MAX`
  across ranks (reported = slowest rank), each rank's median gathered for the
  `imbalance = max/mean` column.

## Reproducibility notes

Every row records the knobs that move timing: seed, `allow_tf32`,
`cudnn.benchmark`, device, block/budget settings, plus GPU SM/mem clocks,
temperature and throttle reasons sampled before and after each case. GPUs boost
and thermally throttle, so for tight cross-run agreement lock the clocks for the
job (`nvidia-smi -pm 1` then `nvidia-smi -lgc <freq>` / `-lmc <freq>`, needs
permission) and check `clocks_start` / `clocks_end` agree in the run file.

The `hw_key` hashes GPU name, driver major, CUDA major, torch version and world
size; a run is only compared against a baseline with the same key.

## Adding a case

Create `cases/<name>.py` exposing `main(argv)`, register it in
`benchmarks/__main__.py::_CASES`. Build your module + dummy inputs, hand
`timing.benchmark` a zero-arg `step` (one fwd+bwd), a `reset` (null grads), and
optionally `flops` / `comm_bytes` callables for the TFLOP/s and GB/s columns.
Distributed cases additionally build a mesh, pass `group=`, and gate on world
size.
