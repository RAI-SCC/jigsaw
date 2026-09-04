"""Calibration case -- dense square matmul, forward + backward, one device.

This is **not** a jigsaw benchmark. It is a known, roofline-friendly workload
whose achieved TFLOP/s should land in a predictable band for a given GPU. Run it
first to check that the harness and the machine agree with expectations before
trusting jigsaw module numbers, and as the smallest end-to-end exercise of the
timing / fingerprint / run-file path.

    python -m benchmarks run calibration_matmul [--sizes N ...] [--dtypes D ...]

``C = A @ B`` with ``A, B`` square and ``requires_grad``; one step is
``(A @ B).sum().backward()``. FLOPs per step = ``2 n^3`` (fwd) + ``4 n^3`` (bwd,
two matmuls of the same size) = ``6 n^3``.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.harness import fingerprint, timing
from benchmarks.harness.result import (
    SCHEMA_VERSION,
    params_key,
    summary_markdown,
    write_run,
)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _build(device: torch.device, n: int, dtype: str):
    """Return ``(step, reset, flops)`` for an ``n x n`` matmul in ``dtype``."""
    dt = _DTYPES[dtype]
    a = torch.randn(n, n, device=device, dtype=dt, requires_grad=True)
    b = torch.randn(n, n, device=device, dtype=dt, requires_grad=True)

    def step() -> None:
        (a @ b).sum().backward()

    def reset() -> None:
        a.grad = None
        b.grad = None

    return step, reset, 6 * n**3


def main(argv: list[str] | None = None) -> int:
    """Run the matmul grid, write a run file, print the summary + compare hint."""
    ap = argparse.ArgumentParser(
        prog="python -m benchmarks run calibration_matmul",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument(
        "--dtypes", nargs="+", default=["bfloat16", "float32"], choices=list(_DTYPES)
    )
    ap.add_argument(
        "--tf32", action="store_true", help="allow TF32 (recorded either way)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-budget-s", type=float, default=15.0)
    ap.add_argument("--min-block-ms", type=float, default=50.0)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = True

    knobs = {
        "seed": args.seed,
        "allow_tf32": args.tf32,
        "cudnn_benchmark": True,
        "device": device.type,
        "min_block_ms": args.min_block_ms,
        "time_budget_s": args.time_budget_s,
    }
    fp = fingerprint.collect(world_size=1)
    hwk = fingerprint.hw_key(fp)
    meta = {
        "schema": SCHEMA_VERSION,
        "hw_key": hwk,
        "case_file": "calibration_matmul",
        "knobs": knobs,
        "fingerprint": fp,
    }

    print(f"device={device.type}  gpu={fp.get('gpu_name')}  hw_key={hwk}")
    rows: list[dict] = []
    gc.collect()
    gc.disable()
    for dtype in args.dtypes:
        for n in args.sizes:
            step, reset, flops = _build(device, n, dtype)
            clk0 = fingerprint.read_clocks()
            st = timing.benchmark(
                step,
                device=device,
                reset=reset,
                flops=lambda flops=flops: flops,
                min_block_ms=args.min_block_ms,
                time_budget_s=args.time_budget_s,
            )
            clk1 = fingerprint.read_clocks()
            del step, reset
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            gc.disable()

            rows.append(
                {
                    "case": "calibration_matmul",
                    "params": {"n": n, "dtype": dtype},
                    "stats": asdict(st),
                    "knobs": knobs,
                    "clocks_start": clk0,
                    "clocks_end": clk1,
                }
            )
            iqr_pct = (
                100 * (st.p75_ms - st.p25_ms) / st.median_ms if st.median_ms else 0
            )
            tf = f"{st.tflops:7.1f} TFLOP/s" if st.tflops else ""
            print(
                f"  {params_key(rows[-1]['params']):<22} "
                f"{st.median_ms:9.3f} ms  {tf}  IQR {iqr_pct:4.1f}%"
                f"{'' if st.converged else '  (no converge)'}"
            )
    gc.enable()

    ts = fp["timestamp"].replace(":", "").replace("-", "").replace("+0000", "Z")
    path = args.out / f"{ts}-{hwk}-calibration_matmul.jsonl"
    write_run(path, meta, rows)
    print("\n" + summary_markdown(rows))
    print(f"\nwrote {path}")
    print(f"compare: python -m benchmarks compare {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
