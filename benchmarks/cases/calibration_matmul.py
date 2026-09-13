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
from pathlib import Path

import torch

from benchmarks.harness.driver import run_grid

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _builder(device: torch.device):
    def build(params: dict):
        dt = _DTYPES[params["dtype"]]
        n = params["n"]
        a = torch.randn(n, n, device=device, dtype=dt, requires_grad=True)
        b = torch.randn(n, n, device=device, dtype=dt, requires_grad=True)

        def step() -> None:
            (a @ b).sum().backward()

        def reset() -> None:
            a.grad = None
            b.grad = None

        return step, reset, 6 * n**3, None

    return build


def main(argv: list[str] | None = None) -> int:
    """Run the matmul grid and write a run file."""
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
    combos = [{"n": n, "dtype": d} for d in args.dtypes for n in args.sizes]
    run_grid(
        case="calibration_matmul",
        combos=combos,
        build=_builder(device),
        device=device,
        knobs=knobs,
        out_dir=args.out,
        min_block_ms=args.min_block_ms,
        time_budget_s=args.time_budget_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
