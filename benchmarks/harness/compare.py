"""Diff a run against a committed per-hardware baseline.

    python -m benchmarks compare <run.jsonl> [--baseline P] [--tol F]
    python -m benchmarks promote <run.jsonl> [--baseline P]

``compare`` matches rows by ``(case, params)``, reports
``ratio = new_median / baseline_median`` and exits non-zero if any row is slower
by more than ``--tol`` (default 8%). ``promote`` copies a run to
``benchmarks/baselines/<hw_key>.jsonl`` so it becomes the reference in a normal
reviewable diff.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .result import params_key, read_run

_BASELINE_DIR = Path(__file__).resolve().parents[1] / "baselines"

__all__ = ["compare_rows", "main"]


def compare_rows(
    new_rows: list[dict], base_rows: list[dict], tol: float = 0.08
) -> list[dict]:
    """Match rows by ``(case, params)`` and classify each new/base median ratio."""
    base = {
        (r["case"], params_key(r["params"])): r
        for r in base_rows
        if r.get("status", "ok") == "ok"
    }
    seen: set = set()
    out: list[dict] = []
    for r in new_rows:
        if r.get("status", "ok") != "ok":
            continue
        key = (r["case"], params_key(r["params"]))
        seen.add(key)
        b = base.get(key)
        nm = r["stats"].get("median_ms")
        if b is None:
            out.append(
                {
                    "key": key,
                    "status": "new",
                    "new_ms": nm,
                    "base_ms": None,
                    "ratio": None,
                }
            )
            continue
        bm = b["stats"].get("median_ms")
        ratio = nm / bm if (nm and bm) else None
        if ratio is None:
            status = "?"
        elif ratio > 1 + tol:
            status = "REGRESSION"
        elif ratio < 1 - tol:
            status = "improvement"
        else:
            status = "ok"
        out.append(
            {"key": key, "status": status, "new_ms": nm, "base_ms": bm, "ratio": ratio}
        )
    for key, b in base.items():
        if key not in seen:
            out.append(
                {
                    "key": key,
                    "status": "missing",
                    "new_ms": None,
                    "base_ms": b["stats"].get("median_ms"),
                    "ratio": None,
                }
            )
    return out


def _print_table(cmps: list[dict], tol: float) -> None:
    width = max((len(f"{c['key'][0]} {c['key'][1]}") for c in cmps), default=10)
    print(
        f"{'case / params':<{width}}  {'base ms':>10}  {'new ms':>10}  {'ratio':>7}  status"
    )
    for c in sorted(cmps, key=lambda c: -(c["ratio"] or 0)):
        name = f"{c['key'][0]} {c['key'][1]}"
        base = f"{c['base_ms']:.3f}" if c["base_ms"] else "-"
        new = f"{c['new_ms']:.3f}" if c["new_ms"] else "-"
        ratio = f"{c['ratio']:.3f}" if c["ratio"] else "-"
        print(f"{name:<{width}}  {base:>10}  {new:>10}  {ratio:>7}  {c['status']}")
    print(f"\ntolerance: +-{tol:.0%}")


def main(argv: list[str]) -> int:
    """Run ``compare`` or ``promote``; return non-zero on a regression."""
    ap = argparse.ArgumentParser(prog="python -m benchmarks")
    ap.add_argument("command", choices=["compare", "promote"])
    ap.add_argument("run", type=Path)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--tol", type=float, default=0.08)
    args = ap.parse_args(argv)

    meta, rows = read_run(args.run)
    hwk = meta.get("hw_key", "unknown")
    baseline = args.baseline or _BASELINE_DIR / f"{hwk}.jsonl"

    if args.command == "promote":
        baseline.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.run, baseline)
        print(f"promoted {args.run} -> {baseline}")
        return 0

    if not baseline.exists():
        print(f"no baseline at {baseline}")
        print("nothing to compare -- run `python -m benchmarks promote` to create it")
        return 0

    _, base_rows = read_run(baseline)
    cmps = compare_rows(rows, base_rows, tol=args.tol)
    _print_table(cmps, args.tol)

    regressions = [c for c in cmps if c["status"] == "REGRESSION"]
    if regressions:
        print(f"\n{len(regressions)} regression(s)")
        return 1
    print("\nno regressions")
    return 0
