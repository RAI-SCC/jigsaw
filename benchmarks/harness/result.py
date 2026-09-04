"""Run file I/O and a markdown summary.

A run file is JSONL: the first line is ``{"_meta": ...}`` (fingerprint + knobs +
``hw_key``), every following line is one result row::

    {
        "case": str,
        "params": {...},
        "stats": {...},
        "knobs": {...},
        "clocks_start": {...} | null,
        "clocks_end": {...} | null,
    }
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION", "write_run", "read_run", "params_key", "summary_markdown"]


def params_key(params: dict) -> str:
    """Stable string key for a param combo, used to match rows across runs."""
    return ",".join(f"{k}={params[k]}" for k in sorted(params))


def write_run(path: str | Path, meta: dict, rows: list[dict]) -> None:
    """Write ``meta`` as the ``_meta`` header line then one JSON object per row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_run(path: str | Path) -> tuple[dict, list[dict]]:
    """Inverse of :func:`write_run`: return ``(meta, rows)``."""
    meta: dict = {}
    rows: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_meta" in obj:
                meta = obj["_meta"]
            else:
                rows.append(obj)
    return meta, rows


def _fmt(x: float | None, spec: str = ".3f") -> str:
    return format(x, spec) if isinstance(x, (int, float)) else "-"


def summary_markdown(rows: list[dict]) -> str:
    """One-line-per-row markdown table."""
    head = (
        "| case | params | median ms | min ms | IQR% | TFLOP/s | GB/s | "
        "imbalance | blocks |"
    )
    sep = "|" + "---|" * 9
    lines = [head, sep]
    for r in rows:
        s = r["stats"]
        med = s.get("median_ms")
        iqr_pct = 100 * (s["p75_ms"] - s["p25_ms"]) / med if med else None
        lines.append(
            "| {case} | {params} | {med} | {mn} | {iqr} | {tf} | {gb} | {imb} | "
            "{blk}{conv} |".format(
                case=r["case"],
                params=params_key(r["params"]),
                med=_fmt(med),
                mn=_fmt(s.get("min_ms")),
                iqr=_fmt(iqr_pct, ".1f"),
                tf=_fmt(s.get("tflops"), ".1f"),
                gb=_fmt(s.get("comm_gbs"), ".1f"),
                imb=_fmt(s.get("imbalance"), ".2f"),
                blk=s.get("blocks", "-"),
                conv="" if s.get("converged") else "*",
            )
        )
    lines.append("")
    lines.append("`*` = did not converge to the IQR target within the time budget.")
    return "\n".join(lines)
