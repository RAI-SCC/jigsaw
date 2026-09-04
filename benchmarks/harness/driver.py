"""Shared case driver: iterate a param grid, time each combo, write the run file.

A case supplies a ``build(params) -> (step, reset, flops, comm_bytes)`` callable
(any of the last three may be ``None``) and a list of param dicts. ``build`` may
raise :class:`NotImplementedError` or return ``None`` to record the combo as
*skipped* rather than fail the whole run -- used for config combinations a module
does not support yet.

Timing, GC handling, cross-rank aggregation (via ``group``) and rank-0-only
writing all live here so the case modules stay short.
"""

from __future__ import annotations

import gc
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from . import fingerprint, timing
from .result import SCHEMA_VERSION, params_key, summary_markdown, write_run

__all__ = ["run_grid"]


def _is_rank0(group) -> bool:
    return not dist.is_initialized() or dist.get_rank(group) == 0


def run_grid(
    *,
    case: str,
    combos: list[dict],
    build,
    device: torch.device,
    knobs: dict,
    out_dir: str | Path,
    group=None,
    world_size: int = 1,
    file_tag: str = "",
    min_block_ms: float = 50.0,
    time_budget_s: float = 15.0,
) -> Path | None:
    """Run every combo in ``combos`` and, on rank 0, write + summarise the run.

    Returns the run-file path on rank 0, else ``None``.
    """
    fp = fingerprint.collect(world_size=world_size)
    hwk = fingerprint.hw_key(fp)
    meta = {
        "schema": SCHEMA_VERSION,
        "hw_key": hwk,
        "case_file": case,
        "knobs": knobs,
        "fingerprint": fp,
    }

    if _is_rank0(group):
        print(
            f"case={case}  device={device.type}  world={world_size}  "
            f"gpu={fp.get('gpu_name')}  hw_key={hwk}"
        )

    rows: list[dict] = []
    gc.collect()
    gc.disable()
    for params in combos:
        clk0 = fingerprint.read_clocks()
        reason: str | None = None
        try:
            built = build(params)
        except NotImplementedError as exc:
            built, reason = None, f"NotImplementedError: {exc}"

        if built is None:
            rows.append(
                {
                    "case": case,
                    "params": params,
                    "status": "skipped",
                    "reason": reason or "build returned None",
                    "stats": {},
                    "knobs": knobs,
                    "clocks_start": clk0,
                    "clocks_end": clk0,
                }
            )
            if _is_rank0(group):
                print(f"  {params_key(params):<44} SKIPPED  {reason or ''}")
            continue

        step, reset, flops, comm_bytes = built
        st = timing.benchmark(
            step,
            device=device,
            reset=reset,
            flops=(lambda f=flops: f) if flops is not None else None,
            comm_bytes=(lambda b=comm_bytes: b) if comm_bytes is not None else None,
            group=group,
            min_block_ms=min_block_ms,
            time_budget_s=time_budget_s,
        )
        clk1 = fingerprint.read_clocks()
        del step, reset, built
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        gc.disable()

        rows.append(
            {
                "case": case,
                "params": params,
                "status": "ok",
                "stats": asdict(st),
                "knobs": knobs,
                "clocks_start": clk0,
                "clocks_end": clk1,
            }
        )
        if _is_rank0(group):
            tf = f"{st.tflops:9.1f} TFLOP/s" if st.tflops else ""
            gb = f"  {st.comm_gbs:6.1f} GB/s" if st.comm_gbs else ""
            imb = f"  imbal {st.imbalance:.2f}" if st.imbalance else ""
            print(
                f"  {params_key(params):<44} {st.median_ms:9.3f} ms  {tf}{gb}{imb}"
                f"{'' if st.converged else '  (no converge)'}"
            )
    gc.enable()

    if not _is_rank0(group):
        return None

    ts = fp["timestamp"].replace(":", "").replace("-", "").replace("+0000", "Z")
    tag = f"-{file_tag}" if file_tag else ""
    path = Path(out_dir) / f"{ts}-{hwk}-{case}{tag}.jsonl"
    write_run(path, meta, rows)

    ok = [r for r in rows if r["status"] == "ok"]
    print("\n" + summary_markdown(ok))
    skipped = [r for r in rows if r["status"] == "skipped"]
    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for r in skipped:
            print(f"  {params_key(r['params'])}: {r['reason']}")
    print(f"\nwrote {path}")
    print(f"compare: python -m benchmarks compare {path}")
    return path
