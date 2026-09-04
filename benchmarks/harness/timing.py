"""Measurement core: blocked-autorange timing with cross-rank aggregation.

One measurement of a ``step`` callable (one forward+backward):

1. ``n_warmup`` warmup calls, discarded (lets cuDNN autotune settle).
2. Pick an inner-loop count ``inner`` so one timed block lasts at least
   ``min_block_ms`` -- amortises kernel-launch overhead and CUDA-event
   granularity.
3. Time blocks until the inter-quartile range of the per-step times falls under
   ``iqr_target`` of the median (and at least ``min_blocks`` blocks), or the
   time budget / ``max_blocks`` is hit.

Between blocks the L2 cache is flushed and ``reset`` is called (clear ``.grad``).
When a process ``group`` is given, every block is bracketed by a barrier and the
per-step time is reduced with ``MAX`` across ranks, so the reported number is the
slowest rank -- the true wall time. Each rank's own median is also gathered so
callers can see load imbalance.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist

__all__ = ["BenchStats", "benchmark", "asdict"]


@dataclass
class BenchStats:
    """Reported timing for one ``step`` callable."""

    median_ms: float
    min_ms: float
    p25_ms: float
    p75_ms: float
    stddev_ms: float
    iters: int
    blocks: int
    inner: int
    converged: bool
    tflops: Optional[float] = None
    comm_gbs: Optional[float] = None
    per_rank_median_ms: Optional[list[float]] = None
    imbalance: Optional[float] = None


class _L2Flusher:
    """Overwrite a buffer larger than the L2 cache to evict warm data."""

    def __init__(self, device: torch.device) -> None:
        n_bytes = 32 * 1024 * 1024
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            n_bytes = (
                getattr(props, "L2_cache_size", 0)
                or getattr(props, "l2_cache_size", 0)
                or n_bytes
            ) * 2
        self._buf = torch.empty(n_bytes, dtype=torch.int8, device=device)

    def __call__(self) -> None:
        self._buf.zero_()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    step: Callable[[], None],
    *,
    device: torch.device,
    reset: Optional[Callable[[], None]] = None,
    flops: Optional[Callable[[], int]] = None,
    comm_bytes: Optional[Callable[[], int]] = None,
    group: "Optional[dist.ProcessGroup]" = None,
    n_warmup: int = 25,
    min_block_ms: float = 50.0,
    min_blocks: int = 20,
    max_blocks: int = 200,
    time_budget_s: float = 20.0,
    iqr_target: float = 0.05,
) -> BenchStats:
    """Time ``step`` (one fwd+bwd) and return aggregated statistics.

    Parameters
    ----------
    step
        Zero-arg callable performing one forward+backward.
    device
        Device ``step`` runs on; selects CUDA-event vs ``perf_counter`` timing
        and the L2-flush buffer.
    reset
        Called between blocks, e.g. to null gradients. Not timed.
    flops, comm_bytes
        Optional zero-arg callables returning the FLOPs / bytes moved by one
        ``step``; used to derive TFLOP/s and GB/s.
    group
        Process group for cross-rank aggregation. ``None`` for single process.
    n_warmup
        Discarded ``step`` calls before measurement.
    min_block_ms
        Minimum wall time of one timed block; sets the inner-loop count.
    min_blocks, max_blocks
        Lower / upper bound on the number of timed blocks.
    time_budget_s
        Stop measuring once this many seconds have elapsed.
    iqr_target
        Stop once ``IQR / median`` of the per-step times drops to this fraction.
    """
    is_cuda = device.type == "cuda"
    flush = _L2Flusher(device)

    def run(n: int) -> None:
        for _ in range(n):
            step()

    # --- warmup -------------------------------------------------------------
    run(n_warmup)
    if reset is not None:
        reset()
    _sync(device)

    def time_block(inner: int) -> float:
        """Return per-step milliseconds for one block of ``inner`` steps."""
        if reset is not None:
            reset()
        flush()
        if group is not None:
            dist.barrier(group)
        if is_cuda:
            _sync(device)
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            run(inner)
            ev1.record()
            _sync(device)
            ms = ev0.elapsed_time(ev1)
        else:
            t0 = time.perf_counter()
            run(inner)
            ms = (time.perf_counter() - t0) * 1e3
        return max(ms, 1e-6) / inner

    # --- pick inner so a block lasts >= min_block_ms ----------------------
    inner = 1
    per_step = time_block(inner)
    while per_step * inner < min_block_ms and inner < 100_000:
        target = min_block_ms / max(per_step * inner, 1e-6)
        inner = min(100_000, max(inner + 1, int(inner * target * 1.2)))
        per_step = time_block(inner)

    # --- measured blocks -------------------------------------------------
    times: list[float] = []
    t_start = time.perf_counter()
    converged = False
    for _ in range(max_blocks):
        times.append(time_block(inner))
        if len(times) >= min_blocks:
            arr = np.array(times)
            med = float(np.median(arr))
            iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
            if med > 0 and iqr / med <= iqr_target:
                converged = True
                break
        if time.perf_counter() - t_start > time_budget_s:
            break

    local = np.array(times)

    # --- cross-rank aggregation ----------------------------------------
    per_rank_median: Optional[list[float]] = None
    imbalance: Optional[float] = None
    if group is not None and dist.is_initialized():
        # reduce each block to the slowest rank, then summarise
        red = torch.tensor(times, device=device if is_cuda else "cpu")
        dist.all_reduce(red, op=dist.ReduceOp.MAX, group=group)
        agg = red.cpu().numpy()
        medians = [None] * dist.get_world_size(group)
        dist.all_gather_object(medians, float(np.median(local)), group=group)
        per_rank_median = [round(m, 6) for m in medians]  # type: ignore[arg-type]
        mean = float(np.mean(per_rank_median))
        imbalance = float(np.max(per_rank_median) / mean) if mean > 0 else None
    else:
        agg = local

    median_s = float(np.median(agg)) / 1e3
    stats = BenchStats(
        median_ms=float(np.median(agg)),
        min_ms=float(np.min(agg)),
        p25_ms=float(np.percentile(agg, 25)),
        p75_ms=float(np.percentile(agg, 75)),
        stddev_ms=float(np.std(agg)),
        iters=len(times) * inner,
        blocks=len(times),
        inner=inner,
        converged=converged,
        per_rank_median_ms=per_rank_median,
        imbalance=imbalance,
    )
    if flops is not None and median_s > 0:
        stats.tflops = flops() / median_s / 1e12
    if comm_bytes is not None and median_s > 0:
        stats.comm_gbs = comm_bytes() / median_s / 1e9
    return stats
