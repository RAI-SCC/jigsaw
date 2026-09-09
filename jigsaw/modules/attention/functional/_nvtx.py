"""Opt-in NVTX ranges for the ring-attention forward / backward loops.

Every helper here is a no-op unless the ``JIGSAW_NVTX`` environment variable is
set to a truthy value (anything other than unset / ``""`` / ``0`` / ``false`` /
``no``). The gate is read once at import.

``torch.cuda.nvtx`` ranges are themselves cheap when no profiler is attached,
but the env gate also skips the per-step f-string formatting, so leaving the
calls in the hot loop costs nothing in a normal run. Point ``nsys`` at a run
with ``JIGSAW_NVTX=1`` to see, per ring step, whether the attention core idles
waiting on the KV send (comm not overlapped) or runs back-to-back.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator

import torch

_ENABLED = os.environ.get("JIGSAW_NVTX", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)


def enabled() -> bool:
    """Whether ``JIGSAW_NVTX`` selected the annotated path."""
    return _ENABLED


@contextlib.contextmanager
def nvtx_range(msg: str) -> Iterator[None]:
    """Push an NVTX range for the duration of the ``with`` block (no-op if disabled)."""
    if not _ENABLED:
        yield
        return
    torch.cuda.nvtx.range_push(msg)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
