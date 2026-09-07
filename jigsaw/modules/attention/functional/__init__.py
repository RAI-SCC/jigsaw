"""Functional implementations for distributed attention."""

from .backward import ring_backward
from .forward import ring_forward
from .functional import (
    HybridTopology,
    RingAttentionFunction,
    get_hybrid_topology,
)

__all__ = [
    "HybridTopology",
    "RingAttentionFunction",
    "get_hybrid_topology",
    "ring_backward",
    "ring_forward",
]
