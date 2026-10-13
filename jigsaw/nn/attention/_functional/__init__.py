"""Functional implementation of ring and Ulysses context-parallel attention."""

from .ring import (
    HybridTopology,
    get_hybrid_topology,
)
from .ulysses import hierarchical_ulysses_attention, ulysses_attention

__all__ = [
    "HybridTopology",
    "get_hybrid_topology",
    "hierarchical_ulysses_attention",
    "ulysses_attention",
]
