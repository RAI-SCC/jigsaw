"""Distributed attention."""

from jigsaw._core.public import set_module as _set_module

# JIGSAW_TE_AVAILABLE is a read-only copy of the flag: patching it here does not
# change which backend DistributedAttention picks; patch ``_attention``'s.
from ._attention import JIGSAW_TE_AVAILABLE, Attention, DistributedAttention
from ._config import (
    AttnBackend,
    Offload,
    RingAttentionConfig,
    SaveBackward,
    SaveScope,
    SendBackward,
    SendForward,
    SendOp,
)

__all__ = [
    "Attention",
    "DistributedAttention",
    "RingAttentionConfig",
    "AttnBackend",
    "SendForward",
    "SendOp",
    "SaveBackward",
    "SaveScope",
    "SendBackward",
    "Offload",
    "JIGSAW_TE_AVAILABLE",
]

_set_module(globals(), __all__)
