"""Helpers for the ``vmap`` staticmethods of the custom autograd Functions.

A jigsaw Function's ``vmap`` staticmethod runs when ``torch.vmap`` reaches the
Function with at least one batched input. It receives the plain tensors, with
the vmapped axis at ``in_dims[i]`` (``None``: that input is not batched).
"""

from __future__ import annotations

from typing import Optional

import torch


def to_front(
    t: Optional[torch.Tensor], dim: Optional[int], size: int
) -> Optional[torch.Tensor]:
    """Put the vmapped axis of ``t`` at position 0.

    An unbatched ``t`` (``dim is None``) is expanded to ``size`` along a new
    leading axis.
    """
    if t is None:
        return None
    if dim is None:
        return t.expand(size, *t.shape)
    return t.movedim(dim, 0)


def check_not_vmapped(config: object) -> None:
    """Raise if ``config`` is already the vmapped config (nested ``vmap``)."""
    if getattr(config, "vmapped", False):
        raise NotImplementedError(
            "nested torch.vmap over a jigsaw layer is not supported"
        )
