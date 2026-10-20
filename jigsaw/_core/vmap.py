"""Helpers for the ``vmap`` staticmethods of the custom autograd Functions.

A jigsaw Function's ``vmap`` staticmethod runs when ``torch.vmap`` reaches the
Function with at least one batched input. It receives the plain tensors, with
the vmapped axis at ``in_dims[i]`` (``None``: that input is not batched).
"""

from __future__ import annotations

import copy
from typing import Optional, TypeVar, cast

import torch

_C = TypeVar("_C", bound="VmapConfig")


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


class VmapConfig:
    """Mixin for kernel configs with a variant for vmapped layer weights.

    A Function's ``vmap`` staticmethod passes :meth:`for_vmap` instead of the
    config when the layer weight is vmapped (e.g. stacked ensemble members), so
    the vmapped axis rides through the kernels as a leading members axis.
    """

    #: Whether this is the config for a vmapped layer weight.
    vmapped: bool = False
    _vmap_config: Optional[VmapConfig] = None

    def for_vmap(self: _C) -> _C:
        """Return the config for a vmapped layer weight (cached)."""
        if self.vmapped:
            return self
        if self._vmap_config is None:
            c = copy.copy(self)
            c.vmapped = True
            # Rebind the selected methods to the copy, so they see ``vmapped``.
            for name, value in vars(self).items():
                if getattr(value, "__self__", None) is self:
                    setattr(c, name, getattr(c, value.__func__.__name__))
            c._select_vmapped()
            self._vmap_config = c
        return cast(_C, self._vmap_config)

    def _select_vmapped(self) -> None:
        """Swap in the members-axis kernels on the vmapped copy.

        Hook for subclasses, called by :meth:`for_vmap` on the copy. The default
        is a no-op, for configs whose kernels need no swap.
        """

    def __getstate__(self) -> dict:
        """Drop the cached vmapped config (it holds ``torch.vmap`` closures)."""
        state = self.__dict__.copy()
        state.pop("_vmap_config", None)
        return state
