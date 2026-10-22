"""Layout type and the conversion back from ``bshd`` for the TE cores.

:mod:`.cudnn` and :mod:`.flash` compute in ``bshd`` and accept a ``layout`` of
``"bhsd"`` (the ring's convention) or ``"bshd"``. Only the conversion back to
the caller's layout is shared. Each core converts into ``bshd`` itself: cuDNN
needs a contiguous copy, while FA2 only needs ``stride(-1) == 1`` and skips it.
"""

from __future__ import annotations

from typing import Literal

import torch

Layout = Literal["bhsd", "bshd"]


def from_bshd(t: torch.Tensor, layout: Layout) -> torch.Tensor:
    """Bring a kernel's native ``bshd`` tensor back to the caller's ``layout``.

    Parameters
    ----------
    t : torch.Tensor
        Tensor in ``bshd``.
    layout : {"bhsd", "bshd"}
        Layout to return.

    Returns
    -------
    torch.Tensor
        ``t`` in ``layout`` (``t`` itself for ``"bshd"``).
    """
    return t if layout == "bshd" else t.transpose(1, 2).contiguous()
