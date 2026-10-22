"""Layout type and the shared half of the bshd conversion for the TE cores.

Both :mod:`.cudnn` and :mod:`.flash` speak ``bshd`` natively and accept a
caller ``layout`` of ``"bhsd"`` (the ring's convention) or ``"bshd"``. Only
the "back to caller layout" transpose is identical between them, so just that
piece lives here -- the "into bshd" direction is deliberately *not* shared:
cuDNN needs its input contiguous, FA2 needs it not to be (see
:func:`.flash._to_bshd`).
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
