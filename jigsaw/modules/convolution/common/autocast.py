"""Autocast handling shared by the distributed convolution Functions."""

from typing import Optional

import torch


def cast_forward_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Cast ``(x, weight, bias)`` to the active autocast dtype, else pass through.

    A :class:`torch.autograd.Function` is opaque to autocast's op-level policy,
    so -- exactly like ``XWT.forward`` -- the conv Functions opt in here. The
    distributed conv kernels size their channel-gather / halo buffers from
    ``x`` / ``weight`` and only then call ``F.conv*`` on the gathered result;
    casting up front makes the whole pipeline (buffers included) run in the
    low-precision dtype and lets the output carry it. Backward upcasts the saved
    activation to ``weight.dtype`` (the master-weight precision) to match
    ``grad_output``.
    """
    if not torch.is_autocast_enabled(x.device.type):
        return x, weight, bias
    dtype = torch.get_autocast_dtype(x.device.type)
    return (
        x.to(dtype),
        weight.to(dtype),
        None if bias is None else bias.to(dtype),
    )
