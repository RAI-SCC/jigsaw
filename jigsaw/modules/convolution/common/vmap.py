"""Shared ``torch.vmap`` staticmethod body for the distributed conv Functions.

:class:`~jigsaw.modules.convolution.conv.functional.ConvNd` and
:class:`~jigsaw.modules.convolution.conv_transpose.functional.ConvTransposeNd`
both expose an ensemble (``bayesian``) axis purely through their ``vmap``
staticmethod -- there is no construction-time flag. ``torch.vmap`` hands each
operand with its mapped dimension at ``in_dims[i]``; :func:`align_members_axis`
moves it to axis 0 (broadcasting an unmapped operand across the batch) so the
Function can treat the members axis as an ordinary extra leading batch
dimension. The channel-gather / halo collectives are already dimension-agnostic
(negative axes throughout), so they run exactly once -- ``E`` x the payload, not
``E`` rounds -- while the per-member weights ride a ``torch.vmap`` of only the
*local*, collective-free ``F.conv*`` (see the ``ConvNdConfig`` dispatchers).
"""

from typing import Any, Optional

import torch


def align_members_axis(
    info: Any,
    in_dims: tuple[Optional[int], ...],
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Expose the mapped members axis as a contiguous leading dim on every operand.

    ``in_dims[:3]`` are the mapped dimensions of ``x`` / ``weight`` / ``bias``
    (``None`` when that operand is not mapped, e.g. a shared input or a
    parameter that is not part of the ensemble). The mapped dim is moved to 0;
    an unmapped operand is broadcast to ``info.batch_size`` so ``x`` / ``weight``
    / ``bias`` all carry ``E`` in front. ``x`` is made contiguous for the
    downstream collectives; ``weight`` / ``bias`` are left as views (the kernels
    slice + ``contiguous`` them where needed).
    """
    xd, wd, bd = in_dims[0], in_dims[1], in_dims[2]
    x = x.movedim(xd, 0) if xd is not None else x.expand(info.batch_size, *x.shape)
    weight = (
        weight.movedim(wd, 0)
        if wd is not None
        else weight.expand(info.batch_size, *weight.shape)
    )
    if bias is not None:
        bias = (
            bias.movedim(bd, 0)
            if bd is not None
            else bias.expand(info.batch_size, *bias.shape)
        )
    return x.contiguous(), weight, bias
