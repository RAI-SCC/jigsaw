"""Batching rule shared by the convolution autograd Functions."""

from typing import Any, Optional

import torch

from jigsaw._core.vmap import check_not_vmapped, to_front


def conv_vmap(
    fn: Any,
    info: Any,
    in_dims: tuple[Optional[int], ...],
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *args: Any,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
    """Batching rule of :class:`_ConvNd` and :class:`_ConvTransposeNd`.

    Makes one un-vmapped ``fn`` call (``fn(x, weight, bias, *args)``, the last
    of ``args`` being the config), so every collective runs once.

    * Weight and bias not vmapped: the vmapped axis is folded into the batch
      axis.
    * Weight or bias vmapped (e.g. stacked ensemble members): ``fn`` runs with
      the config's :meth:`~jigsaw._core.vmap.VmapConfig.for_vmap`, which vmaps
      the local conv primitives over the leading members axis.
    """
    *rest, config = args
    check_not_vmapped(config)
    size = info.batch_size
    x_dim, w_dim, b_dim = in_dims[:3]
    x = to_front(x, x_dim, size)  # type: ignore[assignment]
    # Each member's input has no batch axis of its own.
    no_batch = x.ndim == config.n_spatial + 2

    # Only ``x`` vmapped, shared weight and bias (e.g. per-example vmap, or
    # perturbed inputs through one model): plain conv over a bigger batch.
    if w_dim is None and b_dim is None:
        # Unbatched samples (``(C, *spatial)``): the vmapped axis is the batch.
        if no_batch:
            return fn(x.contiguous(), weight, bias, *args), (0, 0)
        # Batched samples: fold the vmapped axis into the batch, then unfold.
        y, x_aug = fn(x.flatten(0, 1).contiguous(), weight, bias, *args)
        return (y.unflatten(0, (size, -1)), x_aug.unflatten(0, (size, -1))), (0, 0)

    # Weight or bias vmapped (ensemble members, per-sample weights): run
    # the members config. ``to_front`` expands whichever of x/weight/bias is
    # shared, so no input needs a pre-expanded members axis.
    weight = to_front(weight, w_dim, size).contiguous()  # type: ignore[union-attr]
    bias = to_front(bias, b_dim, size)
    if no_batch:
        # The members primitives expect ``(members, batch, C, *spatial)``.
        x = x.unsqueeze(1)
    y, x_aug = fn(x.contiguous(), weight, bias, *rest, config.for_vmap())
    if no_batch:
        y, x_aug = y.squeeze(1), x_aug.squeeze(1)
    return (y, x_aug), (0, 0)
