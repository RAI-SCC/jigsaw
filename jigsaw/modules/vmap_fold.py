"""Getting a ``torch.vmap``ed ensemble through a kernel that has no batching rule.

Several kernels jigsaw builds on register no ``functorch`` batching rule:
``transformer_engine``'s fused attention, the ring's P2P chunk exchange, the
``c10d`` collectives. ``vmap`` therefore cannot dispatch a Bayesian ensemble
through them one member at a time. All of them do treat their leading axis as
an opaque batch, though, so the way through is always the same: **fold** the
vmapped (members) axis into the batch axis, run one un-vmapped call on an
ordinary ``[members * batch, ...]`` tensor, then **unfold**.

Forward and backward reach that fold by different routes.

*Forward* -- the ``autograd.Function`` overrides the ``vmap`` staticmethod.
``functorch`` peels its own wrappers off the operands before calling it and
reports each one's members axis in ``in_dims``, so all the staticmethod has to
do is :func:`fold` / :func:`unfold` (and report ``out_dim=0``).

*Backward* -- there is no such hook. ``functorch`` re-invokes ``backward``
inside the live ``grad`` transform with wrapped tensors and no ``in_dims``, so
:func:`run_unwrapped` does that same peeling by hand. See its docstring.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

import torch
import torch._C._functorch as _ft


def fold(t: torch.Tensor, bdim: Optional[int], members: int) -> torch.Tensor:
    """Fold the members axis at ``bdim`` into the batch axis.

    ``[..., members, batch, ...] -> [members * batch, ...]``.

    ``bdim is None`` means ``t`` carries no members axis, so it is broadcast
    over all ``members`` first -- the kernel needs one entry per member.
    """
    t = t.expand(members, *t.shape) if bdim is None else t.movedim(bdim, 0)
    return t.flatten(0, 1).contiguous()


def unfold(t: torch.Tensor, members: int) -> torch.Tensor:
    """Inverse of :func:`fold`: ``[members * batch, ...] -> [members, batch, ...]``."""
    return t.unflatten(0, (members, -1))


# --------------------------------------------------------------------------- #
# backward: do functorch's peeling by hand
# --------------------------------------------------------------------------- #
def _members_axis(t: torch.Tensor) -> Optional[int]:
    """Position of ``t``'s vmapped (members) axis, or ``None`` if it has none."""
    while _ft.is_functorch_wrapped_tensor(t):
        if _ft.is_batchedtensor(t):
            return _ft.maybe_get_bdim(t)
        t = _ft.get_unwrapped(t)
    return None


def _peel(t: torch.Tensor) -> torch.Tensor:
    """Strip every ``functorch`` wrapper off ``t``, down to the plain tensor."""
    while _ft.is_functorch_wrapped_tensor(t):
        t = _ft.get_unwrapped(t)
    return t


def run_unwrapped(
    cotangent: torch.Tensor,
    saved: Sequence[Optional[torch.Tensor]],
    compute: Callable[..., tuple[torch.Tensor, ...]],
) -> tuple[torch.Tensor, ...]:
    """``compute(cotangent, *saved) -> grads``, run on plain, folded tensors.

    Outside a transform this is exactly ``compute(cotangent, *saved)``.

    Under ``vmap(grad(...))`` each tensor is an onion of ``functorch`` wrappers
    -- ``GradTracking`` over ``Batched`` over the plain tensor -- and the plain
    tensor at the core is the only one with real storage, so a fused kernel
    handed the wrapper gets an exception instead of data. This does by hand what
    ``functorch`` does for a ``vmap`` staticmethod:

    1. find each tensor's members axis (``None`` if it has none) and peel it;
    2. :func:`fold` the ones that have such an axis. The rest pass through --
       either they were produced *inside* the folded forward and already carry
       the folded batch (log-sum-exps, saved K/V chunks, RNG state), or they are
       genuinely shared across members (the projection weight);
    3. empty the dynamic layer stack, so ``compute``'s plain-tensor ops are not
       lifted straight back into the transform, and run it;
    4. :func:`unfold` the grads and rebuild the wrapper onion on them, innermost
       layer first, so autograd upstream gets tensors shaped like the ones it
       handed us.

    Step 4 has no safety net: a wrong level or axis here yields silently wrong
    gradients rather than an error. The ``vmap``-vs-per-member-loop parity tests
    in ``tests/modules/attention/test_te_vmap_bayesian.py`` are what pin it.

    Exactly one ``vmap`` layer is supported (one ensemble axis); nested
    ``vmap``s are rejected rather than silently folded wrong.
    """
    if _ft.maybe_current_level() is None:
        return compute(cotangent, *saved)

    members: Optional[int] = None

    def prep(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        nonlocal members
        if t is None:
            return None
        bdim = _members_axis(t)
        p = _peel(t)
        if bdim is None:
            return p
        members = p.shape[bdim]
        return fold(p, bdim, members)

    # ``cotangent``'s wrapper chain, outermost layer first: one entry per live
    # transform, ``axis`` being the members axis for a ``vmap`` layer and
    # ``None`` for a ``grad`` layer (which hides no axis).
    chain: list[tuple[int, Optional[int]]] = []
    probe = cotangent
    while _ft.is_functorch_wrapped_tensor(probe):
        batched = _ft.is_batchedtensor(probe)
        chain.append(
            (_ft.maybe_get_level(probe), _ft.maybe_get_bdim(probe) if batched else None)
        )
        probe = _ft.get_unwrapped(probe)

    if sum(axis is not None for _, axis in chain) > 1:
        raise NotImplementedError(
            "nested vmap over a kernel with no batching rule: only one ensemble "
            f"axis can be folded into the batch, found {len(chain)} layers"
        )

    layers = []
    try:
        while _ft.get_dynamic_layer_stack_depth() > 0:
            layers.append(_ft.pop_dynamic_layer_stack())
        grads = [
            g if members is None else unfold(g, members)
            for g in compute(prep(cotangent), *(prep(t) for t in saved))
        ]
    finally:
        for layer in reversed(layers):
            _ft.push_dynamic_layer_stack(layer)

    for level, axis in reversed(chain):  # innermost layer first
        if axis is None:
            grads = [_ft._wrap_for_grad(g, level) for g in grads]
        else:
            grads = [
                _ft._add_batch_dim(g if axis == 0 else g.movedim(0, axis), axis, level)
                for g in grads
            ]
    return tuple(grads)
