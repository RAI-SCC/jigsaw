"""Run a batching-rule-less ``autograd.Function`` backward under ``vmap(grad)``.

:class:`~.functional.RingAttentionFunction` folds a vmapped *ensemble* axis into
the batch in its ``vmap`` staticmethod and runs **one** un-vmapped forward --
the ring's per-chunk cores and its P2P ring exchange register no ``functorch``
batching rule, so that is the only way through.

Its ``backward`` has to do the mirror. ``functorch``'s ``autograd_function``
shim calls it again inside the ``grad`` transform, and on the ensemble path the
cotangent is a ``BatchedTensor`` wrapped in a ``GradTrackingTensor``; the ring
backward's collectives and matmuls dereference storage and see neither wrapper,
and it would in any case issue one async collective *per ensemble member*.

The saved tensors are a **mix**: ``vmap`` runs its nested forward on already
folded ``(members * batch, ...)`` inputs, so tensors *produced inside* that
forward (``lse``, the saved K/V chunks) arrive already folded and unbatched,
while the *inputs* ``functorch`` records for ``setup_context`` (``q`` / ``out``
/ ``x``) still carry a live ``vmap`` batched layer, and the projection weight
carries neither. :func:`run_backward_unwrapped` therefore inspects **each
tensor's own** wrapper chain: any tensor still carrying a ``vmap`` layer is
peeled and has that axis folded into its batch axis; the rest are only peeled.

``compute`` runs on a uniform ``(members * batch, ...)`` view with the dynamic
layer stack emptied (so its plain-tensor ops are not lifted back into the
transform). The grads it returns -- which correspond to the batched inputs --
are unfolded and then re-wrapped **layer by layer as the stack is restored**,
innermost first, so each wrap sees only the layers below it and nothing
double-wraps.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def functorch_active() -> bool:
    """Whether any ``functorch`` transform (``vmap`` / ``grad`` / ...) is live."""
    return torch._C._are_functorch_transforms_active()


def _chain(t: torch.Tensor) -> list[tuple[str, int, Optional[int]]]:
    """Return ``t``'s wrapper chain, outer -> inner: ``(kind, level, bdim)`` each."""
    import torch._C._functorch as _ft

    out: list[tuple[str, int, Optional[int]]] = []
    p = t
    while _ft.is_functorch_wrapped_tensor(p):
        if _ft.is_batchedtensor(p):
            out.append(("vmap", _ft.maybe_get_level(p), _ft.maybe_get_bdim(p)))
        else:
            out.append(("grad", _ft.maybe_get_level(p), None))
        p = _ft.get_unwrapped(p)
    return out


def _peel(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Strip every ``functorch`` wrapper off ``t`` down to the plain tensor."""
    import torch._C._functorch as _ft

    if t is None:
        return None
    while _ft.is_functorch_wrapped_tensor(t):
        t = _ft.get_unwrapped(t)
    return t


def run_backward_unwrapped(
    cotangent: torch.Tensor,
    saved: tuple[Optional[torch.Tensor], ...],
    compute: Callable[..., tuple[torch.Tensor, ...]],
) -> tuple[Optional[torch.Tensor], ...]:
    """``compute(cotangent, *saved) -> grads`` with ``functorch`` wrappers removed.

    Off the ``functorch`` path (an ordinary ``.backward()`` / ``autograd.grad``)
    this is exactly ``compute(cotangent, *saved)`` -- no behaviour change.

    Under a live transform: every tensor that still carries a ``vmap`` batched
    layer has that axis folded into axis 0 after peeling; tensors without one
    are only peeled. ``compute`` runs with the dynamic layer stack emptied, and
    each returned grad is unfolded along axis 0 and re-wrapped onto the
    cotangent's chain as the stack is rebuilt. ``saved`` / returned entries may
    be ``None`` (non-differentiable slots).
    """
    if not functorch_active():
        return compute(cotangent, *saved)

    import torch._C._functorch as _ft

    cot_chain = _chain(cotangent)  # outer -> inner
    members: Optional[int] = None

    def _prep(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Peel ``t``; if it carried a vmap layer, fold that axis into axis 0."""
        nonlocal members
        if t is None:
            return None
        bdim = next((bd for k, _, bd in _chain(t) if k == "vmap"), None)
        p = _peel(t)
        assert p is not None
        if bdim is None:
            return p
        members = p.shape[bdim]
        return p.movedim(bdim, 0).flatten(0, 1).contiguous()

    def _wrap(t: torch.Tensor, kind: str, bd: Optional[int], lvl: int) -> torch.Tensor:
        if kind == "vmap":
            assert bd is not None
            if bd != 0:
                t = t.movedim(0, bd)
            return _ft._add_batch_dim(t, bd, lvl)
        return _ft._wrap_for_grad(t, lvl)

    def _unfold(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None or members is None:
            return t
        return t.unflatten(0, (members, -1))

    saved_layers = []
    try:
        while _ft.get_dynamic_layer_stack_depth() > 0:
            saved_layers.append(_ft.pop_dynamic_layer_stack())

        raw = compute(_prep(cotangent), *(_prep(t) for t in saved))
        grads: list[Optional[torch.Tensor]] = [_unfold(g) for g in raw]

        # Restore the stack innermost-first, re-wrapping the grads for each layer
        # as it comes back so every wrap sees only the layers already below it
        # (the tail of the pop order lines up with ``cot_chain`` inner -> outer).
        inner = list(reversed(cot_chain))
        for i, layer in enumerate(reversed(saved_layers)):
            _ft.push_dynamic_layer_stack(layer)
            if i < len(inner):
                kind, lvl, bd = inner[i]
                grads = [
                    None if g is None else _wrap(g, kind, bd, lvl) for g in grads
                ]
        saved_layers = []  # fully restored in the loop above
    finally:
        for layer in reversed(saved_layers):
            _ft.push_dynamic_layer_stack(layer)

    return tuple(grads)
