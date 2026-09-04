"""Per-chunk inner-attention cores for the ring.

Each submodule -- :mod:`.sdpa`, :mod:`.math`, :mod:`.te` -- exposes the same pair
of callables so the ring loop never branches on the backend::

    forward(q, k, v, scale, dropout_p) -> (out, logsumexp)
    backward(grad_out, q, k, v, out, logsumexp, scale, *extra) -> (dq, dk, dv)

Tensors are in the ring's ``(b, h, s, d)`` layout and ``logsumexp`` is
``(b, h, s)``. :func:`..config._select_attn_core_forward_fn` /
:func:`..config._select_attn_core_backward_fn` pick one pair per
:class:`~..config.AttnBackend` (and, for ``SDPA``, per device).

Submodules are imported lazily by the selectors, so pulling in :mod:`..config`
never imports :mod:`transformer_engine`.
"""
