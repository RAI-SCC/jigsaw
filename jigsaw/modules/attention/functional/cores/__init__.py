"""Attention cores for the ring's per-chunk inner attention.

Each submodule (:mod:`.sdpa`, :mod:`.math`, :mod:`.te`) provides the same pair
of functions::

    forward(q, k, v, scale, dropout_p, seed=None) -> (out, logsumexp, rng_state)
    backward(grad_out, q, k, v, out, logsumexp, scale, *extra,
             dropout_p=0.0, seed=None, rng_state=None) -> (dq, dk, dv)

Tensors have shape ``(b, h, s, d)`` and ``logsumexp`` has shape ``(b, h, s)``.
:mod:`.math` replays dropout from ``seed``, while :mod:`.sdpa` and :mod:`.te`
return the kernel's ``rng_state`` from ``forward`` for ``backward`` to replay.
"""
