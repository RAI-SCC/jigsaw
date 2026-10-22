"""Attention cores for the ring's per-chunk inner attention.

Each core (:mod:`.sdpa`, :mod:`.math`, :mod:`.te.cudnn`, :mod:`.te.flash`)
exposes::

    forward(q, k, v, scale, dropout_p, seed=None, *, layout=...)
        -> (out, logsumexp, rng_state)
    backward(grad_out, q, k, v, out, logsumexp, scale, *extra,
             dropout_p=0.0, seed=None, rng_state=None, layout=...) -> (dq, dk, dv)

Tensors are ``(b, h, s, d)`` (``bshd`` also accepted by :mod:`.te`), and
``logsumexp`` is ``(b, h, s)``. Dropout is replayed from ``seed`` in
:mod:`.math` and from the kernel's ``rng_state`` elsewhere. Submodules are
imported lazily, so :mod:`.._config` never imports ``transformer_engine``.
"""
