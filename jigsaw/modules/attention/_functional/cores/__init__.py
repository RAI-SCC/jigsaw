"""Attention cores for the ring's per-chunk inner attention.

Each core -- :mod:`.sdpa`, :mod:`.math`, and the two arms of :mod:`.te`
(:mod:`.te.cudnn`, :mod:`.te.flash`) -- exposes the same pair of callables so
the ring loop never branches on the backend::

    forward(q, k, v, scale, dropout_p, seed=None, *, layout=...)
        -> (out, logsumexp, rng_state)
    backward(grad_out, q, k, v, out, logsumexp, scale, *extra,
             dropout_p=0.0, seed=None, rng_state=None, layout=...) -> (dq, dk, dv)

Tensors are the ring's ``(b, h, s, d)`` layout and ``logsumexp`` is
``(b, h, s)``; the :mod:`.te` cores also accept ``layout="bshd"`` for the
non-ring core, skipping transposes their kernels would otherwise need.
:func:`.._config._select_attn_core_forward_fn`/``_backward_fn`` pick one pair
per :class:`~.._config.AttnBackend` (``SDPA`` per device, ``TE`` per GPU
architecture -- see :func:`.te.select_core`).

Dropout-mask replay -- the backward recomputes ``dQ/dK/dV`` from the *saved*
global ``out``/``logsumexp`` rather than a live autograd graph, so forward's
mask must match exactly -- splits two ways: :mod:`.math` has no opaque kernel
state, so it derives the mask from ``seed`` (independently recomputed by both
ring loops from one shared base seed, see
:func:`.._functional._combine_dropout_seed`); :mod:`.sdpa` and both :mod:`.te`
arms wrap kernels with their own opaque RNG state, which they return from
``forward`` and the ring threads back verbatim into ``backward``.

Submodules are imported lazily by the selectors, so pulling in :mod:`.._config`
never imports :mod:`transformer_engine`.
"""
