"""Per-chunk inner-attention cores for the ring.

Each submodule -- :mod:`.sdpa`, :mod:`.math`, :mod:`.te` -- exposes the same pair
of callables so the ring loop never branches on the backend::

    forward(q, k, v, scale, dropout_p, seed=None) -> (out, logsumexp, rng_state)
    backward(grad_out, q, k, v, out, logsumexp, scale, *extra,
             dropout_p=0.0, seed=None, rng_state=None) -> (dq, dk, dv)

Tensors are in the ring's ``(b, h, s, d)`` layout and ``logsumexp`` is
``(b, h, s)``. :func:`..config._select_attn_core_forward_fn` /
:func:`..config._select_attn_core_backward_fn` pick one pair per
:class:`~..config.AttnBackend` (and, for ``SDPA``, per device).

Dropout-mask replay (forward's mask must match exactly in backward, since the
ring's backward recomputes ``dQ/dK/dV`` from the *saved* global ``out`` /
``logsumexp`` rather than differentiating through a live autograd graph) is
split two ways:

* :mod:`.math` has no opaque kernel state to thread back, so it derives its
  mask from ``seed`` (an int, independently recomputed by both the forward and
  backward ring loops from one shared per-call base seed -- see
  :func:`..functional.combine_dropout_seed`) and always returns/ignores
  ``rng_state``.
* :mod:`.sdpa` / :mod:`.te` wrap kernels that generate their own opaque
  philox / cuDNN RNG state internally; they ignore ``seed`` and instead return
  the real ``rng_state`` from ``forward``, which the ring threads back
  verbatim into ``backward`` (per ring step, keyed by global chunk id -- see
  :func:`..forward.ring_forward` / :func:`..backward.ring_backward`).

Submodules are imported lazily by the selectors, so pulling in :mod:`..config`
never imports :mod:`transformer_engine`.
"""
