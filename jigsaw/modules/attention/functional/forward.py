"""Ring attention forward implementation.

A single driver loop (:func:`ring_forward`) handles every communication strategy
by delegating to a :class:`~.forward_chunk_provider._ForwardChunkProvider`,
mirroring how :func:`.backward.ring_backward` delegates to a
:class:`~.backward_chunk_provider._BackwardChunkProvider`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from jigsaw.modules.attention.config import RingAttentionConfig

from ._merge_kernel import fused_online_softmax_merge_
from .forward_chunk_provider import build_forward_chunk_provider
from .functional import bhnd, combine_dropout_seed


def _draw_base_seed(device: torch.device) -> int:
    """Draw a fresh per-call dropout base seed from the ambient default RNG.

    Consumed once per ring-attention forward; every per-chunk sub-seed
    (:func:`..functional.combine_dropout_seed`) derives from it deterministically,
    so only this one int needs saving for backward to replay every chunk's mask.
    """
    return int(torch.randint(0, 2**31 - 1, (1,), device=device).item())


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def _online_softmax_merge_(
    o1: torch.Tensor,
    l1: torch.Tensor,
    o2: torch.Tensor,
    l2: torch.Tensor,
    layout: str = "bhsd",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge a partial online-softmax result ``(o2, l2)`` into ``(o1, l1)`` in place.

    ``o1`` and ``l1`` are the ring's running accumulators, owned by
    :func:`ring_forward` and aliased nowhere else, so this mutates them in place:
    no per-step ``(b, h, n, d)`` / ``(b, h, n)`` allocation, and the accumulator
    itself is allocated once instead of being rebuilt every ring step.

    Algebra (``l1``/``l2`` are log-sum-exps, not raw maxima): with
    ``l_new = logaddexp(l1, l2)`` the two blend weights of the old
    ``exp(l1 - l_new) * o1 + exp(l2 - l_new) * o2`` are just
    ``1 - w`` and ``w`` for ``w = exp(l2 - l_new) = sigmoid(l2 - l1)``, so the
    running-max cancels and only one small ``(b, h, n)`` weight is materialised.
    The output update is then ``o1 <- (1 - w) * o1 + w * o2`` as two in-place
    big-tensor kernels (``mul_`` then ``addcmul_``) with no temporaries -- down
    from three big kernels and three big allocations. ``addcmul_`` promotes
    ``o2`` (fp16/bf16 from the core) into ``o1``'s fp32 accumulator for free.
    At the seed state ``l1 = -inf`` -> ``w = sigmoid(+inf) = 1`` -> ``o1 := o2``.

    ``l1`` (small, ``(b, h, n)``) is returned fresh rather than updated in place:
    an ``out=`` ``logaddexp`` cannot run when ``l2`` still carries autograd
    history (the TE cores return a grad-tracked ``lse``), and the caller rebinds
    it anyway.

    ``o1``/``o2`` follow ``layout`` (``"bhsd"`` or ``"bshd"``); ``l1``/``l2`` are
    ``(b, h, n)`` either way (every core emits the LSE in that shape), so for
    ``"bshd"`` the blend weight is transposed to ``(b, n, h, 1)`` before it
    broadcasts over ``o1``'s trailing ``d``.

    Parameters
    ----------
    o1, l1 : torch.Tensor
        Running output (``layout``-shaped, updated in place) and log-sum-exp
        ``(b, h, n)``, both at least fp32.
    o2, l2 : torch.Tensor
        This step's partial output and log-sum-exp.
    layout : str
        Layout of ``o1`` / ``o2``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``o1`` (mutated) and the new ``l1``.

    A Triton kernel (:func:`._merge_kernel.fused_online_softmax_merge_`) fuses
    the whole thing -- ``sigmoid`` + the blend + ``logaddexp`` -- into one launch
    with a single read of ``o2`` and one read-modify-write of ``o1``, ~1.7x
    faster than the eager path below. It returns ``None`` (and this falls back to
    eager) when Triton is unavailable or the inputs are outside its supported
    set.
    """
    fused = fused_online_softmax_merge_(o1, l1, o2, l2, layout)
    if fused is not None:
        return fused

    w = torch.sigmoid(l2 - l1)  # exp(l2 - logaddexp(l1, l2)), in (0, 1); (b, h, n)
    if layout == "bshd":
        w = w.transpose(1, 2)  # (b, n, h)
    w = w.unsqueeze(-1)
    o1.mul_(1 - w).addcmul_(w, o2)
    return o1, torch.logaddexp(l1, l2)


def _attend_step(
    ring_config: RingAttentionConfig,
    q: torch.Tensor,
    kj: torch.Tensor,
    vj: torch.Tensor,
    scale: float,
    dropout_p: float,
    out: torch.Tensor,
    lse: torch.Tensor,
    seed: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Run the attention core on (q, kj, vj) and merge into the running (out, lse).

    kj/vj may be views into a KV buffer; flash attention requires contiguous
    inputs, so they're materialized here. Returns the merged ``(out, lse)``
    plus this step's ``rng_state`` (``None`` unless the core is one that
    threads back opaque kernel RNG state instead of using ``seed`` -- see
    :mod:`.cores`).
    """
    kjc, vjc = kj.contiguous(), vj.contiguous()
    out_j, l_j, rng_state_j = ring_config.attn_core_forward_fn(
        q,
        kjc,
        vjc,
        scale,
        dropout_p,
        seed=seed,
        layout=ring_config.core_layout,
    )

    out, lse = _online_softmax_merge_(
        out, lse, out_j, l_j, layout=ring_config.core_layout
    )
    return out, lse, rng_state_j


# --------------------------------------------------------------------------- #
# forward driver
# --------------------------------------------------------------------------- #
def ring_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    scale: float,
    dropout_p: float,
    group: dist.ProcessGroup,
    local_rank: int,
    ring_group_size: int,
    ring_config: RingAttentionConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor | None, ...],
    Optional[int],
    tuple[torch.Tensor | None, ...],
]:
    """Forward pass for ring attention.

    One loop over ``ring_group_size`` ring steps: the
    :class:`~.forward_chunk_provider._ForwardChunkProvider` for
    ``ring_config.send_op`` makes each chunk current (``get``), records it for
    the backward pass (``save_for_backward``), and progresses communication
    (``advance``); this function only runs the attention core and the online
    softmax merge.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Local query, key, and value tensors, shaped ``(b, h, n_local, d)``.
    x : torch.Tensor
        Local input sequence chunk.
    w_qkv : torch.Tensor
        Fused query, key, and value projection weights.
    scale : float
        Attention scaling factor.
    dropout_p : float
        Attention dropout probability.
    group : dist.ProcessGroup
        Distributed process group used for ring communication.
    local_rank : int
        Rank within ``group``.
    ring_group_size : int
        Number of ranks in ``group``.
    ring_config : RingAttentionConfig
        Ring-attention configuration.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...], int | None, tuple[torch.Tensor | None, ...]]
        Attention output, log-sum-exp values, the chunks saved for backward,
        the dropout base seed (``None`` if ``dropout_p == 0``), and per-global-chunk
        RNG state (see :func:`_attend_step`).
    """
    # Accumulate ``out`` / ``lse`` in at least fp32, mutated in place across ring
    # steps by ``_online_softmax_merge_``. ``lse`` is a softmax normaliser, never
    # matmul'd, and the fused SDPA/TE backward kernels read it as fp32 -- a
    # 16-bit ``lse`` makes them recompute the softmax wrong (see
    # ``backward.ring_backward``); the old code only reached fp32 here by
    # accident of dtype promotion. ``out`` is cast back to ``q.dtype`` on return.
    # An fp64 model keeps fp64 for both. Shapes track ``q`` so this is agnostic
    # to ``ring_config.core_layout`` (bhsd / bshd).
    # Build the provider first: its ``__init__`` kicks the ring's first KV
    # send (a P2P/all-to-all whose bytes then move on the NCCL stream while the
    # host sets everything else up). The ``out``/``lse`` zero-fills and the
    # dropout seed below don't depend on it, so issuing the send ahead of them
    # overlaps the send launch with that host work -- it matters most for the
    # hierarchical Ulysses inter-node ring, where that first send crosses the
    # slow fabric.
    provider = build_forward_chunk_provider(
        ring_config, q, k, v, x, w_qkv, group, local_rank, ring_group_size
    )

    acc_dtype = torch.promote_types(q.dtype, torch.float32)
    b, heads, n_local, _ = bhnd(q.shape, ring_config.core_layout)
    out = torch.zeros(q.shape, device=q.device, dtype=acc_dtype)
    # ``lse`` is always ``(b, h, n)`` -- both the SDPA and TE cores emit and
    # consume it in that shape regardless of the q/k/v layout; the merge
    # transposes the blend weight instead (see ``_online_softmax_merge_``).
    lse = torch.full((b, heads, n_local), float("-inf"), device=q.device, dtype=acc_dtype)

    base_seed = _draw_base_seed(q.device) if dropout_p > 0.0 else None
    rng_states: list[Optional[torch.Tensor]] = [None] * ring_group_size

    for step in range(ring_group_size):
        kj, vj = provider.get(step)
        provider.save_for_backward(step)
        global_chunk_idx = provider.global_chunk_idx(step)
        seed = (
            combine_dropout_seed(base_seed, local_rank, global_chunk_idx)
            if base_seed is not None
            else None
        )
        out, lse, rng_state_j = _attend_step(
            ring_config, q, kj, vj, scale, dropout_p, out, lse, seed
        )
        rng_states[global_chunk_idx] = rng_state_j
        provider.advance(step, ring_group_size)

    ring_config.wait_offloads()
    out = out.to(q.dtype)
    return out, lse, provider.result(), base_seed, tuple(rng_states)
