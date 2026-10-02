"""Ring attention forward pass."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from jigsaw.modules.attention.config import RingAttentionConfig

from ._merge_kernel import fused_online_softmax_merge_
from .forward_chunk_provider import build_forward_chunk_provider
from .ring import _bhnd, _combine_dropout_seed


def _draw_base_seed(device: torch.device) -> int:
    """Draw a dropout base seed from the default RNG."""
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
    causal: bool = False,
    key_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Attend to one chunk and merge it into the running output.

    kj/vj may be views into a KV buffer; flash attention requires contiguous
    inputs, so they're materialized here. ``causal`` masks the diagonal block
    (see :func:`_causal_step_kind`); it is never set together with a skip.
    ``key_mask`` is this step's ``[B, n_local]`` slice of the global
    key-padding mask (see :func:`ring_forward`).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        Merged output and log-sum-exp, and the ``rng_state`` of the core
        (``None`` unless the core is one that threads back opaque kernel RNG
        state instead of using ``seed`` -- see :mod:`.cores`).
    """
    kjc, vjc = kj.contiguous(), vj.contiguous()
    out_j, l_j, rng_state_j = ring_config.attn_core_forward_fn(
        q,
        kjc,
        vjc,
        scale,
        dropout_p,
        causal=causal,
        key_mask=key_mask,
        seed=seed,
        layout=ring_config.core_layout,
    )

    out, lse = _online_softmax_merge_(
        out, lse, out_j, l_j, layout=ring_config.core_layout
    )
    return out, lse, rng_state_j


def _causal_step_kind(kv_global_idx: int, q_global_idx: int) -> Optional[bool]:
    """Classify a ring step's K/V chunk relative to this rank's own Q chunk.

    Both chunks are ``n_local``-sized, contiguous, evenly-split shards of the
    global sequence, so global chunk id alone (not position within the chunk)
    decides visibility:

    * ``kv_global_idx > q_global_idx``: every key is strictly in the future of
      every query in this chunk -- fully masked, so the caller skips the step
      entirely (returns ``None``).
    * ``kv_global_idx == q_global_idx``: the "diagonal" block -- q and k start
      at the same global offset, so the plain top-left triangular mask is the
      causal boundary (returns ``True``).
    * ``kv_global_idx < q_global_idx``: every key is at or before every query
      in this chunk -- fully visible, no mask needed (returns ``False``).
    """
    if kv_global_idx > q_global_idx:
        return None
    return kv_global_idx == q_global_idx


# --------------------------------------------------------------------------- #
# forward driver
# --------------------------------------------------------------------------- #
def _gather_key_padding_mask(
    key_padding_mask: torch.Tensor, group: dist.ProcessGroup, ring_group_size: int
) -> torch.Tensor:
    """All-gather local ``[B, n_local]`` key-padding shards into ``[B, S_global]``.

    A one-time, small collective (bandwidth negligible next to the K/V ring
    traffic) done once per forward call, so every rank can slice out the
    ``[B, n_local]`` key-validity slab for whatever global chunk it currently
    holds (via ``provider.global_chunk_idx(step)``) without threading a new
    tensor through every :class:`~.forward_chunk_provider._ForwardChunkProvider`
    subclass's communication path.
    """
    shards = [torch.empty_like(key_padding_mask) for _ in range(ring_group_size)]
    dist.all_gather(shards, key_padding_mask.contiguous(), group=group)
    return torch.cat(shards, dim=1)


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
    key_padding_mask: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor | None, ...],
    Optional[int],
    tuple[torch.Tensor | None, ...],
]:
    """Run the forward pass of ring attention.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Local query, key, and value tensors, shaped ``(b, h, n_local, d)``.
    x : torch.Tensor
        Local input sequence chunk.
    w_qkv : torch.Tensor
        Fused query, key, and value projection weights.
    scale : float
        Softmax scale.
    dropout_p : float
        Dropout probability.
    group : dist.ProcessGroup
        Process group of the ring.
    local_rank : int
        Rank within ``group``.
    ring_group_size : int
        Number of ranks in ``group``.
    ring_config : RingAttentionConfig
        Ring-attention configuration.
    key_padding_mask : Optional[torch.Tensor]
        This rank's local shard (``[B, n_local]``, bool, ``True`` = attend) of
        a global key-padding mask.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...], int | None, tuple[torch.Tensor | None, ...]]
        Output, log-sum-exp, chunks saved for backward, dropout base seed
        (``None`` without dropout) and per-chunk RNG states.
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
    b, heads, n_local, _ = _bhnd(q.shape, ring_config.core_layout)
    out = torch.zeros(q.shape, device=q.device, dtype=acc_dtype)
    # ``lse`` is always ``(b, h, n)`` -- both the SDPA and TE cores emit and
    # consume it in that shape regardless of the q/k/v layout; the merge
    # transposes the blend weight instead (see ``_online_softmax_merge_``).
    lse = torch.full(
        (b, heads, n_local), float("-inf"), device=q.device, dtype=acc_dtype
    )

    global_key_mask = None
    if key_padding_mask is not None:
        global_key_mask = _gather_key_padding_mask(
            key_padding_mask, group, ring_group_size
        )

    base_seed = _draw_base_seed(q.device) if dropout_p > 0.0 else None
    rng_states: list[Optional[torch.Tensor]] = [None] * ring_group_size

    for step in range(ring_group_size):
        kj, vj = provider.get(step)
        provider.save_for_backward(step)
        chunk_idx = provider.global_chunk_idx(step)
        step_causal = False
        if ring_config.causal:
            kind = _causal_step_kind(chunk_idx, local_rank)
            if kind is None:
                provider.advance(step, ring_group_size)
                continue
            step_causal = kind
        step_key_mask = (
            global_key_mask[:, chunk_idx * n_local : (chunk_idx + 1) * n_local]
            if global_key_mask is not None
            else None
        )
        seed = (
            _combine_dropout_seed(base_seed, local_rank, chunk_idx)
            if base_seed is not None
            else None
        )
        # A chunk recomputed from a received ``x`` comes out in ``w_qkv``'s dtype,
        # while ``q`` is already in the core dtype (bf16 on CUDA): the cores
        # matmul ``q`` against the chunk, so a mixed-dtype pair raises. Fold the
        # chunk into ``q``'s dtype here, as ``ring_backward`` does for its cores;
        # the provider keeps (and saves) the chunk in its original precision.
        out, lse, rng_state_j = _attend_step(
            ring_config,
            q,
            kj.to(q.dtype),
            vj.to(q.dtype),
            scale,
            dropout_p,
            out,
            lse,
            seed,
            causal=step_causal,
            key_mask=step_key_mask,
        )
        rng_states[chunk_idx] = rng_state_j
        provider.advance(step, ring_group_size)

    ring_config.wait_offloads()
    out = out.to(q.dtype)
    return out, lse, provider.result(), base_seed, tuple(rng_states)
