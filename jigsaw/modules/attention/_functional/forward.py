"""Ring attention forward pass."""

from __future__ import annotations

import torch
import torch.distributed as dist

from jigsaw.modules.attention._config import RingAttentionConfig

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

    Computes ``o1 <- (1 - w) * o1 + w * o2`` with ``w = sigmoid(l2 - l1)`` and
    ``l_new = logaddexp(l1, l2)``. Uses the fused Triton kernel
    (:func:`._merge_kernel.fused_online_softmax_merge_`) when available. ``l1`` is
    returned fresh, since ``l2`` may carry autograd history.

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
    seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Attend to one chunk and merge it into the running output.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        Merged output and log-sum-exp, and the ``rng_state`` of the core.
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
    int | None,
    tuple[torch.Tensor | None, ...],
]:
    """Run the forward pass of ring attention.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Local query, key, and value tensors in ``ring_config.core_layout``.
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

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...], int | None, tuple[torch.Tensor | None, ...]]
        Output, log-sum-exp, chunks saved for backward, dropout base seed
        (``None`` without dropout) and per-chunk RNG states.
    """
    # Build the provider first: the plain ring starts its first send here,
    # overlapping it with the setup below.
    provider = build_forward_chunk_provider(
        ring_config, q, k, v, x, w_qkv, group, local_rank, ring_group_size
    )

    # Accumulate in >= fp32; the fused backward kernels need an fp32 ``lse``.
    acc_dtype = torch.promote_types(q.dtype, torch.float32)
    b, heads, n_local, _ = _bhnd(q.shape, ring_config.core_layout)
    out = torch.zeros(q.shape, device=q.device, dtype=acc_dtype)
    # ``lse`` is ``(b, h, n)`` regardless of layout.
    lse = torch.full(
        (b, heads, n_local), float("-inf"), device=q.device, dtype=acc_dtype
    )

    base_seed = _draw_base_seed(q.device) if dropout_p > 0.0 else None
    rng_states: list[torch.Tensor | None] = [None] * ring_group_size

    for step in range(ring_group_size):
        kj, vj = provider.get(step)
        provider.save_for_backward(step)
        global_chunk_idx = provider.global_chunk_idx(step)
        seed = (
            _combine_dropout_seed(base_seed, local_rank, global_chunk_idx)
            if base_seed is not None
            else None
        )
        # Chunks rebuilt from ``x`` are in ``w_qkv``'s dtype; match ``q``'s.
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
        )
        rng_states[global_chunk_idx] = rng_state_j
        # Drain this step's D2H offload behind its attention core
        ring_config.wait_offloads()
        provider.advance(step, ring_group_size)

    ring_config.wait_offloads()
    out = out.to(q.dtype)
    return out, lse, provider.result(), base_seed, tuple(rng_states)
