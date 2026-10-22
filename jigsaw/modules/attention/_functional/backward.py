"""Backward implementations for ring attention."""

from typing import Optional

import torch
import torch.distributed as dist

from jigsaw.modules.attention._config import (
    RingAttentionConfig,
    SaveScope,
    SendBackward,
    SendOp,
)

from ._comm_overlap import consume, issue_on_comm_stream
from .backward_chunk_provider import _build_backward_chunk_provider
from .ring import (
    HybridTopology,
    _bhnd,
    _combine_dropout_seed,
    _seq_dim,
    get_hybrid_topology,
)

# Hybrid send-ops whose backward chunk provider also rotates chunks within the
# intra group, so a step's dK/dV belongs to local index ``local_idx + step``
# rather than ``step``. HYBRID_BROADCAST broadcasts within the intra group
# instead of rotating.
_INTRA_ROTATING_HYBRIDS = frozenset({SendOp.HYBRID_P2P, SendOp.HYBRID_ALLTOALL})


# --------------------------------------------------------------------------- #
# backward: which global chunk does step `step` hold?
#
# Shared by the dropout-mask seed lookup and `accumulate_dkv`'s dK/dV
# bucketing, which both need, for LOCAL save_scope, the global chunk the
# backward ring brought to this step.
# --------------------------------------------------------------------------- #
def _local_scope_global_chunk_idx(
    ring_config: RingAttentionConfig,
    local_rank: int,
    step: int,
    ring_group_size: int,
    group: dist.ProcessGroup,
) -> int:
    """Return the global chunk index held at backward ``step`` for ``SaveScope.LOCAL``."""
    match ring_config.send_op:
        case SendOp.HYBRID_P2P | SendOp.HYBRID_BROADCAST | SendOp.HYBRID_ALLTOALL:
            # The chunk sits on intra group `node_idx - step // gpn` (the outer
            # ring runs backwards relative to the forward). HYBRID_P2P and
            # HYBRID_ALLTOALL also rotate it within the intra group, so its
            # local index is `local_idx + step`. HYBRID_BROADCAST doesn't:
            # local rank `step % gpn` was the broadcast source.
            topo = get_hybrid_topology(group, ring_config.gpus_per_node)
            node_step = step // topo.gpus_per_node
            local = (
                topo.local_idx + step
                if ring_config.send_op in _INTRA_ROTATING_HYBRIDS
                else step
            )
            return topo.rank_at(topo.node_idx - node_step, local)
        case SendOp.BROADCAST:
            # `_LocalChunkProviderBroadcast.get(step)` broadcasts from rank
            # `step` (its *own* local chunk, i.e. global chunk `step`) to
            # everyone; the receiving rank's identity doesn't matter, mirroring
            # the forward `_BroadcastForwardChunkProvider.global_chunk_idx(step) == step`.
            return step
        case _:
            return (local_rank + step) % ring_group_size


def global_chunk_idx_backward(
    ring_config: RingAttentionConfig,
    local_rank: int,
    step: int,
    ring_group_size: int,
    group: dist.ProcessGroup,
) -> int:
    """Return the global chunk index processed at backward ``step``.

    Parameters
    ----------
    ring_config : RingAttentionConfig
        Ring-attention configuration.
    local_rank : int
        Rank within the ring.
    step : int
        Current ring step.
    ring_group_size : int
        Number of ranks in the ring.
    group : dist.ProcessGroup
        Process group of the ring.

    Returns
    -------
    int
        Global index of the key/value chunk.
    """
    if ring_config.save_scope == SaveScope.GLOBAL:
        return step
    return _local_scope_global_chunk_idx(
        ring_config, local_rank, step, ring_group_size, group
    )


# --------------------------------------------------------------------------- #
# backward: main loop
# --------------------------------------------------------------------------- #


def ring_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    out: torch.Tensor,
    x: torch.Tensor,
    lse: torch.Tensor,
    w_qkv: torch.Tensor,
    scale: float,
    group: dist.ProcessGroup,
    local_rank: int,
    ring_group_size: int,
    chunks_to_save: tuple[torch.Tensor, ...],
    ring_config: RingAttentionConfig,
    dropout_p: float = 0.0,
    base_seed: Optional[int] = None,
    rng_states: Optional[tuple[Optional[torch.Tensor], ...]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for ring attention.

    Mirrors :func:`~.forward.ring_forward`'s loop: the
    :class:`~.backward_chunk_provider._BackwardChunkProvider` for
    ``ring_config.send_op`` makes each chunk current (``get``) and progresses
    communication (``maybe_forward``); this function runs the attention core
    per step and accumulates dQ/dK/dV.

    Parameters
    ----------
    dout, q, out : torch.Tensor
        Output gradient, local query, and the forward's output, as returned by
        :func:`~.forward.ring_forward`.
    x : torch.Tensor
        Local input sequence chunk.
    lse : torch.Tensor
        The forward's log-sum-exp, as returned by :func:`~.forward.ring_forward`.
    w_qkv : torch.Tensor
        Fused query, key, and value projection weights.
    scale : float
        Attention scaling factor.
    group : dist.ProcessGroup
        Distributed process group used for ring communication.
    local_rank : int
        Rank within ``group``.
    ring_group_size : int
        Number of ranks in ``group``.
    chunks_to_save : tuple[torch.Tensor, ...]
        Chunks the forward saved for backward (see :func:`~.forward.ring_forward`).
    ring_config : RingAttentionConfig
        Ring-attention configuration.
    dropout_p : float
        Attention dropout probability.
    base_seed : int, optional
        Dropout base seed from the forward (``None`` if ``dropout_p == 0``).
    rng_states : tuple[torch.Tensor | None, ...], optional
        Per-global-chunk RNG state from the forward (see :func:`~.forward._attend_step`).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Local ``dq``, ``dk``, ``dv``.
    """
    layout = ring_config.core_layout
    b, h, seq_local, d = _bhnd(q.shape, layout)
    # Run the cores in ``q``'s dtype, the one ``attn_core_dtype()`` picked in
    # forward. Under autocast, ``out``, ``lse`` and the saved K/V chunks may be
    # bf16/fp16, and the cores matmul ``q`` against the chunks, so mixed dtypes
    # raise. ``w_qkv.dtype`` won't do: under ``master_weights`` it is the fp32
    # master copy, and the TE/cuDNN cores don't support fp32.
    core_dtype = q.dtype
    dout, q = dout.to(core_dtype), q.to(core_dtype)
    out = out.to(core_dtype)
    # ``lse`` is *not* folded into ``core_dtype``: it is a softmax normalizer,
    # never matmul'd against a chunk, and the fused SDPA/TE backward kernels
    # read it as fp32; a 16-bit ``lse`` makes them recompute the softmax
    # wrong. Each core casts it to what its kernel wants (the ``math`` core
    # keeps whatever precision the forward produced).
    dq = torch.zeros_like(q, dtype=core_dtype)
    dk_buf, dv_buf = get_dkv_buffer(ring_config, q, ring_group_size)

    # Cases where every rank processes global chunk `step` at ring step `step`:
    # the partial dK/dV for that chunk only needs a SUM all-reduce, kept by its
    # owner (`local_rank == step`). The result is already the final local
    # gradient, so it skips `postprocess_dkv` (whose GATHER_DKV branch would
    # slice a global buffer this path never builds).
    #   * BROADCAST + LOCAL: rank `step` broadcasts its chunk each step.
    #   * GLOBAL + SEND_DKV: `_GlobalChunkProvider` hands every rank global
    #     chunk `step`. GATHER_DKV keeps a global buffer and a final
    #     all-reduce instead.
    all_reduce_dkv = (
        ring_config.send_op == SendOp.BROADCAST
        and ring_config.save_scope == SaveScope.LOCAL
    ) or (
        ring_config.save_scope == SaveScope.GLOBAL
        and ring_config.send_backward == SendBackward.SEND_DKV
    )

    # Comm-stream overlap for the re-ringing path, gated in two parts:
    #   * `kv_overlap`: the K/V chunk provider. Every LOCAL-scope re-ring op
    #     has an overlap variant (see `_build_backward_chunk_provider`). It
    #     also sets the loop order below: with overlap, `maybe_forward` runs
    #     before each step's core.
    #   * `dkv_overlap`: the dK/dV relay (`_DkvRelay`), which also needs
    #     SEND_DKV. HYBRID_BROADCAST's accumulation (an intra-group all-reduce
    #     plus a conditional outer-ring send) doesn't fit the relay, so it stays
    #     synchronous even though its K/V provider overlaps.
    kv_overlap = (
        ring_config.overlap_active
        and ring_config.save_scope == SaveScope.LOCAL
        and ring_config.send_op
        in (
            SendOp.P2P,
            SendOp.ALLTOALL,
            SendOp.HYBRID_P2P,
            SendOp.HYBRID_ALLTOALL,
            SendOp.HYBRID_BROADCAST,
        )
    )
    dkv_overlap = (
        kv_overlap
        and not all_reduce_dkv
        and ring_config.send_op
        in (SendOp.P2P, SendOp.ALLTOALL, SendOp.HYBRID_P2P, SendOp.HYBRID_ALLTOALL)
        and ring_config.send_backward == SendBackward.SEND_DKV
    )
    dkv_relay = _DkvRelay(ring_config, group, dq) if dkv_overlap else None

    chunk_provider = _build_backward_chunk_provider(
        ring_config,
        chunks_to_save,
        x,
        w_qkv,
        b,
        seq_local,
        h,
        d,
        group,
        local_rank,
    )

    for step in range(ring_group_size):
        kj, vj, kv = chunk_provider.get(step)
        if kv_overlap:
            # Prefetch the next chunk's K/V now so its transfer runs under
            # this step's core instead of after it.
            chunk_provider.maybe_forward(step, ring_group_size)
        global_chunk_idx = global_chunk_idx_backward(
            ring_config, local_rank, step, ring_group_size, group
        )
        seed = (
            _combine_dropout_seed(base_seed, local_rank, global_chunk_idx)
            if base_seed is not None
            else None
        )
        rng_state = rng_states[global_chunk_idx] if rng_states is not None else None
        dq_j, dk_j, dv_j = ring_config.attn_core_backward_fn(
            dout,
            q,
            kj.to(core_dtype),
            vj.to(core_dtype),
            out,
            lse,
            scale,
            seq_local,
            seq_local,
            dropout_p=dropout_p,
            seed=seed,
            rng_state=rng_state,
            layout=layout,
        )
        dq += dq_j
        del dq_j, kj, vj, kv

        if all_reduce_dkv:
            dist.all_reduce(dk_j, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(dv_j, op=dist.ReduceOp.SUM, group=group)
            if local_rank == step:
                dk_buf = dk_j
                dv_buf = dv_j
        elif dkv_overlap:
            assert dkv_relay is not None
            dkv_relay.step(dk_buf, dv_buf, dk_j, dv_j, step)
        else:
            dk_buf, dv_buf = accumulate_dkv(
                ring_config,
                dk_buf,
                dv_buf,
                dk_j,
                dv_j,
                step,
                ring_group_size,
                seq_local,
                local_rank,
                group,
            )

        if not kv_overlap:
            # Overlap issues this before the core; otherwise relay here.
            # No-op for providers that don't re-circulate chunks (e.g. global).
            chunk_provider.maybe_forward(step, ring_group_size)

    if dkv_overlap:
        assert dkv_relay is not None
        dkv_relay.drain(dk_buf, dv_buf)

    if all_reduce_dkv:
        # `dk_buf`/`dv_buf` already hold this rank's final local gradient.
        return dq, dk_buf, dv_buf

    dk_local, dv_local = postprocess_dkv(
        ring_config, dk_buf, dv_buf, local_rank, seq_local, group
    )

    return dq, dk_local, dv_local


# --------------------------------------------------------------------------- #
# backward: dK/dV accumulation across the ring
# --------------------------------------------------------------------------- #
def _use_global_dkv_buffer(ring_config: RingAttentionConfig) -> bool:
    """Return whether dK and dV accumulate in a global-sized buffer."""
    return ring_config.send_backward == SendBackward.GATHER_DKV


def get_dkv_buffer(
    ring_config: RingAttentionConfig, q: torch.Tensor, ring_group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the dK and dV buffers of the accumulation strategy.

    Parameters
    ----------
    ring_config : RingAttentionConfig
        Ring-attention configuration.
    q : torch.Tensor
        Local query, which sets the shape, device and dtype of the buffers.
    ring_group_size : int
        Number of ranks in the ring.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Zero-initialized dk and dv buffers. They span the full sequence if the
        strategy gathers the gradients and the local sequence otherwise.
    """
    # Two tensors, not views of one: an in-place add of a grad-tracked partial
    # makes the written tensor require grad, which would break in-place writes
    # to a sibling view ("a view of a leaf Variable that requires grad").
    shape = list(q.shape)
    if _use_global_dkv_buffer(ring_config):
        # Global buffer spans every rank's chunk along the sequence axis.
        sd = _seq_dim(ring_config.core_layout)
        shape[sd] = ring_group_size * shape[sd]
    return (
        torch.zeros(shape, device=q.device, dtype=q.dtype),
        torch.zeros(shape, device=q.device, dtype=q.dtype),
    )


def accumulate_dkv(
    ring_config: RingAttentionConfig,
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    dk_j: torch.Tensor,
    dv_j: torch.Tensor,
    step: int,
    ring_group_size: int,
    seq_local: int,
    local_rank: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate the dK and dV of one step into the buffers.

    Parameters
    ----------
    ring_config : RingAttentionConfig
        Ring-attention configuration.
    dk_buf : torch.Tensor
        Buffer accumulating dk.
    dv_buf : torch.Tensor
        Buffer accumulating dv.
    dk_j, dv_j : torch.Tensor
        Gradients of the current step.
    step : int
        Current ring step.
    ring_group_size : int
        Number of ranks in the ring.
    seq_local : int
        Local sequence length.
    local_rank : int
        Rank within the ring.
    group : dist.ProcessGroup
        Process group of the ring.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Updated dk and dv buffers.
    """
    if not _use_global_dkv_buffer(ring_config):
        if (
            ring_config.send_op == SendOp.HYBRID_BROADCAST
            and ring_config.save_scope == SaveScope.LOCAL
        ):
            return _accumulate_dkv_hybrid_broadcast(
                ring_config, dk_buf, dv_buf, dk_j, dv_j, step, group
            )
        return _accumulate_dkv_send(
            ring_config, dk_buf, dv_buf, dk_j, dv_j, step, group
        )

    global_chunk_idx = global_chunk_idx_backward(
        ring_config, local_rank, step, ring_group_size, group
    )
    return _accumulate_dkv_gather(
        dk_buf,
        dv_buf,
        dk_j,
        dv_j,
        seq_local,
        global_chunk_idx,
        _seq_dim(ring_config.core_layout),
    )


def _accumulate_dkv_gather(
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    dk_j: torch.Tensor,
    dv_j: torch.Tensor,
    seq_local: int,
    global_chunk_idx: int,
    sd: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the dk and dv of one step to their slot in the global buffers.

    Parameters
    ----------
    dk_buf : torch.Tensor
        Buffer accumulating dk.
    dv_buf : torch.Tensor
        Buffer accumulating dv.
    dk_j, dv_j : torch.Tensor
        Gradients of the current step.
    seq_local : int
        Local sequence length.
    global_chunk_idx : int
        Index of the sequence chunk represented by the local gradients.
    sd : int
        Sequence axis of ``dk_buf``/``dv_buf``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Updated dk and dv buffers.
    """
    lo = global_chunk_idx * seq_local
    dk_buf.narrow(sd, lo, seq_local).add_(dk_j)
    dv_buf.narrow(sd, lo, seq_local).add_(dv_j)
    return dk_buf, dv_buf


def _hybrid_dkv_relay_peers(topo: HybridTopology, step: int) -> tuple[int, int]:
    """(send_to, recv_from) for relaying a dK/dV partial to its home rank.

    At backward ``step`` this rank holds the gradient for the chunk owned by
    ``rank_at(node_idx - step // gpn, local_idx + step)``. Hand it straight to
    that owner, and receive (from the mirror-image rank) the partial destined
    for this rank's own chunk.

    Unlike the K/V sends, this one changes both the intra group and the local
    index at once, so neither sub-group can address it and it uses ranks of
    the full ring group.
    """
    node_step = step // topo.gpus_per_node
    send_to = topo.rank_at(topo.node_idx - node_step, topo.local_idx + step)
    recv_from = topo.rank_at(topo.node_idx + node_step, topo.local_idx - step)
    return send_to, recv_from


def _accumulate_dkv_send(
    ring_config: RingAttentionConfig,
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    dk_j: torch.Tensor,
    dv_j: torch.Tensor,
    step: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relay dk and dv to the rank owning the chunk.

    Parameters
    ----------
    ring_config : RingAttentionConfig
        Ring-attention configuration.
    dk_buf : torch.Tensor
        Buffer accumulating dk.
    dv_buf : torch.Tensor
        Buffer accumulating dv.
    dk_j, dv_j : torch.Tensor
        Gradients of the current step.
    step : int
        Current ring step.
    group : dist.ProcessGroup
        Process group of the ring.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Updated dk and dv buffers.
    """
    if step == 0:
        dk_buf += dk_j
        dv_buf += dv_j
        return dk_buf, dv_buf

    dkv = torch.cat([dk_j, dv_j], dim=0).contiguous()
    recv_buf = torch.empty_like(dkv)

    # Hybrid rings relay the partial straight to its owner, which is addressed
    # by full-ring-group rank; the other send ops shift it by ``step``.
    if ring_config.send_op in _INTRA_ROTATING_HYBRIDS:
        topo = get_hybrid_topology(group, ring_config.gpus_per_node)
        send_recv_ranks = _hybrid_dkv_relay_peers(topo, step)
        recv_buf, _ = ring_config.p2p_fn(
            dkv,
            recv_buf,
            send_recv_ranks=send_recv_ranks,
            group=group,
            async_op=False,
            step=step,
        )
    else:
        recv_buf, _ = ring_config.p2p_fn(
            dkv,
            recv_buf,
            step if ring_config.send_op != SendOp.BROADCAST else -step,
            group=group,
            async_op=False,
            step=step,
        )
    dkj_r, dvj_r = torch.chunk(recv_buf, 2, dim=0)
    dk_buf += dkj_r
    dv_buf += dvj_r
    return dk_buf, dv_buf


class _DkvRelay:
    """Async dK/dV relay on the comm stream for the SEND_DKV ring backward.

    Overlapped version of :func:`_accumulate_dkv_send`. Each step's partial is
    staged into a persistent buffer, sent on ``ring_config.comm_stream`` with
    ``async_op=True``, and added into ``dk_buf`` / ``dv_buf`` one step later,
    so the transfer runs under the next step's core. Two send/receive buffer
    pairs alternate.

    Step 0 stays local, as in :func:`_accumulate_dkv_send`. P2P/all-to-all send
    the partial ``step`` ranks along the ring; ``HYBRID_P2P`` /
    ``HYBRID_ALLTOALL`` use :func:`_hybrid_dkv_relay_peers`, which moves along
    the outer ring and within the intra group at once. ``BROADCAST`` and
    ``HYBRID_BROADCAST`` never reach this class (see ``all_reduce_dkv`` and
    ``dkv_overlap`` in :func:`ring_backward`).
    """

    def __init__(
        self,
        ring_config: RingAttentionConfig,
        group: dist.ProcessGroup,
        like: torch.Tensor,
    ) -> None:
        """Allocate the staging buffers from ``like`` (a per-rank dK/dV tensor)."""
        self.cfg = ring_config
        self.group = group
        self.cs = ring_config.comm_stream
        self._topo = (
            get_hybrid_topology(group, ring_config.gpus_per_node)
            if ring_config.send_op in _INTRA_ROTATING_HYBRIDS
            else None
        )
        shape = (2 * like.shape[0], *like.shape[1:])  # cat(dk, dv) on dim 0
        self._send = [
            torch.empty(shape, device=like.device, dtype=like.dtype) for _ in range(2)
        ]
        self._recv = [
            torch.empty(shape, device=like.device, dtype=like.dtype) for _ in range(2)
        ]
        self._pending: Optional[tuple[torch.cuda.Event, torch.Tensor]] = None
        self._work: object = None

    def _drain(self, dk_buf: torch.Tensor, dv_buf: torch.Tensor) -> None:
        """Fold the in-flight relay's result (if any) into ``dk_buf`` / ``dv_buf``."""
        if self._pending is None:
            return
        event, recv = self._pending
        consume(event, recv)
        dk_r, dv_r = torch.chunk(recv, 2, dim=0)
        dk_buf.add_(dk_r)
        dv_buf.add_(dv_r)
        self._pending = None

    def step(
        self,
        dk_buf: torch.Tensor,
        dv_buf: torch.Tensor,
        dk_j: torch.Tensor,
        dv_j: torch.Tensor,
        step: int,
    ) -> None:
        """Relay this step's dK/dV partial; fold in the previous step's."""
        if step == 0:
            dk_buf.add_(dk_j)
            dv_buf.add_(dv_j)
            return

        i = step & 1
        send, recv = self._send[i], self._recv[i]
        b = dk_j.shape[0]
        send[:b].copy_(dk_j)
        send[b:].copy_(dv_j)
        # Same routing as `_accumulate_dkv_send`: HYBRID_P2P/HYBRID_ALLTOALL
        # hand the partial to `_hybrid_dkv_relay_peers` (moves along both the
        # outer ring and the intra group); everything else sends `step` ranks along
        # the ring toward the chunk's home rank.
        if self._topo is not None:
            send_recv_ranks = _hybrid_dkv_relay_peers(self._topo, step)
            produce = lambda: self.cfg.p2p_fn(  # noqa: E731
                send,
                recv,
                send_recv_ranks=send_recv_ranks,
                group=self.group,
                async_op=True,
                step=step,
            )
        else:
            direction = step if self.cfg.send_op != SendOp.BROADCAST else -step
            produce = lambda: self.cfg.p2p_fn(  # noqa: E731
                send, recv, direction, group=self.group, async_op=True, step=step
            )
        assert self.cs is not None
        _, self._work, event = issue_on_comm_stream(
            self.cs, produce, keep_alive=(send, recv)
        )
        self._drain(dk_buf, dv_buf)
        self._pending = (event, recv)

    def drain(self, dk_buf: torch.Tensor, dv_buf: torch.Tensor) -> None:
        """Fold in the last outstanding relay. Call once after the ring loop."""
        self._drain(dk_buf, dv_buf)


def _accumulate_dkv_hybrid_broadcast(
    ring_config: RingAttentionConfig,
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    dk_j: torch.Tensor,
    dv_j: torch.Tensor,
    step: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate dK and dV of a hybrid broadcast step into local-sized buffers."""
    topo = get_hybrid_topology(group, ring_config.gpus_per_node)
    node_step, phase = divmod(step, topo.gpus_per_node)
    is_owner = topo.local_idx == phase

    dkv = torch.cat([dk_j, dv_j], dim=0).contiguous()
    dist.all_reduce(dkv, op=dist.ReduceOp.SUM, group=topo.intranode_group)

    if node_step != 0:
        # Outer-ring exchange as a permutation of `internode_group`: every rank
        # sends its intra-group sum to its counterpart in intra group
        # ``node_idx - node_step`` and receives from ``node_idx + node_step``.
        # Only local rank ``phase`` gets the partial for a chunk it owns; the
        # rest take part to keep the collective whole and discard what they
        # receive.
        recv = torch.empty_like(dkv)
        recv, _ = ring_config.p2p_fn(
            dkv,
            recv,
            send_recv_ranks=(
                topo.internode_peer(-node_step),
                topo.internode_peer(node_step),
            ),
            group=topo.internode_group,
            async_op=False,
            step=step,
        )
        dkv = recv

    if is_owner:
        dkj_r, dvj_r = torch.chunk(dkv, 2, dim=0)
        dk_buf += dkj_r
        dv_buf += dvj_r
    return dk_buf, dv_buf


def postprocess_dkv(
    ring_config: RingAttentionConfig,
    dk_buf: torch.Tensor,
    dv_buf: torch.Tensor,
    local_rank: int,
    seq_local: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce the global dk and dv buffers and select the local slice.

    Parameters
    ----------
    ring_config : RingAttentionConfig
        Ring-attention configuration.
    dk_buf : torch.Tensor
        Buffer accumulating dk.
    dv_buf : torch.Tensor
        Buffer accumulating dv.
    local_rank : int
        Rank within the ring group.
    seq_local : int
        Local sequence length.
    group : dist.ProcessGroup
        Process group of the ring.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Local dk and dv.
    """
    if _use_global_dkv_buffer(ring_config):
        # One stacked buffer for a single all-reduce; safe now that the
        # in-place accumulation is done (see `get_dkv_buffer`).
        dkv_buf = torch.stack((dk_buf, dv_buf), dim=0)
        dist.all_reduce(dkv_buf, op=dist.ReduceOp.SUM, group=group)
        dk_buf, dv_buf = dkv_buf[0], dkv_buf[1]
        sd = _seq_dim(ring_config.core_layout)
        lo = local_rank * seq_local
        return (
            dk_buf.narrow(sd, lo, seq_local),
            dv_buf.narrow(sd, lo, seq_local),
        )
    return dk_buf, dv_buf
