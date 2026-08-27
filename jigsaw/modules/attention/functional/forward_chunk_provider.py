"""Chunk providers for the ring-attention forward pass.

The forward twin of :mod:`chunk_provider`. Where the backward pass drives a
single loop over :class:`_BackwardChunkProvider`, the forward pass drives a
single loop over :class:`_ForwardChunkProvider`: every communication strategy
(plain P2P/all-to-all ring, broadcast ring, and their hybrid intra-/inter-node
variants) is expressed as one of these classes so :func:`.forward.ring_forward`
never branches on :class:`~jigsaw.modules.attention.config.SendOp` itself.

Each provider exposes the same three hooks, mirroring the backward providers:

``get(step)``
    Make the chunk for ring ``step`` current (waiting on / kicking off whatever
    communication that needs) and return its contiguous ``(kj, vj)``.
``save_for_backward(step)``
    Record the chunk currently held for the backward pass. Only does work for
    ``SaveScope.GLOBAL``; ``LOCAL`` scope saves the single local chunk up front
    in ``__init__``.
``advance(step, world_size)``
    Kick off / complete the communication that makes ``step + 1`` available.
    A no-op for strategies that fold everything into ``get``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.distributed as dist

from jigsaw.modules.attention.config import (
    RingAttentionConfig,
    SaveScope,
    SendForward,
    SendOp,
)

from .functional import _compute_kv, get_hybrid_topology


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
class _ForwardChunkProvider(ABC):
    """Yields ``(kj, vj)`` for each forward ring step and collects ``chunks_to_save``."""

    def __init__(
        self,
        ring_config: RingAttentionConfig,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        group: dist.ProcessGroup,
        empty: torch.Tensor,
        local_rank: int,
        world_size: int,
    ) -> None:
        """Initialize the provider and its running chunk state.

        Parameters
        ----------
        ring_config : RingAttentionConfig
            Ring-attention configuration controlling communication and saving.
        q, k, v : torch.Tensor
            Local query, key, and value tensors, shaped ``(b, h, n_local, d)``.
        x : torch.Tensor
            Local input sequence chunk.
        w_qkv : torch.Tensor
            Fused query, key, and value projection weights.
        group : dist.ProcessGroup
            Process group used for ring communication.
        empty : torch.Tensor
            Empty tensor used by the communication helpers.
        local_rank : int
            Rank of the current process within ``group``.
        world_size : int
            Number of ranks in ``group`` (== number of ring steps).
        """
        self.config = ring_config
        self.q, self.k, self.v, self.x, self.w_qkv = q, k, v, x, w_qkv
        self.group = group
        self.empty = empty
        self.local_rank = local_rank
        self.world_size = world_size
        self.b, self.h, self.n_local, self.d = q.shape
        self.send_x = ring_config.send_in_forward == SendForward.X

        # Local KV, kept private so an in-flight send can never alias k / v.
        self.kv = torch.concat([k.unsqueeze(0), v.unsqueeze(0)]).clone().contiguous()

        # Running "chunk currently in hand" state.
        self.x_j = x
        self.kv_j, self.kj, self.vj = self.kv, k, v

        self.chunks_to_save: list[Optional[torch.Tensor]]
        if ring_config.save_scope == SaveScope.GLOBAL:
            # Filled per-step, indexed by global chunk id so the backward
            # `_GlobalChunkProvider` (a plain `chunks[step]` lookup) sees
            # ring step == global chunk index.
            self.chunks_to_save = [None] * world_size
        else:
            # LOCAL: only this rank's own chunk is kept; the backward re-rings
            # the rest.
            self.chunks_to_save = [ring_config.save_for_backward_fn(x, self.kv)]

    # -- helpers --------------------------------------------------------- #
    def _kv_from_x(self) -> None:
        """Recompute ``kv_j / kj / vj`` from the chunk currently in ``x_j``."""
        self.kv_j, self.kj, self.vj = _compute_kv(
            self.x_j, self.w_qkv, self.b, self.n_local, self.h, self.d
        )

    def _recv(self, recv_buf: torch.Tensor) -> None:
        """Fold ``recv_buf`` into the running chunk state via ``forward_recv_fn``."""
        self.x_j, self.kv_j, self.kj, self.vj = self.config.forward_recv_fn(
            recv_buf, self.x_j, self.kv_j, self.kj, self.vj
        )

    # -- interface ----------------------------------------------------- #
    @abstractmethod
    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Make chunk ``step`` current and return its ``(kj, vj)``."""

    def _global_chunk_idx(self, step: int) -> int:
        """Global chunk id of the chunk held at ring ``step`` (GLOBAL save only)."""
        raise NotImplementedError

    def save_for_backward(self, step: int) -> None:
        """Record the chunk currently in hand for the backward pass."""
        if self.config.save_scope != SaveScope.GLOBAL:
            return
        self.chunks_to_save[self._global_chunk_idx(step)] = (
            self.config.save_for_backward_fn(self.x_j, self.kv_j)
        )

    def advance(self, step: int, world_size: int) -> None:
        """Progress communication so ``step + 1`` is available. Default: no-op."""
        return

    def result(self) -> tuple[Optional[torch.Tensor], ...]:
        """Return the collected ``chunks_to_save`` tuple."""
        return tuple(self.chunks_to_save)


# --------------------------------------------------------------------------- #
# plain P2P / all-to-all ring
# --------------------------------------------------------------------------- #
class _RingForwardChunkProvider(_ForwardChunkProvider):
    """One P2P (or all-to-all) send/recv per step around a ``direction=-1`` ring.

    The forward twin of :class:`~.chunk_provider._LocalChunkProvider`.
    """

    def __init__(self, *args: object) -> None:
        """Kick off the first ring send of the local chunk."""
        super().__init__(*args)  # type: ignore[arg-type]
        self._recv_buf, self._handle = self.config.forward_send_fn(
            self.x_j, self.kv_j, direction=-1, group=self.group, tag=0, empty=self.empty
        )

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Receive chunk ``step``, relay it onward, and return its ``(kj, vj)``.

        Parameters
        ----------
        step : int
            Current ring step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Contiguous key and value tensors for chunk ``step``.
        """
        if step > 0:
            self._handle.wait()
            self._recv(self._recv_buf)
            if step + 1 < self.world_size:
                self._recv_buf, self._handle = self.config.forward_send_fn(
                    self.x_j,
                    self.kv_j,
                    direction=-1,
                    group=self.group,
                    tag=step,
                    empty=self.empty,
                )
            if self.send_x:
                self._kv_from_x()
        return self.kj, self.vj

    def _global_chunk_idx(self, step: int) -> int:
        """``direction=-1`` ring: after ``step`` receives, ``x_j`` came from here."""
        return (self.local_rank + step) % self.world_size


# --------------------------------------------------------------------------- #
# broadcast ring
# --------------------------------------------------------------------------- #
class _BroadcastForwardChunkProvider(_ForwardChunkProvider):
    """Per-step broadcast: at step ``s`` rank ``s`` fans its chunk out to the group.

    The forward twin of
    :class:`~.chunk_provider._LocalChunkProviderBroadcast`.
    """

    def __init__(self, *args: object) -> None:
        """Initialize the broadcast bookkeeping."""
        super().__init__(*args)  # type: ignore[arg-type]
        self._recv_buf: torch.Tensor = self.empty
        self._handle: Optional[dist.Work] = None

    def _kick(self, src_rank: int) -> None:
        """Start the broadcast rooted at ``src_rank`` of that rank's chunk."""
        self._recv_buf, self._handle = self.config.forward_send_fn(
            x=self.x,
            kv=self.kv,
            local_rank=self.local_rank,
            group=self.group,
            src_rank=src_rank % self.world_size,
            empty=self.empty,
        )

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Complete the broadcast of chunk ``step`` and prefetch chunk ``step + 1``.

        Parameters
        ----------
        step : int
            Current ring step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Contiguous key and value tensors for chunk ``step``.
        """
        if step == 0:
            self._kick(0)
        assert self._handle is not None
        self._handle.wait()
        self._recv(self._recv_buf)
        if step + 1 < self.world_size:
            self._kick(step + 1)

        if self.send_x:
            if step == self.local_rank:
                # Our own chunk: reuse local K/V instead of recomputing.
                self.kv_j, self.kj, self.vj = self.kv, self.k, self.v
            else:
                self._kv_from_x()
        return self.kj, self.vj

    def _global_chunk_idx(self, step: int) -> int:
        """Step ``step`` broadcasts global chunk ``step`` to every rank."""
        return step


# --------------------------------------------------------------------------- #
# hybrid: intra-node ring + inter-node P2P
# --------------------------------------------------------------------------- #
class _HybridP2PForwardChunkProvider(_ForwardChunkProvider):
    """``gpus_per_node`` intra-node ring steps per node round, one inter-node hop each.

    Serves both ``SendOp.HYBRID_P2P`` and ``SendOp.HYBRID_ALLTOALL`` -- the ring
    logic is identical, only the underlying comm primitive differs (the
    ``ring_config`` picks ``comm_1d_p2p`` vs ``comm_1d_alltoall`` for the
    inter-node hop; see :meth:`RingAttentionConfig._p2p_fn`).

    The forward twin of
    :class:`~.chunk_provider._LocalChunkProviderHybridP2P`. Ring step ``s`` maps
    to ``divmod(s, gpus_per_node) == (node_step, inner)``; the inter-node send of
    this rank's original chunk is launched at ``inner == 0`` and consumed at the
    start of the next round.
    """

    def __init__(self, *args: object) -> None:
        """Resolve the hybrid topology and its peer lists."""
        super().__init__(*args)  # type: ignore[arg-type]
        self.topo = get_hybrid_topology(self.group, self.config.gpus_per_node)
        self.gpn = self.topo.gpus_per_node
        self.n_nodes = self.topo.n_nodes
        # Intra-node ring: send to the previous local rank, receive from the next.
        # Inter-node hop: send to the previous node, receive from the next.
        self._intra_ranks = [self.topo.intra_peer(-1), self.topo.intra_peer(1)]
        self._inter_ranks = [self.topo.inter_peer(-1), self.topo.inter_peer(1)]
        self._intra_recv_buf: Optional[torch.Tensor] = None
        self._intra_handle: Optional[dist.Work] = None
        self._inter_recv_buf: Optional[torch.Tensor] = None
        self._inter_handle: Optional[dist.Work] = None

    def _launch_inter(self, node_step: int) -> None:
        """Start this round's inter-node send of the original local chunk."""
        buf = torch.empty_like(self.x_j if self.send_x else self.kv_j)
        self._inter_recv_buf, self._inter_handle = self.config.p2p_fn(
            data=(self.x.clone() if self.send_x else self.kv.clone()),
            buffer=buf,
            send_recv_ranks=self._inter_ranks,
            group=self.group,
            async_op=True,
            tag=node_step,
            empty=self.empty,
        )

    def _launch_intra(self, inner: int) -> None:
        """Start the intra-node ring send of the chunk currently in hand."""
        src = self.x_j if self.send_x else self.kv_j
        self._intra_recv_buf, self._intra_handle = self.config.send_op_fn(
            src,
            torch.empty_like(src),
            send_recv_ranks=self._intra_ranks,
            group=self.group,
            tag=inner,
        )

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Make chunk ``step`` current within the hybrid ring.

        Parameters
        ----------
        step : int
            Current ring step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Contiguous key and value tensors for chunk ``step``.
        """
        node_step, inner = divmod(step, self.gpn)
        if inner == 0:
            if node_step > 0:
                assert self._inter_handle is not None
                assert self._inter_recv_buf is not None
                self._inter_handle.wait()
                self._recv(self._inter_recv_buf)
            if node_step < self.n_nodes - 1:
                self._launch_inter(node_step)
        if inner + 1 < self.gpn:
            self._launch_intra(inner)

        if self.send_x:
            self._kv_from_x()
        else:
            self.kj, self.vj = self.kv_j[0], self.kv_j[1]
        return self.kj, self.vj

    def advance(self, step: int, world_size: int) -> None:
        """Complete the intra-node ring send so the next inner step is ready.

        Parameters
        ----------
        step : int
            Current ring step.
        world_size : int
            Number of ranks in the ring.
        """
        _, inner = divmod(step, self.gpn)
        if inner + 1 < self.gpn:
            assert self._intra_handle is not None
            assert self._intra_recv_buf is not None
            self._intra_handle.wait()
            self._recv(self._intra_recv_buf)

    def _global_chunk_idx(self, step: int) -> int:
        """Chunk at ``step`` is owned by ``(node_idx + node_step, local_idx + inner)``."""
        node_step, inner = divmod(step, self.gpn)
        cur_node = (self.topo.node_idx + node_step) % self.n_nodes
        return cur_node * self.gpn + (self.topo.local_idx + inner) % self.gpn


# --------------------------------------------------------------------------- #
# hybrid: intra-node broadcast + inter-node P2P
# --------------------------------------------------------------------------- #
class _HybridBroadcastForwardChunkProvider(_ForwardChunkProvider):
    """Per-round sweep of ``gpus_per_node`` intra-node broadcasts, one inter-node hop.

    The forward twin of
    :class:`~.chunk_provider._LocalChunkProviderHybridBroadcast`. At inner step
    ``s`` the node-local rank ``s`` broadcasts its round chunk to the node; the
    round chunk is stable for the round, so there is no intra-node rotation.
    """

    def __init__(self, *args: object) -> None:
        """Resolve the hybrid topology and its inter-node peer list."""
        super().__init__(*args)  # type: ignore[arg-type]
        self.topo = get_hybrid_topology(self.group, self.config.gpus_per_node)
        self.gpn = self.topo.gpus_per_node
        self.n_nodes = self.topo.n_nodes
        self._intranode_group = self.topo.intranode_group
        # Inter-node hop mirrors _HybridP2PForwardChunkProvider.
        self._inter_ranks = [self.topo.inter_peer(-1), self.topo.inter_peer(1)]
        self._inter_recv_buf: Optional[torch.Tensor] = None
        self._inter_handle: Optional[dist.Work] = None
        # The chunk this rank contributes to the current node round.
        self._own_kv: Optional[torch.Tensor] = None
        self._own_buf: Optional[torch.Tensor] = None

    def _launch_inter(self, node_step: int) -> None:
        """Start this round's inter-node send of the original local chunk."""
        buf = torch.empty_like(self.x_j if self.send_x else self.kv_j)
        self._inter_recv_buf, self._inter_handle = self.config.p2p_fn(
            data=(self.x.clone() if self.send_x else self.kv.clone()),
            buffer=buf,
            send_recv_ranks=self._inter_ranks,
            group=self.group,
            async_op=True,
            tag=node_step,
            empty=self.empty,
        )

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Broadcast the chunk for inner step ``inner`` and return its ``(kj, vj)``.

        Parameters
        ----------
        step : int
            Current ring step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Contiguous key and value tensors for the current chunk.
        """
        node_step, inner = divmod(step, self.gpn)
        if inner == 0:
            if node_step > 0:
                assert self._inter_handle is not None
                assert self._inter_recv_buf is not None
                self._inter_handle.wait()
                self._recv(self._inter_recv_buf)
                if self.send_x:
                    self._kv_from_x()
            self._own_kv = self.kv_j
            self._own_buf = self.x_j if self.send_x else self.kv_j
            if node_step < self.n_nodes - 1:
                self._launch_inter(node_step)

        assert self._own_buf is not None and self._own_kv is not None
        src = self.topo.rank_at(self.topo.node_idx, inner)
        is_src = self.topo.local_idx == inner
        recv_buf = self._own_buf if is_src else torch.empty_like(self._own_buf)
        dist.broadcast(recv_buf, src=src, group=self._intranode_group)

        if self.send_x:
            self.x_j = recv_buf
            if is_src:
                self.kv_j, self.kj, self.vj = (
                    self._own_kv,
                    self._own_kv[0],
                    self._own_kv[1],
                )
            else:
                self._kv_from_x()
        else:
            self.kv_j, self.kj, self.vj = recv_buf, recv_buf[0], recv_buf[1]
        return self.kj, self.vj

    def _global_chunk_idx(self, step: int) -> int:
        """Inner step ``inner`` broadcasts global chunk ``cur_node * gpn + inner``."""
        node_step, inner = divmod(step, self.gpn)
        cur_node = (self.topo.node_idx + node_step) % self.n_nodes
        return cur_node * self.gpn + inner


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def build_forward_chunk_provider(
    ring_config: RingAttentionConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    group: dist.ProcessGroup,
    empty: torch.Tensor,
    local_rank: int,
    world_size: int,
) -> _ForwardChunkProvider:
    """Pick the forward chunk provider that implements ``ring_config.send_op``.

    Parameters
    ----------
    ring_config : RingAttentionConfig
        Ring-attention configuration.
    q, k, v : torch.Tensor
        Local query, key, and value tensors.
    x : torch.Tensor
        Local input sequence chunk.
    w_qkv : torch.Tensor
        Fused query, key, and value projection weights.
    group : dist.ProcessGroup
        Process group used for ring communication.
    empty : torch.Tensor
        Empty tensor used by the communication helpers.
    local_rank : int
        Rank of the current process within ``group``.
    world_size : int
        Number of ranks in ``group``.

    Returns
    -------
    _ForwardChunkProvider
        The provider for the configured communication strategy.
    """
    args = (ring_config, q, k, v, x, w_qkv, group, empty, local_rank, world_size)
    match ring_config.send_op:
        case SendOp.BROADCAST:
            return _BroadcastForwardChunkProvider(*args)
        case SendOp.P2P | SendOp.ALLTOALL:
            return _RingForwardChunkProvider(*args)
        case SendOp.HYBRID_P2P | SendOp.HYBRID_ALLTOALL:
            return _HybridP2PForwardChunkProvider(*args)
        case SendOp.HYBRID_BROADCAST:
            return _HybridBroadcastForwardChunkProvider(*args)
        case _:
            raise NotImplementedError(f"Unknown send_op: {ring_config.send_op}")
