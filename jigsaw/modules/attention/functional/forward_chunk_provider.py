"""Chunk providers for the ring-attention forward pass.

Each provider implements one communication strategy through the hooks ``get``,
``save_for_backward`` and ``advance``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.distributed as dist

from jigsaw.comm import get_empty
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
    """Base class yielding the key and value chunk of each forward ring step."""

    def __init__(
        self,
        ring_config: RingAttentionConfig,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        group: dist.ProcessGroup,
        local_rank: int,
        world_size: int,
    ) -> None:
        """Initialize the provider.

        Parameters
        ----------
        ring_config : RingAttentionConfig
            Ring-attention configuration.
        q, k, v : torch.Tensor
            Local query, key, and value tensors, shaped ``(b, h, n_local, d)``.
        x : torch.Tensor
            Local input sequence chunk.
        w_qkv : torch.Tensor
            Fused query, key, and value projection weights.
        group : dist.ProcessGroup
            Process group of the ring.
        local_rank : int
            Rank within ``group``.
        world_size : int
            Number of ranks in ``group``.
        """
        self.config = ring_config
        self.q, self.k, self.v, self.x, self.w_qkv = q, k, v, x, w_qkv
        self.group = group
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
        """Recompute the key and value from the current input chunk."""
        self.kv_j, self.kj, self.vj = _compute_kv(
            self.x_j, self.w_qkv, self.b, self.n_local, self.h, self.d
        )

    def _recv(self, recv_buf: torch.Tensor) -> None:
        """Store a received buffer as the current chunk."""
        self.x_j, self.kv_j, self.kj, self.vj = self.config.forward_recv_fn(
            recv_buf, self.x_j, self.kv_j, self.kj, self.vj
        )

    # -- interface ----------------------------------------------------- #
    @abstractmethod
    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the key and value of chunk ``step``."""

    def global_chunk_idx(self, step: int) -> int:
        """Return the global index of the chunk held at ``step``."""
        raise NotImplementedError

    def save_for_backward(self, step: int) -> None:
        """Record the current chunk for the backward pass."""
        if self.config.save_scope != SaveScope.GLOBAL:
            return
        self.chunks_to_save[self.global_chunk_idx(step)] = (
            self.config.save_for_backward_fn(self.x_j, self.kv_j)
        )

    def advance(self, step: int, world_size: int) -> None:
        """Progress communication so that chunk ``step + 1`` is available."""
        return

    def result(self) -> tuple[Optional[torch.Tensor], ...]:
        """Return the chunks saved for the backward pass."""
        return tuple(self.chunks_to_save)


# --------------------------------------------------------------------------- #
# plain P2P / all-to-all ring
# --------------------------------------------------------------------------- #
class _RingForwardChunkProvider(_ForwardChunkProvider):
    """Ring with one P2P or all-to-all send and receive per step."""

    def __init__(self, *args: object) -> None:
        """Start the first ring send."""
        super().__init__(*args)  # type: ignore[arg-type]
        self._recv_buf, self._handle = self.config.forward_send_fn(
            self.x_j, self.kv_j, direction=-1, group=self.group, tag=0
        )

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Receive chunk ``step``, relay it and return its key and value.

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
            assert self._handle is not None
            self._handle.wait()
            self._recv(self._recv_buf)
            if step + 1 < self.world_size:
                self._recv_buf, self._handle = self.config.forward_send_fn(
                    self.x_j,
                    self.kv_j,
                    direction=-1,
                    group=self.group,
                    tag=step,
                )
            if self.send_x:
                self._kv_from_x()
        return self.kj, self.vj

    def global_chunk_idx(self, step: int) -> int:
        """Return the global index of the chunk held at ``step``."""
        return (self.local_rank + step) % self.world_size


# --------------------------------------------------------------------------- #
# broadcast ring
# --------------------------------------------------------------------------- #
class _BroadcastForwardChunkProvider(_ForwardChunkProvider):
    """Ring in which rank ``s`` broadcasts its chunk at step ``s``."""

    def __init__(self, *args: object) -> None:
        """Initialize the broadcast state."""
        super().__init__(*args)  # type: ignore[arg-type]
        self._recv_buf: torch.Tensor = get_empty(self.q.device, self.q.dtype)
        self._handle: Optional[dist.Work] = None

    def _kick(self, src_rank: int) -> None:
        """Start the broadcast of the chunk of ``src_rank``."""
        self._recv_buf, self._handle = self.config.forward_send_fn(
            x=self.x,
            kv=self.kv,
            local_rank=self.local_rank,
            group=self.group,
            src_rank=src_rank % self.world_size,
        )

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Complete the broadcast of chunk ``step`` and start the next one.

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

    def global_chunk_idx(self, step: int) -> int:
        """Return the global index of the chunk held at ``step``."""
        return step


# --------------------------------------------------------------------------- #
# hybrid: shared intra-node + inter-node plumbing
# --------------------------------------------------------------------------- #
class _HybridForwardChunkProvider(_ForwardChunkProvider):
    """Base class of the hybrid intra-node and inter-node providers."""

    def __init__(self, *args: object) -> None:
        """Resolve the topology and the inter-node peers."""
        super().__init__(*args)  # type: ignore[arg-type]
        self.topo = get_hybrid_topology(self.group, self.config.gpus_per_node)
        self.gpn = self.topo.gpus_per_node
        self.n_nodes = self.topo.n_nodes
        # Inter-node hop: send to the previous node, receive from the next.
        self._inter_ranks = [self.topo.inter_peer(-1), self.topo.inter_peer(1)]
        self._inter_recv_buf: Optional[torch.Tensor] = None
        self._inter_handle: Optional[dist.Work] = None

    def _launch_inter_forward(self, node_step: int) -> None:
        """Start the inter-node send of the local chunk."""
        buf = torch.empty_like(self.x_j if self.send_x else self.kv_j)
        self._inter_recv_buf, self._inter_handle = self.config.p2p_fn(
            data=(self.x.clone() if self.send_x else self.kv.clone()),
            buffer=buf,
            send_recv_ranks=self._inter_ranks,
            group=self.group,
            async_op=True,
            tag=node_step,
        )


# --------------------------------------------------------------------------- #
# hybrid: intra-node ring + inter-node P2P
# --------------------------------------------------------------------------- #
class _HybridP2PForwardChunkProvider(_HybridForwardChunkProvider):
    """Hybrid ring with P2P intra-node rotation."""

    def __init__(self, *args: object) -> None:
        """Resolve the intra-node peers."""
        super().__init__(*args)  # type: ignore[arg-type]
        # Intra-node ring: send to the previous local rank, receive from the next.
        self._intra_ranks = [self.topo.intra_peer(-1), self.topo.intra_peer(1)]
        self._intra_recv_buf: Optional[torch.Tensor] = None
        self._intra_handle: Optional[dist.Work] = None

    def _launch_intra_forward(self, inner: int) -> None:
        """Start the intra-node send of the current chunk."""
        src = self.x_j if self.send_x else self.kv_j
        self._intra_recv_buf, self._intra_handle = self.config.send_op_fn(
            src,
            torch.empty_like(src),
            send_recv_ranks=self._intra_ranks,
            group=self.group,
            tag=inner,
        )

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Make chunk ``step`` current and return its key and value.

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
                self._launch_inter_forward(node_step)
        if inner + 1 < self.gpn:
            self._launch_intra_forward(inner)

        if self.send_x:
            self._kv_from_x()
        else:
            self.kj, self.vj = self.kv_j[0], self.kv_j[1]
        return self.kj, self.vj

    def advance(self, step: int, world_size: int) -> None:
        """Complete the intra-node send.

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

    def global_chunk_idx(self, step: int) -> int:
        """Return the global index of the chunk held at ``step``."""
        node_step, inner = divmod(step, self.gpn)
        cur_node = (self.topo.node_idx + node_step) % self.n_nodes
        return cur_node * self.gpn + (self.topo.local_idx + inner) % self.gpn


# --------------------------------------------------------------------------- #
# hybrid: intra-node broadcast + inter-node P2P
# --------------------------------------------------------------------------- #
class _HybridBroadcastForwardChunkProvider(_HybridForwardChunkProvider):
    """Hybrid ring with intra-node broadcasts."""

    def __init__(self, *args: object) -> None:
        """Resolve the intra-node broadcast group."""
        super().__init__(*args)  # type: ignore[arg-type]
        self._intranode_group = self.topo.intranode_group
        # The chunk this rank contributes to the current node round.
        self._own_kv: Optional[torch.Tensor] = None
        self._own_buf: Optional[torch.Tensor] = None

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Broadcast the chunk of ``step`` and return its key and value.

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
                self._launch_inter_forward(node_step)

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

    def global_chunk_idx(self, step: int) -> int:
        """Return the global index of the chunk held at ``step``."""
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
    local_rank: int,
    world_size: int,
) -> _ForwardChunkProvider:
    """Build the forward chunk provider of ``ring_config.send_op``.

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
        Process group of the ring.
    local_rank : int
        Rank of the current process within ``group``.
    world_size : int
        Number of ranks in ``group``.

    Returns
    -------
    _ForwardChunkProvider
        Chunk provider.
    """
    args = (ring_config, q, k, v, x, w_qkv, group, local_rank, world_size)
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
