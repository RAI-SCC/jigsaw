"""Chunk providers for the ring-attention forward pass.

Each provider implements one communication strategy through the hooks ``get``,
``save_for_backward`` and ``advance``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, cast

import torch
import torch.distributed as dist

from jigsaw._core.comm import get_empty
from jigsaw.modules.attention._config import (
    RingAttentionConfig,
    SaveScope,
    SendForward,
    SendOp,
)

from ._comm_overlap import PrefetchBuffer, consume, issue_on_comm_stream
from .ring import _bhnd, _compute_kv, get_hybrid_topology


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
            Local query, key, and value tensors in ``ring_config.core_layout``.
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
        self.layout = ring_config.core_layout
        self.b, self.h, self.n_local, self.d = _bhnd(q.shape, self.layout)
        self.send_x = ring_config.send_in_forward == SendForward.X

        # Local KV, kept private so an in-flight send can never alias k / v
        # (``torch.concat`` already copies, so no ``.clone()``).
        self.kv = torch.concat([k.unsqueeze(0), v.unsqueeze(0)]).contiguous()

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

    def _kv_from_x(self) -> None:
        """Recompute the key and value from the current input chunk."""
        self.kv_j, self.kj, self.vj = _compute_kv(
            self.x_j, self.w_qkv, self.b, self.n_local, self.h, self.d, self.layout
        )

    def _recv(self, recv_buf: torch.Tensor) -> None:
        """Store a received buffer as the current chunk."""
        self.x_j, self.kv_j, self.kj, self.vj = self.config.forward_recv_fn(
            recv_buf, self.x_j, self.kv_j, self.kj, self.vj
        )

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
        """Progress communication so that chunk ``step + 1`` is available.

        Parameters
        ----------
        step : int
            Current ring step.
        world_size : int
            Number of ranks in the ring.
        """
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
        self._start_first_send()

    def _start_first_send(self) -> None:
        """Issue the first ring send (local chunk). Overridden by the overlap variant."""
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


class _RingForwardChunkProviderOverlap(_RingForwardChunkProvider):
    """:class:`_RingForwardChunkProvider` with the ring send on a dedicated stream.

    Two receive buffers alternate. ``get(step)`` issues the send of chunk
    ``step + 1`` on ``ring_config.comm_stream`` before the step's attention core
    runs, so the transfer overlaps the core instead of blocking the host in
    ``Work.wait()``. Used only when :attr:`RingAttentionConfig.overlap_active`.
    """

    def _start_first_send(self) -> None:
        """Allocate the ping-pong buffers and kick the local chunk on the comm stream."""
        self._cs = self.config.comm_stream
        src = self.x_j if self.send_x else self.kv_j
        # Two buffers of the circulated tensor's shape. `_pending` names the one
        # the next `get` will read; the other is free to receive into.
        self._bufs = [torch.empty_like(src), torch.empty_like(src)]
        self._pending = 0
        self._work: object = None
        self._event: torch.cuda.Event | None = None
        self._kick(src, into=0, tag=0)

    def _kick(self, src: torch.Tensor, into: int, tag: int) -> None:
        """Issue a ``direction=-1`` ring send of ``src`` into ``self._bufs[into]``."""
        buf = self._bufs[into]
        assert self._cs is not None
        _, self._work, self._event = issue_on_comm_stream(
            self._cs,
            lambda: self.config.send_op_fn(
                src, buf, direction=-1, group=self.group, async_op=True, tag=tag
            ),
            keep_alive=(src, buf),
        )
        self._pending = into

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Adopt chunk ``step`` off the comm stream and prefetch ``step + 1``."""
        if step > 0:
            assert self._event is not None
            recv_buf = self._bufs[self._pending]
            consume(self._event, recv_buf)
            self._recv(recv_buf)
            if step + 1 < self.world_size:
                # `_recv` rebound x_j / kv_j onto `recv_buf`; relay it into the
                # other buffer. The next `_kick` waits for the compute stream to
                # finish reading this buffer before overwriting it.
                src = self.x_j if self.send_x else self.kv_j
                self._kick(src, into=1 - self._pending, tag=step)
            if self.send_x:
                self._kv_from_x()
        return self.kj, self.vj


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
                # This rank's own chunk: reuse local K/V instead of recomputing.
                self.kv_j, self.kj, self.vj = self.kv, self.k, self.v
            else:
                self._kv_from_x()
        return self.kj, self.vj

    def global_chunk_idx(self, step: int) -> int:
        """Return the global index of the chunk held at ``step``."""
        return step


# --------------------------------------------------------------------------- #
# hybrid: shared intra-group + outer-ring plumbing
# --------------------------------------------------------------------------- #
class _HybridForwardChunkProvider(_ForwardChunkProvider):
    """Base class of the hybrid providers (intra group plus outer ring)."""

    def __init__(self, *args: object) -> None:
        """Resolve the topology and the outer-ring peers."""
        super().__init__(*args)  # type: ignore[arg-type]
        self.topo = get_hybrid_topology(self.group, self.config.gpus_per_node)
        self.gpn = self.topo.gpus_per_node
        self.n_nodes = self.topo.n_nodes
        # Outer ring: send to the previous intra group and receive from the
        # next, addressed inside `internode_group` so the intra group's
        # communicator can progress alongside.
        self._inter_group = self.topo.internode_group
        self._inter_ranks = [self.topo.internode_peer(-1), self.topo.internode_peer(1)]
        self._inter_recv_buf: Optional[torch.Tensor] = None
        self._inter_handle: Optional[dist.Work] = None

    def _launch_inter_forward(self, node_step: int) -> None:
        """Start this round's outer-ring send of the chunk in hand.

        Sends ``x_j`` / ``kv_j``: this rank's own chunk at round 0, then
        whatever the previous round's outer-ring receive delivered. Relaying
        the held chunk, not the original ``x`` / ``kv``, is what lets chunks
        travel past the neighboring intra group when ``n_nodes > 2``. The source is never
        written in place afterwards, so it is sent without a clone.
        """
        src = self.x_j if self.send_x else self.kv_j
        self._inter_recv_buf, self._inter_handle = self.config.p2p_fn(
            data=src,
            buffer=torch.empty_like(src),
            send_recv_ranks=self._inter_ranks,
            group=self._inter_group,
            async_op=True,
            tag=node_step,
        )


# --------------------------------------------------------------------------- #
# hybrid: intra-group ring + outer-ring P2P
# --------------------------------------------------------------------------- #
class _HybridP2PForwardChunkProvider(_HybridForwardChunkProvider):
    """Hybrid ring rotating chunks within the intra group (P2P on the outer ring)."""

    def __init__(self, *args: object) -> None:
        """Resolve the intra-group peers."""
        super().__init__(*args)  # type: ignore[arg-type]
        # Intra-group ring: send to the previous local rank, receive from the
        # next, addressed inside `intranode_group`. For the all-to-all send ops
        # this limits the all-to-all to the `gpus_per_node` ranks that exchange
        # anything, rather than the whole ring.
        self._intra_group = self.topo.intranode_group
        self._intra_ranks = [self.topo.intranode_peer(-1), self.topo.intranode_peer(1)]
        self._intra_recv_buf: Optional[torch.Tensor] = None
        self._intra_handle: Optional[dist.Work] = None

    def _launch_intra_forward(self, inner: int) -> None:
        """Start the intra-group send of the current chunk."""
        src = self.x_j if self.send_x else self.kv_j
        self._intra_recv_buf, self._intra_handle = self.config.send_op_fn(
            src,
            torch.empty_like(src),
            send_recv_ranks=self._intra_ranks,
            group=self._intra_group,
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
        """Complete the intra-group send.

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


class _HybridP2PForwardChunkProviderOverlap(_HybridP2PForwardChunkProvider):
    """:class:`_HybridP2PForwardChunkProvider` with both sends on the comm stream.

    Prefetches both transfers on the comm stream: the send within the intra
    group and the send to the previous intra group around the ring. Each is started
    in ``get(step)``, before the step's attention core runs, and waited on only
    when its chunk is needed: the intra-group one in ``advance``, the other at
    the start of the next round. Used only when
    :attr:`RingAttentionConfig.overlap_active`.
    """

    def __init__(self, *args: object) -> None:
        """Allocate the intra-group and outer-ring prefetch buffers."""
        super().__init__(*args)  # type: ignore[arg-type]
        cs = self.config.comm_stream
        assert cs is not None
        like = self.x_j if self.send_x else self.kv_j
        self._intra_buf = PrefetchBuffer(cs, like)
        self._inter_buf = PrefetchBuffer(cs, like)

    def _launch_inter_forward(self, node_step: int) -> None:
        """Prefetch this round's outer-ring send on the comm stream."""
        src = self.x_j if self.send_x else self.kv_j

        def produce(buf: torch.Tensor) -> "tuple[torch.Tensor, Optional[dist.Work]]":
            return cast(
                "tuple[torch.Tensor, Optional[dist.Work]]",
                self.config.p2p_fn(
                    data=src,
                    buffer=buf,
                    send_recv_ranks=self._inter_ranks,
                    group=self._inter_group,
                    async_op=True,
                    tag=node_step,
                ),
            )

        self._inter_buf.kick(produce, extra_keep_alive=(src,))

    def _launch_intra_forward(self, inner: int) -> None:
        """Prefetch this step's intra-group send on the comm stream."""
        src = self.x_j if self.send_x else self.kv_j

        def produce(buf: torch.Tensor) -> "tuple[torch.Tensor, Optional[dist.Work]]":
            return cast(
                "tuple[torch.Tensor, Optional[dist.Work]]",
                self.config.send_op_fn(
                    src,
                    buf,
                    send_recv_ranks=self._intra_ranks,
                    group=self._intra_group,
                    async_op=True,
                    tag=inner,
                ),
            )

        self._intra_buf.kick(produce, extra_keep_alive=(src,))

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Adopt chunk ``step`` off the comm stream, then prefetch what comes next."""
        node_step, inner = divmod(step, self.gpn)
        if inner == 0:
            if node_step > 0:
                self._recv(self._inter_buf.wait())
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
        """Adopt the intra-group send prefetched in ``get`` so the next inner step is ready."""
        _, inner = divmod(step, self.gpn)
        if inner + 1 < self.gpn:
            self._recv(self._intra_buf.wait())


# --------------------------------------------------------------------------- #
# hybrid: intra-group broadcast + outer-ring P2P
# --------------------------------------------------------------------------- #
class _HybridBroadcastForwardChunkProvider(_HybridForwardChunkProvider):
    """Hybrid ring with broadcasts within the intra group."""

    def __init__(self, *args: object) -> None:
        """Resolve the intra-group broadcast group."""
        super().__init__(*args)  # type: ignore[arg-type]
        self._intranode_group = self.topo.intranode_group
        # The chunk this rank contributes to the current round.
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
        # `inner` is a local index, i.e. the source's rank inside
        # `intranode_group`: hence `group_src`, not the global `src`, which a
        # ring that is a mesh sub-group would get wrong.
        is_src = self.topo.local_idx == inner
        recv_buf = self._own_buf if is_src else torch.empty_like(self._own_buf)
        dist.broadcast(recv_buf, group_src=inner, group=self._intranode_group)

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


class _HybridBroadcastForwardChunkProviderOverlap(_HybridBroadcastForwardChunkProvider):
    """:class:`_HybridBroadcastForwardChunkProvider` with its transfers on the comm stream.

    Prefetches both transfers on the comm stream: the broadcasts within the
    intra group and the send to the previous intra group around the ring, which
    works as in :class:`_HybridP2PForwardChunkProviderOverlap`. A round's
    broadcasts depend only on the chunk each rank contributes to that round,
    so each ``get`` waits on the broadcast started by the previous call and
    starts the next one. A round's first broadcast is started and waited on
    in the same call.
    """

    def __init__(self, *args: object) -> None:
        """Allocate the intra-group and outer-ring prefetch buffers."""
        super().__init__(*args)  # type: ignore[arg-type]
        cs = self.config.comm_stream
        assert cs is not None
        like = self.x_j if self.send_x else self.kv_j
        self._bc_buf = PrefetchBuffer(cs, like)
        self._inter_buf = PrefetchBuffer(cs, like)

    def _launch_inter_forward(self, node_step: int) -> None:
        """Prefetch this round's outer-ring send on the comm stream."""
        src = self.x_j if self.send_x else self.kv_j

        def produce(buf: torch.Tensor) -> "tuple[torch.Tensor, Optional[dist.Work]]":
            return cast(
                "tuple[torch.Tensor, Optional[dist.Work]]",
                self.config.p2p_fn(
                    data=src,
                    buffer=buf,
                    send_recv_ranks=self._inter_ranks,
                    group=self._inter_group,
                    async_op=True,
                    tag=node_step,
                ),
            )

        self._inter_buf.kick(produce, extra_keep_alive=(src,))

    def _kick_broadcast(self, inner: int) -> None:
        """Prefetch this round's intra-group broadcast for local index ``inner``."""
        is_src = self.topo.local_idx == inner
        own = self._own_buf
        assert own is not None

        def produce(buf: torch.Tensor) -> "tuple[torch.Tensor, Optional[dist.Work]]":
            if is_src:
                # Broadcast reads the source's own tensor in place; the
                # persistent buffer has to hold that content before the
                # collective fires.
                buf.copy_(own)
            work = dist.broadcast(
                buf, group_src=inner, group=self._intranode_group, async_op=True
            )
            return buf, work

        self._bc_buf.kick(produce, extra_keep_alive=(own,) if is_src else ())

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Wait on the broadcast of chunk ``step``, start the next one, and return its key and value."""
        node_step, inner = divmod(step, self.gpn)
        if inner == 0:
            if node_step > 0:
                self._recv(self._inter_buf.wait())
                if self.send_x:
                    self._kv_from_x()
            self._own_kv = self.kv_j
            self._own_buf = self.x_j if self.send_x else self.kv_j
            if node_step < self.n_nodes - 1:
                self._launch_inter_forward(node_step)
            self._kick_broadcast(0)

        recv_buf = self._bc_buf.wait()
        if inner + 1 < self.gpn:
            self._kick_broadcast(inner + 1)

        is_src = self.topo.local_idx == inner
        if self.send_x:
            self.x_j = recv_buf
            if is_src:
                assert self._own_kv is not None
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
            if ring_config.overlap_active:
                return _RingForwardChunkProviderOverlap(*args)
            return _RingForwardChunkProvider(*args)
        case SendOp.HYBRID_P2P | SendOp.HYBRID_ALLTOALL:
            if ring_config.overlap_active:
                return _HybridP2PForwardChunkProviderOverlap(*args)
            return _HybridP2PForwardChunkProvider(*args)
        case SendOp.HYBRID_BROADCAST:
            if ring_config.overlap_active:
                return _HybridBroadcastForwardChunkProviderOverlap(*args)
            return _HybridBroadcastForwardChunkProvider(*args)
        case _:
            raise NotImplementedError(f"Unknown send_op: {ring_config.send_op}")
