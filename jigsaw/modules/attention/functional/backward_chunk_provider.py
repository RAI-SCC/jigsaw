"""Chunk providers for ring-attention backward passes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, cast

import torch
import torch.distributed as dist

from jigsaw.comm import get_empty
from jigsaw.modules.attention.config import (
    RingAttentionConfig,
    SaveBackward,
    SaveScope,
    SendForward,
    SendOp,
)
from jigsaw.offload import NVLinkActivationOffloader

from ._comm_overlap import consume, issue_on_comm_stream
from .functional import _compute_kv, get_hybrid_topology


def _build_backward_chunk_provider(
    ring_config: RingAttentionConfig,
    chunks_to_save: tuple[torch.Tensor, ...],
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    b: int,
    seq_local: int,
    h: int,
    d: int,
    group: dist.ProcessGroup,
    local_rank: int,
) -> _BackwardChunkProvider:
    """
    Build corresponding class that will return the Chunk provider.

    Parameters
    ----------
    ring_config: RingAttentionConfig
        Configuration for ring attention.
    chunks_to_save: tuple[torch.Tensor, ...]
        Chunks (either X or KV) saved between the forward and backward pass.
    x: torch.Tensor
        Input data x.
    w_qkv: torch.Tensor
        Weight matrix for attention.
    b: int
        Batch size.
    seq_local: int
        Local sequence length.
    h: int
        Number of attention heads.
    d: int
        Per-head embedding dimension.
    group: dist.ProcessGroup
        Process group for ring attention.
    local_rank: int
        Rank within the distributed attention group.

    Returns
    -------
    _BackwardChunkProvider
    """
    if ring_config.save_scope == SaveScope.GLOBAL:
        return _GlobalChunkProvider(
            ring_config, chunks_to_save, w_qkv, b, seq_local, h, d
        )
    if ring_config.save_scope != SaveScope.LOCAL:
        raise NotImplementedError(f"Unknown save_scope: {ring_config.save_scope}")

    # Every LOCAL-scope provider takes the same positional args; only the plain
    # P2P/all-to-all ring has no use for the trailing `local_rank`.
    args = (
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
    match ring_config.send_op:
        case SendOp.P2P | SendOp.ALLTOALL:
            if ring_config.overlap_active:
                return _LocalChunkProviderOverlap(*args[:-1])
            return _LocalChunkProvider(*args[:-1])
        case SendOp.BROADCAST:
            return _LocalChunkProviderBroadcast(*args)
        case SendOp.HYBRID_P2P | SendOp.HYBRID_ALLTOALL:
            return _LocalChunkProviderHybridP2P(*args)
        case SendOp.HYBRID_BROADCAST:
            return _LocalChunkProviderHybridBroadcast(*args)
        case _:
            raise NotImplementedError(f"Unknown send_op: {ring_config.send_op}")


# --------------------------------------------------------------------------- #
# backward: dK/dV chunk providers
#
# How to obtain (kj, vj, kv) for a given ring step, including any prefetching/communication bookkeeping.
# --------------------------------------------------------------------------- #
class _BackwardChunkProvider(ABC):
    """Yields (kj, vj, kv) for each backward ring step."""

    @abstractmethod
    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return contiguous (kj, vj) and the underlying kv buffer for `step`."""

    def maybe_forward(self, step: int, ring_group_size: int) -> None:
        """Kick off communication for the next ring step, if any.

        No-op by default; providers that re-circulate chunks override this.

        Parameters
        ----------
        step : int
            Current ring step.
        ring_group_size : int
            Number of ranks in the ring group.
        """
        return


class _GlobalChunkProvider(_BackwardChunkProvider):
    """
    Look up the relevant chunk that was saved during the forward pass.

    If the data is offloaded to the CPU, the data movement is handeled here.
    """

    def __init__(
        self,
        ring_config: RingAttentionConfig,
        chunks_to_save: tuple[torch.Tensor, ...],
        w_qkv: torch.Tensor,
        b: int,
        seq_local: int,
        h: int,
        d: int,
    ) -> None:
        """
        Initialize _GlobalChunkProvider.

        Parameters
        ----------
        ring_config: RingAttentionConfig
            Configuration class for ring attention.
        chunks_to_save: tuple[torch.Tensor, ...]
            Chunks saved between the forward and backward pass.
        w_qkv: torch.Tensor
            Weights of the QKV mapping.
        b: int
            Batch dimension.
        seq_local: int
            Local sequence length.
        h: int
            Number of attention heads.
        d: int
            Embedding dimension per head.
        """
        self.config = ring_config
        self.chunks = chunks_to_save
        self.w_qkv = w_qkv
        self.b, self.seq_local, self.h, self.d = b, seq_local, h, d

        self._cur_gpu: Optional[torch.Tensor] = None
        self._cur_event: Optional[torch.cuda.Event] = None
        self._next_gpu: Optional[torch.Tensor] = None
        self._next_event: Optional[torch.cuda.Event] = None

        if ring_config.offloader is not None:
            self._cur_gpu, self._cur_event = self._start_prefetch(0)
            self._next_gpu, self._next_event = self._start_prefetch(1)

    def _start_prefetch(
        self, idx: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.cuda.Event]]:
        """
        Begin prefetching process for a given chunk.

        Parameters
        ----------
        idx: int
            Step of the ring communication.

        Returns
        -------
        Prefetched variable and event.
        """
        if self.config.offloader is None or idx >= len(self.chunks):
            return None, None
        return self.config.offloader.prefetch(self.chunks[idx])

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the current chunk and kick off the process for fetching the next chunk.

        Parameters
        ----------
        step: int
            Current step of ring attention.

        Returns
        -------
        Values of k, v, and kv (duplicate information, retained to reduce number of
        messages sent in future steps.)
        """
        offloader = self.config.offloader
        if isinstance(offloader, NVLinkActivationOffloader):
            offloader.wait_prefetch(cast(torch.cuda.Event, self._cur_event))
            chunk = cast(torch.Tensor, self._cur_gpu)
            # advance the double buffer, kicking off the next prefetch ASAP
            self._cur_gpu, self._cur_event = self._next_gpu, self._next_event
            self._next_gpu, self._next_event = self._start_prefetch(step + 2)
        else:
            chunk = self.chunks[step]

        if self.config.save_for_backward == SaveBackward.KV:
            kv = chunk
            return kv[0].contiguous(), kv[1].contiguous(), kv
        elif self.config.save_for_backward == SaveBackward.X:
            kv, kj, vj = _compute_kv(
                chunk, self.w_qkv, self.b, self.seq_local, self.h, self.d, self.config.core_layout
            )
            return kj.contiguous(), vj.contiguous(), kv
        else:
            raise NotImplementedError


class _LocalChunkProvider(_BackwardChunkProvider):
    """
    Provide the relevant chunk for a given step.

    Only the local (step-0) chunk (either X or KV) was saved;
    Every other rank's chunk is re-circulated around the ring with one P2P send/recv per step, mirroring
    the forward pass's communication pattern but in reverse role.
    """

    def __init__(
        self,
        ring_config: RingAttentionConfig,
        chunks_to_save: tuple[torch.Tensor, ...],
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        b: int,
        seq_local: int,
        h: int,
        d: int,
        group: dist.ProcessGroup,
    ) -> None:
        """
        Initialize LocalChunkProvider.

        Parameters
        ----------
        ring_config: RingAttentionConfig
            configuration for ring attention.
        chunks_to_save: tuple[torch.Tensor, ...]
            Chunks that were saved from the forward pass.
        x: torch.Tensor
            X data.
        w_qkv: torch.Tensor
            Weight matrix for attention.
        b: int
            Batch size.
        seq_local: int
            Local sequence length.
        h: int
            Number of local heads.
        d: int
            Embedding dimension per head.
        group: dist.ProcessGroup
            Process group upon which ring attention is performed.
        """
        self.config = ring_config
        self.chunk0 = chunks_to_save[0]
        self.x = x
        self.w_qkv = w_qkv
        self.b, self.seq_local, self.h, self.d = b, seq_local, h, d
        self.group = group

        self._recv_buf = get_empty(x.device, x.dtype)
        self._handle: Optional[dist.Work] = None

    def _materialize_chunk0(self) -> torch.Tensor:
        """
        Load local saved chunk onto device.

        If the data is not offloaded and is already on the device,
        simply return the data item. Otherwise, if it is offloaded,
        fetch the sample.

        Returns
        -------
        chunk: torch.Tensor
            Data chunk that was offloaded.
        """
        chunk = self.chunk0
        if isinstance(self.config.offloader, NVLinkActivationOffloader):
            gpu, ev = self.config.offloader.prefetch(chunk)
            self.config.offloader.wait_prefetch(ev)
            chunk = gpu
        return chunk

    def _kv0(self) -> torch.Tensor:
        """KV tensor for this rank's own (step-0) saved chunk.

        Returns
        -------
        torch.Tensor
            The ``kv`` buffer, taken straight from the saved chunk when KV was
            saved, or recomputed from the saved ``x`` otherwise.
        """
        chunk = self._materialize_chunk0()
        if self.config.save_for_backward == SaveBackward.KV:
            return chunk
        kv, _, _ = _compute_kv(
            chunk, self.w_qkv, self.b, self.seq_local, self.h, self.d, self.config.core_layout
        )
        return kv

    def _kv_from(
        self, recv_buf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack a received ring buffer into contiguous ``(kj, vj, kv)``.

        Parameters
        ----------
        recv_buf : torch.Tensor
            The buffer that just arrived over the ring -- an ``x`` chunk when
            ``send_in_forward`` is ``X``, otherwise a fused ``kv`` buffer.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Contiguous ``kj``, ``vj`` and the underlying ``kv`` buffer.
        """
        if self.config.send_in_forward == SendForward.X:
            kv, kj, vj = _compute_kv(
                recv_buf, self.w_qkv, self.b, self.seq_local, self.h, self.d, self.config.core_layout
            )
            return kj.contiguous(), vj.contiguous(), kv
        kv = recv_buf
        return kv[0].contiguous(), kv[1].contiguous(), kv

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the tensors for the requested ring step.

        Parameters
        ----------
        step : int
            Current ring step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Result returned by the operation.
        """
        if step == 0:
            kv = self._kv0()

            # Kick off the first ring send of whichever buffer is configured.
            send_buf = self.x if self.config.send_in_forward == SendForward.X else kv
            self._recv_buf = torch.empty_like(send_buf)
            self._recv_buf, self._handle = self.config.send_op_fn(
                send_buf,
                self._recv_buf,
                direction=-1,
                group=self.group,
                async_op=True,
                step=step,
            )

            if self.config.send_in_forward == SendForward.X:
                # Local Q always attends its own local K/V on step 0.
                kv, kj, vj = _compute_kv(
                    self.x, self.w_qkv, self.b, self.seq_local, self.h, self.d, self.config.core_layout
                )
            else:
                kj, vj = kv[0].contiguous(), kv[1].contiguous()
            return kj, vj, kv

        # step > 0: receive the chunk that's been circulating; `maybe_forward`
        # relays it on (unless this is the last step).
        assert self._handle is not None
        self._handle.wait()
        return self._kv_from(self._recv_buf)

    def maybe_forward(self, step: int, ring_group_size: int) -> None:
        """
        Send chunk to next partner rank if necessary.

        Chunk does not get sent if it is the first (treated differently) or final (done) step.

        Parameters
        ----------
        step: int
            stage of the ring attention.
        ring_group_size: int
            Size of the domain-parallel group.
        """
        if step == 0 or step >= ring_group_size - 1:
            return
        next_recv = torch.empty_like(self._recv_buf)
        self._recv_buf, self._handle = self.config.send_op_fn(
            self._recv_buf,
            next_recv,
            direction=-1,
            group=self.group,
            async_op=True,
            step=step,
        )


class _LocalChunkProviderOverlap(_LocalChunkProvider):
    """:class:`_LocalChunkProvider` with the K/V relay on a dedicated stream.

    Two persistent buffers ping-pong. :func:`~.backward.ring_backward` calls
    :meth:`maybe_forward` right after :meth:`get` (before the attention core)
    when overlap is on, so the relay of the next chunk runs on
    ``ring_config.comm_stream`` under the core instead of after it. Selected by
    :func:`_build_backward_chunk_provider` only when
    :attr:`RingAttentionConfig.overlap_active`; the matching dK/dV relay is
    :class:`~.backward._DkvRelay`.
    """

    def __init__(self, *args: object) -> None:
        """Defer buffer sizing to step 0 (needs the K/V shape)."""
        super().__init__(*args)  # type: ignore[arg-type]
        self._cs = self.config.comm_stream
        self._bufs: Optional[list[torch.Tensor]] = None
        self._pending = 0
        self._work: object = None
        self._event: Optional[torch.cuda.Event] = None

    def _kick(self, src: torch.Tensor, into: int, step: int) -> None:
        """Issue a ``direction=-1`` ring send of ``src`` into ``self._bufs[into]``."""
        assert self._bufs is not None
        buf = self._bufs[into]
        _, self._work, self._event = issue_on_comm_stream(
            self._cs,
            lambda: self.config.send_op_fn(
                src, buf, direction=-1, group=self.group, async_op=True, step=step
            ),
            keep_alive=(src, buf),
        )
        self._pending = into

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return chunk ``step`` -- local on step 0, else adopted off the comm stream."""
        if step == 0:
            kv = self._kv0()
            send_buf = (
                self.x if self.config.send_in_forward == SendForward.X else kv
            )
            self._bufs = [torch.empty_like(send_buf), torch.empty_like(send_buf)]
            self._kick(send_buf, into=0, step=step)
            if self.config.send_in_forward == SendForward.X:
                kv, kj, vj = _compute_kv(
                    self.x, self.w_qkv, self.b, self.seq_local, self.h, self.d,
                    self.config.core_layout,
                )
            else:
                kj, vj = kv[0].contiguous(), kv[1].contiguous()
            return kj, vj, kv

        assert self._event is not None and self._bufs is not None
        recv_buf = self._bufs[self._pending]
        consume(self._event, recv_buf)
        return self._kv_from(recv_buf)

    def maybe_forward(self, step: int, ring_group_size: int) -> None:
        """Relay the chunk held at ``step`` onward, on the comm stream.

        Called before the step's attention core (not after) so the transfer
        overlaps it. Step 0's relay was already issued in :meth:`get`; the last
        step has nothing to prefetch.
        """
        if step == 0 or step >= ring_group_size - 1:
            return
        assert self._bufs is not None
        self._kick(self._bufs[self._pending], into=1 - self._pending, step=step)


class _LocalChunkProviderBroadcast(_LocalChunkProvider):
    """Backward broadcast ring: at step ``s`` rank ``s`` fans its chunk to the group.

    The backward twin of
    :class:`~.forward_chunk_provider._BroadcastForwardChunkProvider`. The trailing
    ``local_rank`` positional arg picks which step this rank is the source on.
    """

    def __init__(self, *args: object) -> None:
        """Size the broadcast recv buffer and record this rank's ring position."""
        *base_args, local_rank = args
        super().__init__(*base_args)  # type: ignore[arg-type]
        self._recv_buf = torch.empty_like(
            self.x if self.config.send_in_forward == SendForward.X else self.chunk0
        )
        self._handle: Optional[dist.Work] = None
        self.local_rank = local_rank

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the tensors for the requested ring step.

        Parameters
        ----------
        step : int
            Current ring step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Result returned by the operation.
        """
        kv = self._kv0()

        # Kick off the first ring send of whichever buffer is configured.
        send_buf = self.x if self.config.send_in_forward == SendForward.X else kv
        self._recv_buf = (
            send_buf if self.local_rank == step else torch.empty_like(send_buf)
        )
        self._handle = dist.broadcast(
            self._recv_buf, src=step, group=self.group, async_op=True
        )

        assert self._handle is not None
        self._handle.wait()
        return self._kv_from(self._recv_buf)

    def maybe_forward(self, step: int, ring_group_size: int) -> None:
        """Do nothing: every step is a fresh broadcast issued in ``get``.

        Parameters
        ----------
        step : int
            Current ring step.
        ring_group_size : int
            Number of ranks in the ring group.

        """
        return


class _LocalChunkProviderHybrid(_LocalChunkProvider):
    """Shared plumbing for the two hybrid (intra-node + inter-node) providers.

    Both hybrids move chunks between nodes with the same reversed inter-node P2P
    ring (send to the next node, receive from the previous -- opposite the
    forward). They differ only in how a chunk travels *within* a node:
    :class:`_LocalChunkProviderHybridP2P` rotates it one P2P hop per step,
    :class:`_LocalChunkProviderHybridBroadcast` keeps it put and broadcasts. This
    base holds the common state and the inter-node launch helper; each subclass
    implements ``get`` / ``maybe_forward`` for its intra-node scheme.
    """

    def __init__(
        self,
        ring_config: RingAttentionConfig,
        chunks_to_save: tuple[torch.Tensor, ...],
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        b: int,
        seq_local: int,
        h: int,
        d: int,
        group: dist.ProcessGroup,
        local_rank: int,
    ) -> None:
        """Initialize the shared hybrid state.

        Parameters
        ----------
        ring_config : RingAttentionConfig
            Ring-attention configuration controlling communication and saving.
        chunks_to_save : tuple[torch.Tensor, ...]
            Saved activation chunks used to reconstruct backward inputs.
        x : torch.Tensor
            Input tensor.
        w_qkv : torch.Tensor
            Fused query, key, and value projection weights.
        b : int
            Batch size for the input tensors.
        seq_local : int
            Sequence length owned by the current process.
        h : int
            Number of attention heads.
        d : int
            Embedding dimension per attention head.
        group : dist.ProcessGroup
            Process group used for distributed communication.
        local_rank : int
            Rank of the current process.
        """
        super().__init__(
            ring_config, chunks_to_save, x, w_qkv, b, seq_local, h, d, group
        )
        self._recv_buf = torch.empty_like(
            x if ring_config.send_in_forward == SendForward.X else chunks_to_save[0]
        )
        self._handle: Optional[dist.Work] = None
        self.local_rank = int(local_rank)
        self.topo = get_hybrid_topology(group, ring_config.gpus_per_node)
        self._gpn = self.topo.gpus_per_node
        # Inter-node hop: send to the next node, receive from the previous
        # (opposite the forward's node ring).
        self._inter_ranks = [self.topo.inter_peer(1), self.topo.inter_peer(-1)]
        self._internode_handle: Optional[dist.Work] = None
        self._internode_recv_buf: Optional[torch.Tensor] = None

    def _launch_inter_backward(self, data: torch.Tensor, step: int) -> None:
        """Launch an inter-node transfer of ``data`` along the node ring.

        Parameters
        ----------
        data : torch.Tensor
            Tensor to send on.
        step : int
            Current ring step (only its intra-node phase is used as the tag).
        """
        buf = torch.empty_like(data)
        self._internode_recv_buf, self._internode_handle = self.config.p2p_fn(
            data=data,
            buffer=buf,
            send_recv_ranks=self._inter_ranks,
            group=self.group,
            async_op=True,
            tag=step % self._gpn,
        )


class _LocalChunkProviderHybridP2P(_LocalChunkProviderHybrid):
    """Hybrid backward chunk provider: P2P intra-node, P2P inter-node.

    The backward twin of :func:`hybrid_p2p_forward`. A chunk rotates through the
    ``gpus_per_node`` node-local ranks with one intra-node P2P send per step, and
    hops to the next node over the inter-node ring at each node boundary.
    """

    def __init__(self, *args: object) -> None:
        """Resolve the intra-node peer list on top of the shared hybrid state."""
        super().__init__(*args)  # type: ignore[arg-type]
        # Intra-node ring: send to the previous local rank, receive from the
        # next -- mirroring the forward.
        self._intra_ranks = [self.topo.intra_peer(-1), self.topo.intra_peer(1)]

    def _launch_intra_backward(self, src: torch.Tensor, step: int) -> None:
        """Start the intra-node ring send of ``src`` to the next node-local peer."""
        self._recv_buf, self._handle = self.config.send_op_fn(
            src,
            torch.empty_like(src),
            send_recv_ranks=self._intra_ranks,
            group=self.group,
            async_op=True,
            step=step,
        )

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the tensors for the requested ring step.

        Parameters
        ----------
        step : int
            Current ring step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Result returned by the operation.
        """
        if step == 0:
            kv = self._kv0()

            # Kick off the inter-node hop and the first intra-node ring send.
            self._launch_inter_backward(
                self.x.clone()
                if self.config.send_in_forward == SendForward.X
                else kv.clone(),
                step,
            )

            send_buf = self.x if self.config.send_in_forward == SendForward.X else kv
            self._launch_intra_backward(send_buf, step)

            if self.config.send_in_forward == SendForward.X:
                # Local Q always attends its own local K/V on step 0.
                kv, kj, vj = _compute_kv(
                    self.x, self.w_qkv, self.b, self.seq_local, self.h, self.d, self.config.core_layout
                )
            else:
                kj, vj = kv[0].contiguous(), kv[1].contiguous()
            return kj, vj, kv

        # step > 0: wait on whichever link delivered this step's chunk -- the
        # inter-node hop at a node boundary, the intra-node ring otherwise.
        if step % self._gpn == 0:
            assert self._internode_handle is not None
            assert self._internode_recv_buf is not None
            self._internode_handle.wait()
            recv_buf = self._internode_recv_buf
        else:
            assert self._handle is not None
            self._handle.wait()
            recv_buf = self._recv_buf

        return self._kv_from(recv_buf)

    def maybe_forward(self, step: int, ring_group_size: int) -> None:
        """Relay this step's chunk to the next node-local (and node) peer.

        Parameters
        ----------
        step : int
            Current ring step.
        ring_group_size : int
            Number of ranks in the ring group.
        """
        if step == 0 or step >= ring_group_size - 1:
            return

        node_step = step // self._gpn
        if step % self._gpn == 0:
            assert self._internode_recv_buf is not None
            if node_step < self.topo.n_nodes - 1:
                # Relay the inter-node chunk one more hop along the node ring.
                self._launch_inter_backward(self._internode_recv_buf, step)
            # At a node boundary the intra-node ring restarts from the chunk
            # that just arrived over the inter-node link.
            self._recv_buf = self._internode_recv_buf

        self._launch_intra_backward(self._recv_buf, step)


class _LocalChunkProviderHybridBroadcast(_LocalChunkProviderHybrid):
    """Hybrid backward chunk provider: broadcast intra-node, P2P inter-node.

    The backward twin of :func:`hybrid_broadcast_forward`. Each rank keeps one
    ``_node_chunk`` per node round:

      * round 0  -> its own local x/kv
      * round k  -> the chunk that arrived over the inter-node P2P link

    Within a round of ``gpus_per_node`` steps the node-local rank ``step % gpn``
    broadcasts its ``_node_chunk`` to the node, so after a full round every node
    rank has seen every chunk resident on the node. There is no intra-node
    rotation, so ``_node_chunk`` is stable for the round and can be handed
    straight to the next inter-node hop.
    """

    def __init__(self, *args: object) -> None:
        """Resolve the intra-node broadcast group on top of the shared hybrid state."""
        super().__init__(*args)  # type: ignore[arg-type]
        self.intranode_group = self.topo.intranode_group
        self._node_chunk: Optional[torch.Tensor] = None

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the tensors for the requested ring step.

        Parameters
        ----------
        step : int
            Current ring step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Result returned by the operation.
        """
        if step == 0:
            kv = self._kv0()
            self._node_chunk = (
                self.x if self.config.send_in_forward == SendForward.X else kv
            )
            # Round 0's inter-node hop is launched immediately; it lands at
            # step == gpus_per_node. Clone so the broadcast below (which may
            # alias `_node_chunk`) can never race the in-flight send.
            self._launch_inter_backward(self._node_chunk.clone(), step)

        elif step % self._gpn == 0:
            # New node round: adopt the chunk that arrived from the previous node.
            assert self._internode_handle is not None
            self._internode_handle.wait()
            self._node_chunk = self._internode_recv_buf

        assert self._node_chunk is not None

        # Intra-node broadcast. The source rank reuses its own buffer in place;
        # everyone else receives into scratch.
        src = self.topo.rank_at(self.topo.node_idx, step)
        is_src = self.topo.local_idx == step % self._gpn
        self._recv_buf = (
            self._node_chunk if is_src else torch.empty_like(self._node_chunk)
        )
        dist.broadcast(self._recv_buf, src=src, group=self.intranode_group)

        return self._kv_from(self._recv_buf)

    def maybe_forward(self, step: int, ring_group_size: int) -> None:
        """Schedule the next inter-node hop at a node-round boundary.

        Intra-node movement is handled entirely by the broadcast in ``get``;
        the only thing left is the next inter-node hop, issued a full round
        ahead so it overlaps the ``gpus_per_node`` local broadcasts.

        Parameters
        ----------
        step : int
            Current ring step.
        ring_group_size : int
            Number of ranks in the ring group.
        """
        if step == 0 or step >= ring_group_size - 1:
            return
        if step % self._gpn != 0:
            return
        if step // self._gpn >= self.topo.n_nodes - 1:
            return

        assert self._node_chunk is not None
        # `_node_chunk` is read-only for the rest of the round (broadcast source
        # or idle), so it can be sent without a copy.
        self._launch_inter_backward(self._node_chunk, step)
