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
    """Build the backward chunk provider of ``ring_config.send_op``.

    Parameters
    ----------
    ring_config : RingAttentionConfig
        Ring-attention configuration.
    chunks_to_save : tuple[torch.Tensor, ...]
        Chunks saved in the forward pass.
    x : torch.Tensor
        Local input chunk.
    w_qkv : torch.Tensor
        Fused query, key and value projection weights.
    b : int
        Batch size.
    seq_local : int
        Local sequence length.
    h : int
        Number of attention heads.
    d : int
        Per-head embedding dimension.
    group : dist.ProcessGroup
        Process group of the ring.
    local_rank : int
        Rank within the ring group.

    Returns
    -------
    _BackwardChunkProvider
        Chunk provider.
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
    """Base class yielding the key and value chunk of each backward ring step."""

    @abstractmethod
    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the key, value and fused key-value tensors of ``step``."""

    def maybe_forward(self, step: int, ring_group_size: int) -> None:
        """Start the communication of the next ring step.

        Does nothing by default.

        Parameters
        ----------
        step : int
            Current ring step.
        ring_group_size : int
            Number of ranks in the ring.
        """
        return


class _GlobalChunkProvider(_BackwardChunkProvider):
    """Provider reading chunks saved in the forward pass, prefetching offloaded ones."""

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
        """Initialize the provider.

        Parameters
        ----------
        ring_config : RingAttentionConfig
            Ring-attention configuration.
        chunks_to_save : tuple[torch.Tensor, ...]
            Chunks saved in the forward pass.
        w_qkv : torch.Tensor
            Fused query, key and value projection weights.
        b : int
            Batch size.
        seq_local : int
            Local sequence length.
        h : int
            Number of attention heads.
        d : int
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
        """Start prefetching a chunk.

        Parameters
        ----------
        idx : int
            Chunk index.

        Returns
        -------
        tuple[torch.Tensor, torch.cuda.Event | None]
            Prefetched chunk and its completion event.
        """
        if self.config.offloader is None or idx >= len(self.chunks):
            return None, None
        return self.config.offloader.prefetch(self.chunks[idx])

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the chunk of ``step`` and start prefetching the next one.

        Parameters
        ----------
        step : int
            Current ring step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Contiguous ``kj``, ``vj`` and the underlying ``kv`` buffer.
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
                chunk, self.w_qkv, self.b, self.seq_local, self.h, self.d
            )
            return kj.contiguous(), vj.contiguous(), kv
        else:
            raise NotImplementedError


class _LocalChunkProvider(_BackwardChunkProvider):
    """Provider re-circulating chunks around the ring with one P2P send and receive per step.

    Only the local chunk was saved in the forward pass.
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
        """Initialize the provider.

        Parameters
        ----------
        ring_config : RingAttentionConfig
            Ring-attention configuration.
        chunks_to_save : tuple[torch.Tensor, ...]
            Chunks saved in the forward pass.
        x : torch.Tensor
            Local input chunk.
        w_qkv : torch.Tensor
            Fused query, key and value projection weights.
        b : int
            Batch size.
        seq_local : int
            Local sequence length.
        h : int
            Number of local heads.
        d : int
            Embedding dimension per head.
        group : dist.ProcessGroup
            Process group of the ring.
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
        """Return the saved local chunk on the device.

        Returns
        -------
        torch.Tensor
            Local chunk.
        """
        chunk = self.chunk0
        if isinstance(self.config.offloader, NVLinkActivationOffloader):
            gpu, ev = self.config.offloader.prefetch(chunk)
            self.config.offloader.wait_prefetch(ev)
            chunk = gpu
        return chunk

    def _kv0(self) -> torch.Tensor:
        """Return the fused key-value tensor of the local chunk.

        Returns
        -------
        torch.Tensor
            Saved ``kv``, or ``kv`` recomputed from the saved ``x``.
        """
        chunk = self._materialize_chunk0()
        if self.config.save_for_backward == SaveBackward.KV:
            return chunk
        kv, _, _ = _compute_kv(
            chunk, self.w_qkv, self.b, self.seq_local, self.h, self.d
        )
        return kv

    def _kv_from(
        self, recv_buf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack a received ring buffer into ``kj``, ``vj`` and ``kv``.

        Parameters
        ----------
        recv_buf : torch.Tensor
            Received buffer, holding ``x`` or ``kv`` according to
            ``send_in_forward``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Contiguous ``kj``, ``vj`` and the underlying ``kv`` buffer.
        """
        if self.config.send_in_forward == SendForward.X:
            kv, kj, vj = _compute_kv(
                recv_buf, self.w_qkv, self.b, self.seq_local, self.h, self.d
            )
            return kj.contiguous(), vj.contiguous(), kv
        kv = recv_buf
        return kv[0].contiguous(), kv[1].contiguous(), kv

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the key, value and fused key-value tensors of a ring step.

        Parameters
        ----------
        step : int
            Current ring step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Contiguous ``kj``, ``vj`` and the underlying ``kv`` buffer.
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
                    self.x, self.w_qkv, self.b, self.seq_local, self.h, self.d
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
        """Send the current chunk to the next peer, except at the first and last step.

        Parameters
        ----------
        step : int
            Current ring step.
        ring_group_size : int
            Number of ranks in the ring.
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


class _LocalChunkProviderBroadcast(_LocalChunkProvider):
    """Provider in which rank ``s`` broadcasts its chunk at step ``s``."""

    def __init__(self, *args: object) -> None:
        """Allocate the receive buffer and record the ring position."""
        *base_args, local_rank = args
        super().__init__(*base_args)  # type: ignore[arg-type]
        self._recv_buf = torch.empty_like(
            self.x if self.config.send_in_forward == SendForward.X else self.chunk0
        )
        self._handle: Optional[dist.Work] = None
        self.local_rank = local_rank

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the key, value and fused key-value tensors of a ring step.

        Parameters
        ----------
        step : int
            Current ring step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Contiguous ``kj``, ``vj`` and the underlying ``kv`` buffer.
        """
        kv = self._kv0()

        # Kick off the first ring send of whichever buffer is configured.
        send_buf = self.x if self.config.send_in_forward == SendForward.X else kv
        self._recv_buf = (
            send_buf if self.local_rank == step else torch.empty_like(send_buf)
        )
        self._handle = dist.broadcast(
            self._recv_buf, group_src=step, group=self.group, async_op=True
        )

        assert self._handle is not None
        self._handle.wait()
        return self._kv_from(self._recv_buf)

    def maybe_forward(self, step: int, ring_group_size: int) -> None:
        """Do nothing, since each broadcast is issued in ``get``.

        Parameters
        ----------
        step : int
            Current ring step.
        ring_group_size : int
            Number of ranks in the ring.
        """
        return


class _LocalChunkProviderHybrid(_LocalChunkProvider):
    """Base class of the hybrid intra-node and inter-node providers."""

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
            Ring-attention configuration.
        chunks_to_save : tuple[torch.Tensor, ...]
            Chunks saved in the forward pass.
        x : torch.Tensor
            Local input chunk.
        w_qkv : torch.Tensor
            Fused query, key and value projection weights.
        b : int
            Batch size.
        seq_local : int
            Local sequence length.
        h : int
            Number of attention heads.
        d : int
            Embedding dimension per head.
        group : dist.ProcessGroup
            Process group of the ring.
        local_rank : int
            Rank within the ring group.
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
        """Start an inter-node transfer of ``data``.

        Parameters
        ----------
        data : torch.Tensor
            Tensor to send.
        step : int
            Current ring step.
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
    """Hybrid provider with P2P intra-node rotation."""

    def __init__(self, *args: object) -> None:
        """Resolve the intra-node peers."""
        super().__init__(*args)  # type: ignore[arg-type]
        # Intra-node ring: send to the previous local rank, receive from the
        # next -- mirroring the forward.
        self._intra_ranks = [self.topo.intra_peer(-1), self.topo.intra_peer(1)]

    def _launch_intra_backward(self, src: torch.Tensor, step: int) -> None:
        """Start the intra-node send of ``src``."""
        self._recv_buf, self._handle = self.config.send_op_fn(
            src,
            torch.empty_like(src),
            send_recv_ranks=self._intra_ranks,
            group=self.group,
            async_op=True,
            step=step,
        )

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the key, value and fused key-value tensors of a ring step.

        Parameters
        ----------
        step : int
            Current ring step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Contiguous ``kj``, ``vj`` and the underlying ``kv`` buffer.
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
                    self.x, self.w_qkv, self.b, self.seq_local, self.h, self.d
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
        """Relay the current chunk to the next intra-node and inter-node peers.

        Parameters
        ----------
        step : int
            Current ring step.
        ring_group_size : int
            Number of ranks in the ring.
        """
        if step == 0 or step >= ring_group_size - 1:
            return

        node_step = step // self._gpn
        if step % self._gpn == 0:
            assert self._internode_recv_buf is not None
            # Launching the relay replaces ``_internode_recv_buf`` with the (still
            # empty) buffer of the next hop, so keep the chunk that just arrived.
            arrived = self._internode_recv_buf
            if node_step < self.topo.n_nodes - 1:
                # Relay the inter-node chunk one more hop along the node ring.
                self._launch_inter_backward(arrived, step)
            # At a node boundary the intra-node ring restarts from the chunk
            # that just arrived over the inter-node link.
            self._recv_buf = arrived

        if (step + 1) % self._gpn == 0:
            # The next chunk arrives over the inter-node link, so an intra-node
            # send would never be waited on -- and overwriting its handle and
            # receive buffer while it is in flight hangs Gloo.
            return

        self._launch_intra_backward(self._recv_buf, step)


class _LocalChunkProviderHybridBroadcast(_LocalChunkProviderHybrid):
    """Hybrid provider with intra-node broadcasts."""

    def __init__(self, *args: object) -> None:
        """Resolve the intra-node broadcast group."""
        super().__init__(*args)  # type: ignore[arg-type]
        self.intranode_group = self.topo.intranode_group
        self._node_chunk: Optional[torch.Tensor] = None

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the key, value and fused key-value tensors of a ring step.

        Parameters
        ----------
        step : int
            Current ring step.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Contiguous ``kj``, ``vj`` and the underlying ``kv`` buffer.
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
        src = step % self._gpn
        is_src = self.topo.local_idx == src
        self._recv_buf = (
            self._node_chunk if is_src else torch.empty_like(self._node_chunk)
        )
        dist.broadcast(self._recv_buf, group_src=src, group=self.intranode_group)

        return self._kv_from(self._recv_buf)

    def maybe_forward(self, step: int, ring_group_size: int) -> None:
        """Start the next inter-node hop at a node-round boundary.

        Parameters
        ----------
        step : int
            Current ring step.
        ring_group_size : int
            Number of ranks in the ring.
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
