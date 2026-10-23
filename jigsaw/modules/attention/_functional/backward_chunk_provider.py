"""Chunk providers for ring-attention backward passes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import torch
import torch.distributed as dist

from jigsaw._core.comm import get_empty
from jigsaw._core.offload import NVLinkActivationOffloader
from jigsaw.modules.attention._config import (
    RingAttentionConfig,
    SaveBackward,
    SaveScope,
    SendForward,
    SendOp,
)

from ._comm_overlap import PrefetchBuffer, consume, issue_on_comm_stream
from .ring import _compute_kv, get_hybrid_topology


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
            if ring_config.overlap_active:
                return _LocalChunkProviderOverlap(*args[:-1])
            return _LocalChunkProvider(*args[:-1])
        case SendOp.BROADCAST:
            return _LocalChunkProviderBroadcast(*args)
        case SendOp.HYBRID_P2P | SendOp.HYBRID_ALLTOALL:
            if ring_config.overlap_active:
                return _LocalChunkProviderHybridP2POverlap(*args)
            return _LocalChunkProviderHybridP2P(*args)
        case SendOp.HYBRID_BROADCAST:
            if ring_config.overlap_active:
                return _LocalChunkProviderHybridBroadcastOverlap(*args)
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

        self._cur_gpu: torch.Tensor | None = None
        self._cur_event: torch.cuda.Event | None = None
        self._next_gpu: torch.Tensor | None = None
        self._next_event: torch.cuda.Event | None = None

        if ring_config.offloader is not None:
            self._cur_gpu, self._cur_event = self._start_prefetch(0)
            self._next_gpu, self._next_event = self._start_prefetch(1)

    def _start_prefetch(
        self, idx: int
    ) -> tuple[torch.Tensor | None, torch.cuda.Event | None]:
        """Start prefetching a chunk.

        Parameters
        ----------
        idx : int
            Chunk index.

        Returns
        -------
        tuple[torch.Tensor | None, torch.cuda.Event | None]
            Prefetched chunk and its completion event, or ``(None, None)``
            without an offloader or past the last chunk.
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
                chunk,
                self.w_qkv,
                self.b,
                self.seq_local,
                self.h,
                self.d,
                self.config.core_layout,
            )
            return kj.contiguous(), vj.contiguous(), kv
        else:
            raise NotImplementedError


class _LocalChunkProvider(_BackwardChunkProvider):
    """Provider re-circulating chunks around the ring with one send and receive per step.

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
        self._handle: dist.Work | None = None

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
            chunk,
            self.w_qkv,
            self.b,
            self.seq_local,
            self.h,
            self.d,
            self.config.core_layout,
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
                recv_buf,
                self.w_qkv,
                self.b,
                self.seq_local,
                self.h,
                self.d,
                self.config.core_layout,
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
                    self.x,
                    self.w_qkv,
                    self.b,
                    self.seq_local,
                    self.h,
                    self.d,
                    self.config.core_layout,
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
        """Send the current chunk to the previous rank, except at the first and last step.

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


class _LocalChunkProviderOverlap(_LocalChunkProvider):
    """:class:`_LocalChunkProvider` with the K/V relay on the comm stream.

    Alternates between two persistent receive buffers. With overlap,
    :func:`~.backward.ring_backward` calls :meth:`maybe_forward` before the
    step's attention core, so the next chunk's transfer runs under the core.
    Used only when :attr:`RingAttentionConfig.overlap_active`.
    """

    def __init__(self, *args: object) -> None:
        """Defer buffer sizing to step 0 (needs the K/V shape)."""
        super().__init__(*args)  # type: ignore[arg-type]
        self._cs = self.config.comm_stream
        self._bufs: list[torch.Tensor] | None = None
        self._pending = 0
        self._work: object = None
        self._event: torch.cuda.Event | None = None

    def _kick(self, src: torch.Tensor, into: int, step: int) -> None:
        """Issue a ``direction=-1`` ring send of ``src`` into ``self._bufs[into]``."""
        assert self._bufs is not None
        assert self._cs is not None
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
        """Return chunk ``step``: local on step 0, else adopted off the comm stream."""
        if step == 0:
            kv = self._kv0()
            send_buf = self.x if self.config.send_in_forward == SendForward.X else kv
            self._bufs = [torch.empty_like(send_buf), torch.empty_like(send_buf)]
            self._kick(send_buf, into=0, step=step)
            if self.config.send_in_forward == SendForward.X:
                kv, kj, vj = _compute_kv(
                    self.x,
                    self.w_qkv,
                    self.b,
                    self.seq_local,
                    self.h,
                    self.d,
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

        Step 0's relay is issued in :meth:`get`; the last step has nothing to
        relay.
        """
        if step == 0 or step >= ring_group_size - 1:
            return
        assert self._bufs is not None
        self._kick(self._bufs[self._pending], into=1 - self._pending, step=step)


class _LocalChunkProviderBroadcast(_LocalChunkProvider):
    """Provider in which rank ``s`` broadcasts its chunk at step ``s``."""

    def __init__(self, *args: object) -> None:
        """Allocate the receive buffer and record the ring position."""
        *base_args, local_rank = args
        super().__init__(*base_args)  # type: ignore[arg-type]
        self._recv_buf = torch.empty_like(
            self.x if self.config.send_in_forward == SendForward.X else self.chunk0
        )
        self._handle: dist.Work | None = None
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

        # Rank ``step`` broadcasts its own buffer; the others receive into scratch.
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
    """Base class of the hybrid providers (intra group plus outer ring)."""

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
        # Allocated on `x.device`, not the template's: under CPU offload
        # `chunks_to_save[0]` lives on the host, but this buffer seeds the
        # Overlap subclasses' `PrefetchBuffer`s, which NCCL receives into.
        # `x` is never offloaded.
        template = (
            x if ring_config.send_in_forward == SendForward.X else chunks_to_save[0]
        )
        self._recv_buf = torch.empty(
            template.shape, dtype=template.dtype, device=x.device
        )
        self._handle: dist.Work | None = None
        self.local_rank = int(local_rank)
        self.topo = get_hybrid_topology(group, ring_config.intra_size)
        self._intra_size = self.topo.intra_size
        # Outer ring: send to the next intra group and receive from the
        # previous (opposite the forward), addressed inside `inter_group`
        # so the intra group's communicator can progress alongside.
        self._inter_group = self.topo.inter_group
        self._inter_ranks = [
            self.topo.inter_group_peer(1),
            self.topo.inter_group_peer(-1),
        ]
        self._inter_handle: dist.Work | None = None
        self._inter_recv_buf: torch.Tensor | None = None

    def _launch_inter_backward(self, data: torch.Tensor, step: int) -> None:
        """Start the send of ``data`` to the next intra group around the ring.

        Parameters
        ----------
        data : torch.Tensor
            Tensor to send.
        step : int
            Current ring step.
        """
        buf = torch.empty_like(data)
        self._inter_recv_buf, self._inter_handle = self.config.p2p_fn(
            data=data,
            buffer=buf,
            send_recv_ranks=self._inter_ranks,
            group=self._inter_group,
            async_op=True,
            tag=step % self._intra_size,
        )


class _LocalChunkProviderHybridP2P(_LocalChunkProviderHybrid):
    """Hybrid provider rotating chunks within the intra group (P2P on the outer ring)."""

    def __init__(self, *args: object) -> None:
        """Resolve the intra-group peers."""
        super().__init__(*args)  # type: ignore[arg-type]
        # Intra-group ring: send to the previous local rank, receive from the
        # next, addressed inside `intra_group`.
        self._intra_group = self.topo.intra_group
        self._intra_ranks = [
            self.topo.intra_group_peer(-1),
            self.topo.intra_group_peer(1),
        ]

    def _launch_intra_backward(self, src: torch.Tensor, step: int) -> None:
        """Start the intra-group send of ``src``."""
        self._recv_buf, self._handle = self.config.send_op_fn(
            src,
            torch.empty_like(src),
            send_recv_ranks=self._intra_ranks,
            group=self._intra_group,
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

            # Both sends read the local chunk in place: the backward never
            # writes it, and it stays referenced for the whole pass.
            send_buf = self.x if self.config.send_in_forward == SendForward.X else kv
            self._launch_inter_backward(send_buf, step)
            self._launch_intra_backward(send_buf, step)

            if self.config.send_in_forward == SendForward.X:
                # Local Q always attends its own local K/V on step 0.
                kv, kj, vj = _compute_kv(
                    self.x,
                    self.w_qkv,
                    self.b,
                    self.seq_local,
                    self.h,
                    self.d,
                    self.config.core_layout,
                )
            else:
                kj, vj = kv[0].contiguous(), kv[1].contiguous()
            return kj, vj, kv

        # The first step of a round receives from the previous intra group;
        # every other step from the intra-group ring.
        if step % self._intra_size == 0:
            assert self._inter_handle is not None
            assert self._inter_recv_buf is not None
            self._inter_handle.wait()
            recv_buf = self._inter_recv_buf
        else:
            assert self._handle is not None
            self._handle.wait()
            recv_buf = self._recv_buf

        return self._kv_from(recv_buf)

    def maybe_forward(self, step: int, ring_group_size: int) -> None:
        """Relay the current chunk within the intra group and around the outer ring.

        Parameters
        ----------
        step : int
            Current ring step.
        ring_group_size : int
            Number of ranks in the ring.
        """
        if step == 0 or step >= ring_group_size - 1:
            return

        group_step, inner = divmod(step, self._intra_size)
        if inner == 0:
            assert self._inter_recv_buf is not None
            # Capture before `_launch_inter_backward` rebinds
            # `_inter_recv_buf` to the next round's empty buffer.
            arrived = self._inter_recv_buf
            if group_step < self.topo.n_groups - 1:
                self._launch_inter_backward(arrived, step)
            # The intra-group ring restarts from the chunk that just arrived.
            self._recv_buf = arrived

        if inner + 1 < self._intra_size:
            # Skip the round's last step: the next `get` reads only the outer
            # ring. Sends are matched in post order (tags are unused), so an
            # unwaited send here would shift every later intra-group receive by
            # one step.
            self._launch_intra_backward(self._recv_buf, step)


class _LocalChunkProviderHybridP2POverlap(_LocalChunkProviderHybridP2P):
    """:class:`_LocalChunkProviderHybridP2P` with both sends on the comm stream.

    Prefetches both transfers on the comm stream: the send within the intra
    group and the send to the next intra group around the ring. Each is started
    in :meth:`maybe_forward`, before the step's attention core runs, and waited
    on in the :meth:`get` that needs its chunk. The arrived chunk is kept in
    ``_raw_recv_buf`` for :meth:`maybe_forward` to relay.
    """

    def __init__(self, *args: object) -> None:
        """Allocate the intra-group and outer-ring prefetch buffers."""
        super().__init__(*args)  # type: ignore[arg-type]
        cs = self.config.comm_stream
        assert cs is not None
        self._intra_buf = PrefetchBuffer(cs, self._recv_buf)
        self._inter_buf = PrefetchBuffer(cs, self._recv_buf)
        self._raw_recv_buf: torch.Tensor | None = None
        self._relay_src: torch.Tensor | None = None

    def _launch_inter_backward(self, data: torch.Tensor, step: int) -> None:
        """Prefetch this round's outer-ring send on the comm stream."""

        def produce(buf: torch.Tensor) -> tuple[torch.Tensor, dist.Work | None]:
            return cast(
                "tuple[torch.Tensor, dist.Work | None]",
                self.config.p2p_fn(
                    data=data,
                    buffer=buf,
                    send_recv_ranks=self._inter_ranks,
                    group=self._inter_group,
                    async_op=True,
                    tag=step % self._intra_size,
                ),
            )

        self._inter_buf.kick(produce, extra_keep_alive=(data,))

    def _launch_intra_backward(self, src: torch.Tensor, step: int) -> None:
        """Prefetch this step's intra-group send on the comm stream."""

        def produce(buf: torch.Tensor) -> tuple[torch.Tensor, dist.Work | None]:
            return cast(
                "tuple[torch.Tensor, dist.Work | None]",
                self.config.send_op_fn(
                    src,
                    buf,
                    send_recv_ranks=self._intra_ranks,
                    group=self._intra_group,
                    async_op=True,
                    step=step,
                ),
            )

        self._intra_buf.kick(produce, extra_keep_alive=(src,))

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the tensors for the requested ring step."""
        if step == 0:
            kv = self._kv0()
            send_buf = self.x if self.config.send_in_forward == SendForward.X else kv
            self._launch_inter_backward(send_buf, step)
            self._launch_intra_backward(send_buf, step)

            if self.config.send_in_forward == SendForward.X:
                kv, kj, vj = _compute_kv(
                    self.x,
                    self.w_qkv,
                    self.b,
                    self.seq_local,
                    self.h,
                    self.d,
                    self.config.core_layout,
                )
            else:
                kj, vj = kv[0].contiguous(), kv[1].contiguous()
            return kj, vj, kv

        # The first step of a round receives from the previous intra group;
        # every other step from the intra-group ring.
        if step % self._intra_size == 0:
            self._raw_recv_buf = self._inter_buf.wait()
        else:
            self._raw_recv_buf = self._intra_buf.wait()
        return self._kv_from(self._raw_recv_buf)

    def maybe_forward(self, step: int, ring_group_size: int) -> None:
        """Prefetch the relays of this step's chunk (intra group and outer ring)."""
        if step == 0 or step >= ring_group_size - 1:
            return

        group_step, inner = divmod(step, self._intra_size)
        if inner == 0:
            assert self._raw_recv_buf is not None
            # Captured as in the base class.
            arrived = self._raw_recv_buf
            if group_step < self.topo.n_groups - 1:
                self._launch_inter_backward(arrived, step)
            self._relay_src = arrived

        if inner + 1 < self._intra_size:
            # Skip the round's last step, as in the base class.
            src = self._relay_src if inner == 0 else self._raw_recv_buf
            assert src is not None
            self._launch_intra_backward(src, step)


class _LocalChunkProviderHybridBroadcast(_LocalChunkProviderHybrid):
    """Hybrid provider with broadcasts within the intra group."""

    def __init__(self, *args: object) -> None:
        """Resolve the intra-group broadcast group."""
        super().__init__(*args)  # type: ignore[arg-type]
        self.intra_group = self.topo.intra_group
        self._group_chunk: torch.Tensor | None = None

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
            self._group_chunk = (
                self.x if self.config.send_in_forward == SendForward.X else kv
            )
            # Round 0's outer-ring send, waited on at step == intra_size.
            # `_group_chunk` is only read during the round, so it is sent in place.
            self._launch_inter_backward(self._group_chunk, step)

        elif step % self._intra_size == 0:
            # New round: take the chunk received from the previous intra group.
            assert self._inter_handle is not None
            self._inter_handle.wait()
            self._group_chunk = self._inter_recv_buf

        assert self._group_chunk is not None

        # The source broadcasts its own buffer in place; the others receive
        # into scratch. `step % intra_size` is the source's rank inside
        # `intra_group`: hence `group_src`, not the global `src`, which a
        # ring that is a mesh sub-group would get wrong.
        phase = step % self._intra_size
        is_src = self.topo.local_idx == phase
        self._recv_buf = (
            self._group_chunk if is_src else torch.empty_like(self._group_chunk)
        )
        dist.broadcast(self._recv_buf, group_src=phase, group=self.intra_group)

        return self._kv_from(self._recv_buf)

    def maybe_forward(self, step: int, ring_group_size: int) -> None:
        """Start the next outer-ring send at the first step of a round.

        Parameters
        ----------
        step : int
            Current ring step.
        ring_group_size : int
            Number of ranks in the ring.
        """
        if step == 0 or step >= ring_group_size - 1:
            return
        if step % self._intra_size != 0:
            return
        if step // self._intra_size >= self.topo.n_groups - 1:
            return

        assert self._group_chunk is not None
        # `_group_chunk` is read-only for the rest of the round (broadcast source
        # or idle), so it can be sent without a copy.
        self._launch_inter_backward(self._group_chunk, step)


class _LocalChunkProviderHybridBroadcastOverlap(_LocalChunkProviderHybridBroadcast):
    """:class:`_LocalChunkProviderHybridBroadcast` with its transfers on the comm stream.

    Prefetches both transfers on the comm stream: the broadcasts within the
    intra group and the send to the next intra group around the ring. A
    round's broadcasts depend only on ``_group_chunk``, so each ``get`` waits on
    the broadcast started by the previous call and starts the next one. A
    round's first broadcast is started and waited on in the same call.
    """

    def __init__(self, *args: object) -> None:
        """Allocate the intra-group and outer-ring prefetch buffers."""
        super().__init__(*args)  # type: ignore[arg-type]
        cs = self.config.comm_stream
        assert cs is not None
        self._bc_buf = PrefetchBuffer(cs, self._recv_buf)
        self._inter_buf = PrefetchBuffer(cs, self._recv_buf)

    def _launch_inter_backward(self, data: torch.Tensor, step: int) -> None:
        """Prefetch this round's outer-ring send on the comm stream."""

        def produce(buf: torch.Tensor) -> tuple[torch.Tensor, dist.Work | None]:
            return cast(
                "tuple[torch.Tensor, dist.Work | None]",
                self.config.p2p_fn(
                    data=data,
                    buffer=buf,
                    send_recv_ranks=self._inter_ranks,
                    group=self._inter_group,
                    async_op=True,
                    tag=step % self._intra_size,
                ),
            )

        self._inter_buf.kick(produce, extra_keep_alive=(data,))

    def _kick_broadcast(self, phase: int) -> None:
        """Prefetch this round's intra-group broadcast from local index ``phase``."""
        is_src = self.topo.local_idx == phase
        group_chunk = self._group_chunk
        assert group_chunk is not None

        def produce(buf: torch.Tensor) -> tuple[torch.Tensor, dist.Work | None]:
            if is_src:
                # Broadcast reads the source's own tensor in place; the
                # persistent buffer has to hold that content before the
                # collective fires.
                buf.copy_(group_chunk)
            work = dist.broadcast(
                buf, group_src=phase, group=self.intra_group, async_op=True
            )
            return buf, work

        self._bc_buf.kick(produce, extra_keep_alive=(group_chunk,) if is_src else ())

    def get(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the tensors for the requested ring step."""
        if step == 0:
            kv = self._kv0()
            self._group_chunk = (
                self.x if self.config.send_in_forward == SendForward.X else kv
            )
            self._launch_inter_backward(self._group_chunk, step)
            self._kick_broadcast(0)
        elif step % self._intra_size == 0:
            self._group_chunk = self._inter_buf.wait()
            self._kick_broadcast(0)

        phase = step % self._intra_size
        recv_buf = self._bc_buf.wait()
        if phase + 1 < self._intra_size:
            self._kick_broadcast(phase + 1)

        self._recv_buf = recv_buf
        return self._kv_from(recv_buf)

    def maybe_forward(self, step: int, ring_group_size: int) -> None:
        """Prefetch the next outer-ring send at the first step of a round."""
        if step == 0 or step >= ring_group_size - 1:
            return
        if step % self._intra_size != 0:
            return
        if step // self._intra_size >= self.topo.n_groups - 1:
            return

        assert self._group_chunk is not None
        self._launch_inter_backward(self._group_chunk, step)
