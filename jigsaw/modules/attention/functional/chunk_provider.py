"""Chunk providers for ring-attention backward passes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, cast

import torch
import torch.distributed as dist

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
    empty: torch.Tensor,
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
    empty: torch.Tensor
        Empty tensor for computation reasons.
    local_rank: int
        Rank within the distributed attention group.

    Returns
    -------
    _BackwardChunkProvider
    """
    match ring_config.save_scope:
        case SaveScope.GLOBAL:
            return _GlobalChunkProvider(
                ring_config, chunks_to_save, w_qkv, b, seq_local, h, d
            )
        case SaveScope.LOCAL:
            match ring_config.send_op:
                case SendOp.P2P | SendOp.ALLTOALL:
                    return _LocalChunkProvider(
                        ring_config,
                        chunks_to_save,
                        x,
                        w_qkv,
                        b,
                        seq_local,
                        h,
                        d,
                        group,
                        empty,
                    )
                case SendOp.BROADCAST:
                    return _LocalChunkProviderBroadcast(
                        ring_config,
                        chunks_to_save,
                        x,
                        w_qkv,
                        b,
                        seq_local,
                        h,
                        d,
                        group,
                        empty,
                        local_rank,
                    )
                case SendOp.HYBRID_P2P:
                    return _LocalChunkProviderHybridP2P(
                        ring_config,
                        chunks_to_save,
                        x,
                        w_qkv,
                        b,
                        seq_local,
                        h,
                        d,
                        group,
                        empty,
                        local_rank,
                    )
                case SendOp.HYBRID_BROADCAST:
                    return _LocalChunkProviderHybridBroadcast(
                        ring_config,
                        chunks_to_save,
                        x,
                        w_qkv,
                        b,
                        seq_local,
                        h,
                        d,
                        group,
                        empty,
                        local_rank,
                    )
        case _:
            raise NotImplementedError
    raise RuntimeError("Unable to build a backward chunk provider")


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
            print(f"R{dist.get_rank()} step {step} chunk {chunk.sum()}")
            kv, kj, vj = _compute_kv(
                chunk, self.w_qkv, self.b, self.seq_local, self.h, self.d
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
        empty: torch.Tensor,
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
        empty: torch.Tensor
            Empty tensor for computational reasons.
        """
        self.config = ring_config
        self.chunk0 = chunks_to_save[0]
        self.x = x
        self.w_qkv = w_qkv
        self.b, self.seq_local, self.h, self.d = b, seq_local, h, d
        self.group = group
        self.empty = empty

        self._recv_buf = empty
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
            chunk = self._materialize_chunk0()
            if self.config.save_for_backward == SaveBackward.KV:
                kv = chunk
            else:
                kv, _, _ = _compute_kv(
                    chunk, self.w_qkv, self.b, self.seq_local, self.h, self.d
                )

            # Kick off the first ring send of whichever buffer is configured.
            send_buf = self.x if self.config.send_in_forward == SendForward.X else kv
            self._recv_buf = torch.empty_like(send_buf)
            self._recv_buf, self._handle = self.config.send_op_fn(
                send_buf,
                self._recv_buf,
                direction=-1,
                group=self.group,
                async_op=True,
                empty=self.empty,
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

        # step > 0: receive the chunk that's been circulating, then forward
        # it on (unless this is the last step).
        assert self._handle is not None
        self._handle.wait()
        recv_buf = self._recv_buf
        if self.config.send_in_forward == SendForward.X:
            kv, kj, vj = _compute_kv(
                recv_buf, self.w_qkv, self.b, self.seq_local, self.h, self.d
            )
            kj, vj = kj.contiguous(), vj.contiguous()
        else:
            kv = recv_buf
            kj, vj = kv[0].contiguous(), kv[1].contiguous()

        # `world_size` isn't tracked locally; ring_backward only calls us
        # `world_size` times total, so re-sending on the final step is
        # harmless but wasteful -- guarded via `has_more` below instead.
        return kj, vj, kv

    def maybe_forward(self, step: int, world_size: int) -> None:
        """
        Send chunk to next partner rank if necessary.

        Chunk does not get sent if it is the first (treated differently) or final (done) step.

        Parameters
        ----------
        step: int
            stage of the ring attention.
        world_size: int
            Size of the domain-parallel group.
        """
        if step == 0 or step >= world_size - 1:
            return
        next_recv = torch.empty_like(self._recv_buf)
        self._recv_buf, self._handle = self.config.send_op_fn(
            self._recv_buf,
            next_recv,
            direction=-1,
            group=self.group,
            async_op=True,
            empty=self.empty,
            step=step,
        )


class _LocalChunkProviderBroadcast(_LocalChunkProvider):
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
        empty: torch.Tensor,
        local_rank: int,
    ) -> None:
        """Initialize the object.

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
        empty : torch.Tensor
            Empty tensor used by communication helpers.
        local_rank : int
            Rank of the current process.

        """
        super().__init__(
            ring_config, chunks_to_save, x, w_qkv, b, seq_local, h, d, group, empty
        )
        self._recv_buf = torch.empty_like(
            x if ring_config.send_in_forward == SendForward.X else chunks_to_save[0]
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
        chunk = self._materialize_chunk0()
        if self.config.save_for_backward == SaveBackward.KV:
            kv = chunk
        else:
            kv, _, _ = _compute_kv(
                chunk, self.w_qkv, self.b, self.seq_local, self.h, self.d
            )

        # Kick off the first ring send of whichever buffer is configured.
        send_buf = self.x if self.config.send_in_forward == SendForward.X else kv
        self._recv_buf = (
            send_buf if self.local_rank == step else torch.empty_like(send_buf)
        )
        self._handle = dist.broadcast(
            self._recv_buf, src=step, group=self.group, async_op=True
        )

        # step > 0: receive the chunk that's been circulating, then forward
        # it on (unless this is the last step).
        assert self._handle is not None
        self._handle.wait()
        recv_buf = self._recv_buf

        if self.config.send_in_forward == SendForward.X:
            kv, kj, vj = _compute_kv(
                recv_buf, self.w_qkv, self.b, self.seq_local, self.h, self.d
            )
            kj, vj = kj.contiguous(), vj.contiguous()
        else:
            kv = recv_buf
            kj, vj = kv[0].contiguous(), kv[1].contiguous()

        # `world_size` isn't tracked locally; ring_backward only calls us
        # `world_size` times total, so re-sending on the final step is
        # harmless but wasteful -- guarded via `has_more` below instead.
        return kj, vj, kv

    def maybe_forward(self, step: int, world_size: int) -> None:
        """Schedule communication for the next ring step.

        Parameters
        ----------
        step : int
            Current ring step.
        world_size : int
            Number of processes in the distributed group.

        """
        return


class _LocalChunkProviderHybridP2P(_LocalChunkProvider):
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
        empty: torch.Tensor,
        local_rank: int,
    ) -> None:
        """Initialize the object.

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
        empty : torch.Tensor
            Empty tensor used by communication helpers.
        local_rank : int
            Rank of the current process.

        """
        super().__init__(
            ring_config, chunks_to_save, x, w_qkv, b, seq_local, h, d, group, empty
        )
        self._recv_buf = torch.empty_like(
            x if ring_config.send_in_forward == SendForward.X else chunks_to_save[0]
        )
        self._handle: Optional[dist.Work] = None
        self.local_rank = local_rank
        self.topo = get_hybrid_topology(group, ring_config.gpus_per_node)
        self._gpn = self.topo.gpus_per_node
        # Intra-node ring: send to the previous local rank, receive from the
        # next -- mirroring the forward. Inter-node hop reverses the node ring
        # relative to the forward: send to the next node, receive from the prev.
        self._intra_ranks = [self.topo.intra_peer(-1), self.topo.intra_peer(1)]
        self._inter_ranks = [self.topo.inter_peer(1), self.topo.inter_peer(-1)]
        self._internode_handle = None
        self._internode_recv_buf = None

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
            chunk = self._materialize_chunk0()
            if self.config.save_for_backward == SaveBackward.KV:
                kv = chunk
            else:
                kv, _, _ = _compute_kv(
                    chunk, self.w_qkv, self.b, self.seq_local, self.h, self.d
                )

            # Kick off the first ring send of whichever buffer is configured.
            internode_recv_buf = (
                torch.empty_like(self.x)
                if self.config.send_in_forward == SendForward.X
                else torch.empty_like(kv)
            )

            self._internode_recv_buf, self._internode_handle = self.config.p2p_fn(
                data=(
                    self.x.clone()
                    if self.config.send_in_forward == SendForward.X
                    else kv.clone()
                ),
                buffer=internode_recv_buf,
                send_recv_ranks=self._inter_ranks,
                group=self.group,
                async_op=True,
                tag=step % self._gpn,
                empty=self.empty,
            )

            send_buf = self.x if self.config.send_in_forward == SendForward.X else kv
            self._recv_buf = torch.empty_like(send_buf)

            # Launch send_op to the neighbouring ranks
            self._recv_buf, self._handle = self.config.send_op_fn(
                send_buf,
                self._recv_buf,
                send_recv_ranks=self._intra_ranks,
                group=self.group,
                async_op=True,
                empty=self.empty,
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

        # step > 0: receive the chunk that's been circulating, then forward
        # it on (unless this is the last step).

        if step % self._gpn == 0:  # if we are receiving from internode
            assert self._internode_handle is not None
            self._internode_handle.wait()
            recv_buf = self._internode_recv_buf
        else:
            assert self._handle is not None
            self._handle.wait()
            recv_buf = self._recv_buf

        if self.config.send_in_forward == SendForward.X:
            kv, kj, vj = _compute_kv(
                recv_buf, self.w_qkv, self.b, self.seq_local, self.h, self.d
            )
            kj, vj = kj.contiguous(), vj.contiguous()
        else:
            kv = recv_buf
            kj, vj = kv[0].contiguous(), kv[1].contiguous()

        # `world_size` isn't tracked locally; ring_backward only calls us
        # `world_size` times total, so re-sending on the final step is
        # harmless but wasteful -- guarded via `has_more` below instead.
        return kj, vj, kv

    def maybe_forward(self, step: int, world_size: int) -> None:
        """Schedule communication for the next ring step.

        Parameters
        ----------
        step : int
            Current ring step.
        world_size : int
            Number of processes in the distributed group.

        """
        if step == 0 or step >= world_size - 1:
            return

        node_step = step // self._gpn
        if step % self._gpn == 0 and node_step < self.topo.n_nodes - 1:
            # Relay the inter-node chunk one more hop along the node ring.
            internode_recv_buf = (
                torch.empty_like(self.x)
                if self.config.send_in_forward == SendForward.X
                else torch.empty_like(self._recv_buf)
            )
            self._internode_recv_buf, self._internode_handle = self.config.p2p_fn(
                data=self._internode_recv_buf,
                buffer=internode_recv_buf,
                send_recv_ranks=self._inter_ranks,
                group=self.group,
                async_op=True,
                tag=step % self._gpn,
                empty=self.empty,
            )
        # At a node boundary the intra-node ring restarts from the chunk that
        # just arrived over the inter-node link.
        if step % self._gpn == 0:
            assert self._internode_recv_buf is not None
            self._recv_buf = self._internode_recv_buf

        next_recv = torch.empty_like(self._recv_buf)
        self._recv_buf, self._handle = self.config.send_op_fn(
            self._recv_buf,
            next_recv,
            send_recv_ranks=self._intra_ranks,
            group=self.group,
            async_op=True,
            empty=self.empty,
            step=step,
        )


_INTRANODE_GROUPS: dict[tuple[int, int], dist.ProcessGroup] = {}


def _get_intranode_group(
    group: dist.ProcessGroup, gpus_per_node: int
) -> dist.ProcessGroup:
    """Return (and cache) the sub-group containing this rank's node.

    NOTE: `dist.new_group` is collective -- every rank must build every
    sub-group in the same order. Ranks passed to `new_group` are WORLD ranks,
    so this assumes `group` is WORLD (as the P2P path already does when it
    derives peers from `local_rank`).
    """
    key = (id(group), gpus_per_node)
    cached = _INTRANODE_GROUPS.get(key)
    if cached is not None:
        return cached

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    mine: Optional[dist.ProcessGroup] = None
    for start in range(0, world, gpus_per_node):
        ranks = list(range(start, min(start + gpus_per_node, world)))
        sub = dist.new_group(ranks=ranks)
        if rank in ranks:
            mine = sub
    if mine is None:
        raise RuntimeError("Current rank is not present in any intra-node group")
    _INTRANODE_GROUPS[key] = mine
    return mine


class _LocalChunkProviderHybridBroadcast(_LocalChunkProvider):
    """Hybrid ring provider: broadcast intra-node, P2P inter-node.

    Each rank owns exactly one buffer per "node round" (`_node_chunk`):
      * round 0  -> its own local x/kv
      * round k  -> the chunk received over the inter-node P2P link

    Within a round of `gpus_per_node` steps, the rank whose local index equals
    `step % gpus_per_node` broadcasts its `_node_chunk` to the node sub-group,
    so after a full round every rank has seen every chunk resident on the node.
    Unlike the P2P variant there is no intra-node rotation, so `_node_chunk`
    is stable for the whole round and can be handed straight to the inter-node
    send at the round boundary.
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
        empty: torch.Tensor,
        local_rank: int,
        gpus_per_node: int = 4,
        intranode_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        """Initialize the object.

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
        empty : torch.Tensor
            Empty tensor used by communication helpers.
        local_rank : int
            Rank of the current process.
        gpus_per_node : int
            Number of processes assigned to each node.
        intranode_group : Optional[dist.ProcessGroup]
            Process group containing ranks on the current node.

        """
        super().__init__(
            ring_config, chunks_to_save, x, w_qkv, b, seq_local, h, d, group, empty
        )
        self._recv_buf = torch.empty_like(
            x if ring_config.send_in_forward == SendForward.X else chunks_to_save[0]
        )
        self._handle: Optional[dist.Work] = None
        self.local_rank = int(local_rank)
        self.gpus_per_node = gpus_per_node
        self.intranode_group = intranode_group or _get_intranode_group(
            group, gpus_per_node
        )
        # First rank of this node; broadcast sources are node_base + local idx.
        self._node_base = (self.local_rank // gpus_per_node) * gpus_per_node

        self._node_chunk: Optional[torch.Tensor] = None
        self._internode_handle = None
        self._internode_recv_buf = None

    def _launch_internode(self, data: torch.Tensor, step: int) -> None:
        """Launch an inter-node tensor transfer.

        Parameters
        ----------
        data : torch.Tensor
            Tensor data to communicate.
        step : int
            Current ring step.

        """
        buf = torch.empty_like(data)
        self._internode_recv_buf, self._internode_handle = self.config.p2p_fn(
            data=data,
            buffer=buf,
            direction=self.gpus_per_node,
            group=self.group,
            async_op=True,
            tag=step % self.gpus_per_node,
            empty=self.empty,
            group_size=self.group.size(),
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
            chunk = self._materialize_chunk0()
            if self.config.save_for_backward == SaveBackward.KV:
                kv = chunk
            else:
                kv, _, _ = _compute_kv(
                    chunk, self.w_qkv, self.b, self.seq_local, self.h, self.d
                )
            self._node_chunk = (
                self.x if self.config.send_in_forward == SendForward.X else kv
            )
            # Round 0's inter-node hop is launched immediately; it lands at
            # step == gpus_per_node. Clone so the broadcast below (which may
            # alias `_node_chunk`) can never race the in-flight send.
            self._launch_internode(self._node_chunk.clone(), step)

        elif step % self.gpus_per_node == 0:
            # New node round: adopt the chunk that arrived from the previous node.
            assert self._internode_handle is not None
            self._internode_handle.wait()
            self._node_chunk = self._internode_recv_buf

        assert self._node_chunk is not None

        # Intra-node broadcast. The source rank reuses its own buffer in place
        # (contents unchanged); everyone else receives into scratch.
        src = self._node_base + (step % self.gpus_per_node)
        self._recv_buf = (
            self._node_chunk
            if self.local_rank == src
            else torch.empty_like(self._node_chunk)
        )
        self._handle = dist.broadcast(
            self._recv_buf, src=src, group=self.intranode_group, async_op=True
        )
        self._handle.wait()
        recv_buf = self._recv_buf

        if self.config.send_in_forward == SendForward.X:
            kv, kj, vj = _compute_kv(
                recv_buf, self.w_qkv, self.b, self.seq_local, self.h, self.d
            )
            kj, vj = kj.contiguous(), vj.contiguous()
        else:
            kv = recv_buf
            kj, vj = kv[0].contiguous(), kv[1].contiguous()

        return kj, vj, kv

    def maybe_forward(self, step: int, world_size: int) -> None:
        # Intra-node movement is handled entirely by the broadcast in `get`;
        # the only thing left to schedule is the next inter-node hop, issued a
        # full round ahead so it overlaps the `gpus_per_node` local broadcasts.
        """Schedule communication for the next ring step.

        Parameters
        ----------
        step : int
            Current ring step.
        world_size : int
            Number of processes in the distributed group.

        """
        if step == 0 or step >= world_size - 1:
            return
        if step % self.gpus_per_node != 0:
            return
        if step // self.gpus_per_node >= (self.group.size() // self.gpus_per_node) - 1:
            return

        assert self._node_chunk is not None
        # `_node_chunk` is read-only for the rest of the round (broadcast source
        # or idle), so it can be sent without a copy.
        self._launch_internode(self._node_chunk, step)
