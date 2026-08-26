"""Ring attention forward-backward implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from jigsaw.offload import NVLinkActivationOffloader
from .forward import build_ring_forward
from .backward import ring_backward

from .config import (
    RingAttentionConfig,
    SaveBackward,
    SaveScope,
    SendBackward,
    SendForward,
    SendOp,
)



def _compute_kv(
    xj: torch.Tensor, w_qkv: torch.Tensor, b: int, n_local: int, h: int, d: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (K, V) from X. Returns (KV, Kj, Vj). KV is the underlying buffer."""
    kv = (
        (xj @ w_qkv[d * h :, :].mT).reshape(b, n_local, 2, h, d).permute(2, 0, 3, 1, 4)
    ).contiguous()
    return kv, kv[0], kv[1]



def get_send_recv_ranks_hybrid(local_rank, gpus_per_node=4):
    base_node_rank = local_rank // gpus_per_node
    send_to = (local_rank + 1) % gpus_per_node + base_node_rank * gpus_per_node
    recv_from = (local_rank - 1) % gpus_per_node + base_node_rank * gpus_per_node

    return [send_to, recv_from]



class RingCtx(Protocol):
    """Protocol for types stored within ctx."""

    empty: torch.Tensor
    scale: float
    local_rank: int
    group: dist.ProcessGroup
    world_size: int
    config: RingAttentionConfig
    saved_tensors: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple
    ]


# --------------------------------------------------------------------------- #
# autograd.Function glue
# --------------------------------------------------------------------------- #
class RingAttentionFunction(torch.autograd.Function):
    """Ring Attention autograd function."""

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        scale: float,
        dropout_p: float,
        group: dist.ProcessGroup,
        empty: torch.Tensor,
        local_rank: int,
        config: RingAttentionConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Forward pass of ring attention."""
        world_size = dist.get_world_size(group)
        empty = torch.empty(0, device=x.device, dtype=x.dtype)

        ring_forward_fn = build_ring_forward(config)
        out, lse, chunks_to_save = ring_forward_fn(
            q,
            k,
            v,
            x,
            w_qkv,
            scale,
            dropout_p,
            group,
            empty,
            local_rank,
            world_size,
            config,
        )
        return out, lse, chunks_to_save

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: Any,
        output: Any,
    ) -> None:
        """Set up context. inputs/output match `forward`'s signature/return."""
        q, k, v, x, w_qkv, scale, dropout_p, group, empty, local_rank, config = inputs
        out, lse, chunks_to_save = output
        ctx.mark_non_differentiable(lse)
        ctx.mark_non_differentiable(*chunks_to_save)
        ctx.save_for_backward(
            q, out, x, lse.mT, w_qkv, *chunks_to_save # *[t for pair in chunks_to_save for t in pair]
        )
        c = cast(RingCtx, ctx)
        c.empty = empty
        c.scale = scale
        c.local_rank = local_rank
        c.group = group
        c.world_size = dist.get_world_size(group) # This is recomputed in forward and in setup context
        c.config = config

    @staticmethod
    def backward(
        ctx: RingCtx,
        dout: torch.Tensor,
        grad_l: torch.Tensor,
        grad_kv_chunks: list,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        """Backward pass of ring attention."""
        q, out, x, lse, w_qkv = ctx.saved_tensors[:5]
        chunks_to_save = ctx.saved_tensors[5:]

        dq, dk, dv = ring_backward(
            dout,
            q,
            out,
            x,
            lse,
            w_qkv,
            ctx.scale,
            ctx.group,
            ctx.local_rank,
            ctx.world_size,
            cast(tuple[torch.Tensor, ...], chunks_to_save),
            ring_config=ctx.config,
            empty=ctx.empty,
        )

        return dq, dk, dv, None, None, None, None, None, None, None, None
