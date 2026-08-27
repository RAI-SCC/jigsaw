"""Ring attention forward-backward implementations."""

from __future__ import annotations

from typing import Any, Protocol, cast

import torch
import torch.distributed as dist

from ..config import (
    RingAttentionConfig,
    SendOp
)

def _compute_kv(
    xj: torch.Tensor, w_qkv: torch.Tensor, b: int, n_local: int, h: int, d: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute key and value tensors from an input chunk.

    Parameters
    ----------
    xj : torch.Tensor
        Input sequence chunk.
    w_qkv : torch.Tensor
        Fused query, key, and value projection weights.
    b : int
        Batch size.
    n_local : int
        Sequence length of the local chunk.
    h : int
        Number of attention heads.
    d : int
        Embedding dimension per head.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The fused key-value tensor and its key and value views.
    """
    kv = (
        (xj @ w_qkv[d * h :, :].mT).reshape(b, n_local, 2, h, d).permute(2, 0, 3, 1, 4)
    ).contiguous()
    return kv, kv[0], kv[1]


def get_send_recv_ranks_hybrid(local_rank: int, gpus_per_node: int = 4) -> list[int]:
    """Return neighboring ranks for intra-node hybrid communication.

    Parameters
    ----------
    local_rank : int
        Rank of the current process.
    gpus_per_node : int, default=4
        Number of processes assigned to each node.

    Returns
    -------
    list[int]
        Two-element list containing the send rank and receive rank.
    """
    base_node_rank = local_rank // gpus_per_node
    send_to = (local_rank - 1) % gpus_per_node + base_node_rank * gpus_per_node
    recv_from = (local_rank + 1) % gpus_per_node + base_node_rank * gpus_per_node

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
        """Run the ring-attention forward pass.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Local query, key, and value tensors.
        x : torch.Tensor
            Local input sequence chunk.
        w_qkv : torch.Tensor
            Fused query, key, and value projection weights.
        scale : float
            Attention scaling factor.
        dropout_p : float
            Attention dropout probability.
        group : dist.ProcessGroup
            Distributed process group used for ring communication.
        empty : torch.Tensor
            Empty tensor used by communication helpers.
        local_rank : int
            Rank within the process group.
        config : RingAttentionConfig
            Ring-attention configuration.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, tuple]
            Attention output, log-sum-exp values, and saved chunks.
        """
        world_size = dist.get_world_size(group)
        empty = torch.empty(0, device=x.device, dtype=x.dtype)
        from .forward import build_ring_forward
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
        """Save forward results and metadata for the backward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context receiving saved tensors and metadata.
        inputs : Any
            Inputs passed to :meth:`forward`.
        output : Any
            Values returned by :meth:`forward`.
        """
        q, k, v, x, w_qkv, scale, dropout_p, group, empty, local_rank, config = inputs
        out, lse, chunks_to_save = output
        ctx.mark_non_differentiable(lse)
        ctx.mark_non_differentiable(*chunks_to_save)
        ctx.save_for_backward(
            q,
            out,
            x,
            lse.mT,
            w_qkv,
            *chunks_to_save,  # *[t for pair in chunks_to_save for t in pair]
        )
        c = cast(RingCtx, ctx)
        c.empty = empty
        c.scale = scale
        c.local_rank = local_rank
        c.group = group
        c.world_size = dist.get_world_size(
            group
        )  # This is recomputed in forward and in setup context
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
        """Compute gradients for the ring-attention forward pass.

        Parameters
        ----------
        ctx : RingCtx
            Autograd context containing saved tensors and configuration.
        dout : torch.Tensor
            Gradient of the attention output.
        grad_l : torch.Tensor
            Gradient placeholder for the non-differentiable log-sum-exp output.
        grad_kv_chunks : list
            Gradient placeholders for saved non-differentiable chunks.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Gradients for differentiable inputs followed by ``None`` for
            non-differentiable inputs.
        """
        q, out, x, lse, w_qkv = ctx.saved_tensors[:5]
        chunks_to_save = ctx.saved_tensors[5:]

        from .backward import ring_backward

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
