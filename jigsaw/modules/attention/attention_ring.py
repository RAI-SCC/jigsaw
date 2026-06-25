"""
Ring Attention — bidirectional, sequence-parallel, torch.distributed (NCCL).

Each rank holds a local slice of Q, K, V of shape (B, H, N//world_size, d).
K and V are rotated around the ring for `world_size` steps. Each step calls
F.scaled_dot_product_attention for the local chunk, recovers (m_j, l_j) for
the online softmax merge, and accumulates O.

Backward uses the analytic attention gradient with globally-saved L.
dK/dV rotate counter-clockwise (opposite to forward) back to their home rank.
"""

from typing import Any, cast

import torch
import torch.distributed as dist
from einops import rearrange

from jigsaw.modules import linear

from . import ring_attention_helpers as ring_attn


class Ring(torch.nn.Module):
    """Distributed Ring attention."""

    def __init__(
        self,
        x_features: int,
        embed_dim: int,
        output_dim: int,
        heads: int,
        dropout_p: float,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        spatial_group: dist.ProcessGroup,
        bayesian: bool,
        dtype: torch.dtype,
        ring_config: ring_attn.RingAttentionConfig,
    ):
        """

        Torch Module to perform distributed multi-headed attention.

        x_features: int
            global number of x features
        embed_dim: int
            global embedding dimension
        output_dim: int
            global
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.parallelism = parallelism
        self.spatial_group = spatial_group

        self.embed_dim = embed_dim
        self.x_features = x_features
        self.dtype = dtype
        self.dropout_p = dropout_p

        self.heads = heads
        self.scale = embed_dim**-0.5
        self.empty = torch.empty(0, device=self.device, dtype=self.dtype)
        self.config = ring_config

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w_qkv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for distributed multi-head attention.

        x: (n_batch, vars, n_lat, n_lon)
        """
        out, *_ = ring_attn.RingAttentionFunction.apply(
            q,
            k,
            v,
            x,
            w_qkv,
            q.shape[-1] ** -0.5,
            self.dropout_p,
            self.spatial_group,
            self.empty,
            self.local_rank,
            self.config,
        )

        # Output: [n_batch, seq_length, output_dim]
        out = cast(torch.Tensor, out)
        return out


class RingBaseline(torch.nn.Module):
    """Distributed windowed attention."""

    def __init__(
        self,
        x_features: int,
        embed_dim: int,
        output_dim: int,
        heads: int,
        dropout: float,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        spatial_group: dist.ProcessGroup,
        bayesian: bool,
        dtype: torch.dtype,
    ):
        """

        Torch Module to perform distributed multi-headed attention.

        x_features: int
            global number of x features
        embed_dim: int
            global embedding dimension
        output_dim: int
            global
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.parallelism = parallelism
        self.spatial_group = spatial_group

        self.embed_dim = embed_dim
        self.x_features = x_features
        self.dtype = dtype
        self.dropout = dropout

        self.heads = heads
        self.scale = embed_dim**-0.5
        self.empty = torch.empty(0, device=self.device, dtype=self.dtype)

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass for distributed multi-head attention.

        x: (n_batch, vars, n_lat, n_lon)
        """
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=0.0, attn_mask=None
            )

        # Output: [n_batch, seq_length, output_dim]
        return out


class RingAttention(torch.nn.Module):
    """Distributed windowed attention."""

    def __init__(
        self,
        x_features: int,
        embed_dim: int,
        output_dim: int,
        heads: int,
        dropout: float,
        local_rank: int,
        device: torch.device,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bayesian: bool,
        dtype: torch.dtype,
        ring_config: ring_attn.RingAttentionConfig,
    ):
        """

        Torch Module to perform distributed multi-headed attention.

        x_features: int
            global number of x features
        embed_dim: int
            global embedding dimension
        output_dim: int
            global
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.model_group = model_group
        self.spatial_group = spatial_group
        self.channel_group = channel_group

        self.embed_dim = embed_dim
        self.x_features = x_features
        self.dtype = dtype
        self.dropout = dropout

        self.qkv = linear.DistributedXWT(
            x_features,
            embed_dim * 3,
            bias=False,
            local_rank=self.local_rank,
            device=self.device,
            model_group=self.channel_group,
            spatial_group=self.spatial_group,
            channel_group=self.channel_group,
            bayesian=bayesian,
        )

        self.head_concat = linear.DistributedXWT(
            embed_dim,
            output_dim,
            bias=True,
            local_rank=self.local_rank,
            device=self.device,
            model_group=self.channel_group,
            spatial_group=self.spatial_group,
            channel_group=self.channel_group,
            bayesian=bayesian,
        )

        self.heads = heads // channel_group.size()
        self.scale = embed_dim**-0.5
        self.empty = torch.empty(0, device=self.device, dtype=self.dtype)

        self.ring_config = ring_config
        ring_attn.ring_forward = ring_attn.build_ring_forward(ring_config)

    def _attention_core(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:

        out = ring_attn.RingAttentionFunction.apply(
            q,
            k,
            v,
            x,
            self.qkv.weight,
            self.scale,
            self.spatial_group,
            self.empty,
            None,
            self.local_rank,
            self.spatial_group.size(),
            self.ring_config,
        )
        out = cast(torch.Tensor, out)
        return out

    # @torch.compile(backend="eager")
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for distributed multi-head attention.

        x: (n_batch, vars, n_lat, n_lon)
        """
        # x_local: [n_batch, n_lat, n_lon, variables]
        x_patches_local = x
        # shape: [n_batches, spatial_tokens, channel_tokens]
        qkv = self.qkv(x_patches_local).to(torch.bfloat16)
        qkv = rearrange(
            qkv, "... n i (three h d) -> three ... n h i d", three=3, h=self.heads
        )
        q, k, v = qkv.unbind(0)
        # q = self.q_norm(q).to(torch.bfloat16)
        # k = self.k_norm(k).to(torch.bfloat16)
        # v = v.to(torch.bfloat16)
        mask = mask.to(torch.bfloat16) if mask is not None else None
        out = self._attention_core(q, k, v, x)

        # out has shape [batch, local_heads, local_sequence length, head_dim]
        out = rearrange(out, "... n h s d -> ... n s (h d)").contiguous().to(self.dtype)
        out = cast(torch.Tensor, self.head_concat(out))

        # Output: [n_batch, seq_length, output_dim]

        return out
