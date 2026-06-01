"""
Ring Attention — bidirectional, sequence-parallel, torch.distributed (NCCL).

Each rank holds a local slice of Q, K, V of shape (B, H, N//world_size, d).
K and V are rotated around the ring for `world_size` steps. Each step calls
F.scaled_dot_product_attention for the local chunk, recovers (m_j, l_j) for
the online softmax merge, and accumulates O.

Backward uses the analytic attention gradient with globally-saved L.
dK/dV rotate counter-clockwise (opposite to forward) back to their home rank.
"""

import torch
import torch.distributed as dist
from einops import rearrange

from . import linear
from . import ring_attention_helpers as ring_attn


class Ring(torch.nn.Module):
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
        dtype,
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
        self.dropout = dropout

        self.heads = heads
        self.scale = embed_dim**-0.5
        self.empty = torch.empty(0, device=self.device, dtype=self.dtype)
        self.config = ring_config

        ring_attn.ring_forward = ring_attn.build_ring_forward(ring_config)

    # @torch.compile(backend="eager")
    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w_qkv: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for distributed multi-head attention.

        x: (n_batch, vars, n_lat, n_lon)
        """
        mask = mask.to(torch.bfloat16) if mask is not None else None
        out, *_ = ring_attn.RingAttentionFunction.apply(
            q,
            k,
            v,
            x,
            w_qkv,
            q.shape[-1] ** -0.5,
            self.spatial_group,
            self.empty,
            self.local_rank,
            self.spatial_group.size(),
            self.config,
        )

        # Output: [n_batch, seq_length, output_dim]
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
        dtype,
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
        w_qkv: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for distributed multi-head attention.

        x: (n_batch, vars, n_lat, n_lon)
        """
        mask = mask.to(torch.bfloat16) if mask is not None else None
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=0.0, attn_mask=mask
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
        dtype,
        ring_config,
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

        self.qkv = linear.DistributedXWTSync(
            x_features // 2,
            embed_dim * 3,
            bias=False,
            local_rank=self.local_rank,
            device=self.device,
            parallelism=channel_group.size(),
            model_group=self.channel_group,
            sync_group=self.spatial_group,
            bayesian=bayesian,
        )

        self.concatenator = linear.DistributedXWTSync(
            embed_dim // 2,
            output_dim,
            bias=True,
            local_rank=self.local_rank,
            device=self.device,
            parallelism=channel_group.size(),
            model_group=self.channel_group,
            sync_group=self.spatial_group,
            bayesian=bayesian,
        )

        # self.q_norm, self.k_norm = torch.nn.LayerNorm(embed_dim // heads), torch.nn.LayerNorm(embed_dim // heads)

        self.heads = heads // channel_group.size()
        self.scale = embed_dim**-0.5
        self.empty = torch.empty(0, device=self.device, dtype=self.dtype)
        self.ring_config = ring_config

    def _attention_core(self, q, k, v, x):

        out = RingAttentionFunction.apply(
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
        values = self.concatenator(out)

        # Output: [n_batch, seq_length, output_dim]
        return values


class RingAttentionFunction(torch.autograd.Function):
    """Ring Attention autograd function."""

    @staticmethod
    def forward(q, k, v, x, w_qkv, scale, group, empty, rank, world_size, config):
        """Forward pass of ring attention."""
        b, h, n_local, d = q.shape

        world_size = dist.get_world_size(group)
        out = torch.zeros_like(q)
        lse = torch.full((b, n_local, h), float("-inf"), device=q.device, dtype=q.dtype)
        empty = torch.empty(0, device=x.device, dtype=x.dtype)

        out, lse, chunks_to_save = ring_attn.ring_forward(
            q, k, v, x, w_qkv, scale, group, empty, rank, world_size
        )

        out = out.to(torch.bfloat16)

        return out, lse, chunks_to_save

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, w, b, local_rank, device, selector, spatial_group, bayesian)
        output: xwt
        """
        (q, k, v, x, w_qkv, scale, group, empty, local_rank, world_size, config) = (
            inputs
        )

        out, lse, chunks_to_save = output
        ctx.mark_non_differentiable(lse)
        ctx.mark_non_differentiable(*chunks_to_save)

        ctx.save_for_backward(
            q, out, x, lse.mT, w_qkv, *[chunk for chunk in chunks_to_save]
        )

        ctx.W_qkv = w_qkv
        ctx.empty = empty
        ctx.scale = scale
        ctx.local_rank = local_rank
        ctx.group = group
        ctx.world_size = world_size
        ctx.config = config

    @staticmethod
    def backward(ctx, dout, grad_l, grad_kv_chunks):
        """Backward pass of ring attention."""
        scale = ctx.scale
        group = ctx.group
        world_size = ctx.world_size
        empty = ctx.empty
        local_rank = ctx.local_rank
        config = ctx.config

        q, out, x, lse, w_qkv = ctx.saved_tensors[:5]
        chunks_to_save = ctx.saved_tensors[5:]

        dq, dk, dv = ring_attn.ring_backward(
            dout,
            q,
            out,
            x,
            lse,
            w_qkv,
            scale,
            group,
            local_rank,
            world_size,
            chunks_to_save,
            ring_config=config,
            empty=empty,
        )

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
