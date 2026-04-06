import torch
import torch.distributed as dist
from einops import rearrange

from . import linear, mlp
from . import normalization as norm
from .utils_init import average_weights


class DistributedWindowedAttention(torch.nn.Module):
    """Distributed windowed attention."""

    def __init__(
        self,
        x_features: int,
        embed_dim: int,
        output_dim: int,
        heads: int,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
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
        self.model_group = model_group
        self.spatial_group = spatial_group
        self.channel_group = channel_group

        self.embed_dim = embed_dim
        self.x_features = x_features

        assert self.parallelism == 2, (
            "windowed attention is only designed for parallelism = 2"
        )

        self.qkv = linear.DistributedXWTSync(
            x_features // 2,
            embed_dim * 3,
            bias=False,
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self.parallelism,
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
            parallelism=self.parallelism,
            model_group=self.channel_group,
            sync_group=self.spatial_group,
            bayesian=bayesian,
        )

        self.heads = heads // self.parallelism

    @torch.compile(mode="reduce-overhead")
    def _attention_core(self, qkv, mask):
        # [qkv: [batch, seq, embed]]
        # [3, batch, seq, emb//3]
        # [3, batch, heads, seq, emb//3/heads]

        qkv = rearrange(
            qkv, "... n i (three h d) -> three ... n h i d", three=3, h=self.heads
        )
        q, k, v = qkv.unbind(0)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=0.0, attn_mask=mask
            )
        return out

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for distributed multi-head attention.

        x: (n_batch, vars, n_lat, n_lon)
        """
        # x_local: [n_batch, n_lat, n_lon, variables]
        x_patches_local = x
        # shape: [n_batches, spatial_tokens, channel_tokens]
        qkv = self.qkv(x_patches_local)

        out = self._attention_core(qkv, mask)

        values = rearrange(out, "... n h s d -> ... n s (h d)").contiguous()
        values = self.concatenator(values)

        # Output: [n_batch, seq_length, output_dim]
        return values


# Flash attention
class DistributedWindowedFlashAttention(torch.nn.Module):
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
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
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
        self.model_group = model_group
        self.spatial_group = spatial_group
        self.channel_group = channel_group

        self.embed_dim = embed_dim
        self.x_features = x_features
        self.dtype = dtype
        self.dropout = dropout

        assert self.parallelism == 2, (
            "windowed attention is only designed for parallelism = 2"
        )

        self.qkv = linear.DistributedXWTSync(
            x_features // 2,
            embed_dim * 3,
            bias=False,
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self.parallelism,
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
            parallelism=self.parallelism,
            model_group=self.channel_group,
            sync_group=self.spatial_group,
            bayesian=bayesian,
        )
        
        # self.q_norm, self.k_norm = torch.nn.LayerNorm(embed_dim // heads), torch.nn.LayerNorm(embed_dim // heads)

        self.heads = heads // 2

    @torch.compile(mode="reduce-overhead")
    def _attention_core(self, q, k, v, mask, dropout):
        
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=dropout, attn_mask=mask
            )
        return out

    @torch.compile(mode="reduce-overhead")
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
        out = self._attention_core(q, k, v, mask, self.dropout)

        # out has shape [batch, local_heads, local_sequence length, head_dim]
        out = rearrange(out, "... n h s d -> ... n s (h d)").contiguous().to(self.dtype)
        values = self.concatenator(out)

        # Output: [n_batch, seq_length, output_dim]
        return values


class DistributedWindowedTransformer(torch.nn.Module):
    """
    Windowed Transformer across multiple processes.

    The spatial domain is split across domain-parallel instances,
    and the channel domain retains domain and tensor parallel approaches.
    """

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
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bayesian: bool,
        dtype,
        flash_attn: bool = True,
    ):
        super().__init__()
        if flash_attn:
            self.attention_block = DistributedWindowedFlashAttention(
                x_features,
                embed_dim,
                output_dim,
                heads,
                dropout,
                local_rank,
                device,
                parallelism,
                model_group,
                spatial_group,
                channel_group,
                bayesian,
                dtype,
            ).to(device)
        else:
            self.attention_block = DistributedWindowedAttention(
                x_features,
                embed_dim,
                output_dim,
                heads,
                dropout,
                local_rank,
                device,
                parallelism,
                model_group,
                spatial_group,
                channel_group,
                bayesian,
                dtype,
            ).to(device)
        self.layer_norm1 = torch.nn.LayerNorm(output_dim//2)
        self.layer_norm2 = torch.nn.LayerNorm(output_dim//2)
        
        self.mlp = mlp.DistributedSynchronizedMLP(
            output_dim,
            output_dim,
            output_dim,
            bias=True,
            dropout=dropout,
            local_rank=local_rank,
            device=device,
            parallelism=parallelism,
            model_group=channel_group,
            sync_group=spatial_group,
            bayesian=bayesian,
        ).to(device)
        # average_weights(self, spatial_group, device)

    def forward(self, x: torch.tensor, mask=None):
        """Forward pass of distributed windowed transformer."""
        x = self.attention_block(self.layer_norm1(x), mask) + x
        x = self.mlp(self.layer_norm2(x)) + x
        return x


class DistributedNICEWindowedTransformer(torch.nn.Module):
    """
    Module for Distributed NICE Windowed Transformer.

    The spatial domain is split across domain-parallel instances,
    and the channel domain retains domain and tensor parallel approaches.
    Noise is replicated across all dtp processes.
    """

    def __init__(
        self,
        x_features: int,
        embed_dim: int,
        output_dim: int,
        heads: int,
        noise_dim: int,
        dropout: float,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bayesian: bool,
        dtype,
        flash_attn: bool = True,
    ):
        """Initialize DistributedNICEWindowedTransformer."""
        super().__init__()
        self.noise_dim = noise_dim
        if flash_attn:
            self.attention_block = DistributedWindowedFlashAttention(
                x_features,
                embed_dim,
                output_dim,
                heads,
                dropout,
                local_rank,
                device,
                parallelism,
                model_group,
                spatial_group,
                channel_group,
                bayesian,
                dtype,
            ).to(device)
        else:
            self.attention_block = DistributedWindowedAttention(
                x_features,
                embed_dim,
                output_dim,
                heads,
                local_rank,
                device,
                parallelism,
                model_group,
                spatial_group,
                channel_group,
                bayesian,
                dtype,
            ).to(device)

        self.layer_norm1 = norm.DistributedNICENorm(
            output_dim,
            self.noise_dim,
            local_rank,
            device,
            parallelism,
            spatial_group,
            channel_group,
        ).to(device)
        self.layer_norm2 = norm.DistributedNICENorm(
            output_dim,
            self.noise_dim,
            local_rank,
            device,
            parallelism,
            spatial_group,
            channel_group,
        ).to(device)

        self.mlp = mlp.DistributedSynchronizedMLP(
            output_dim,
            output_dim,
            output_dim,
            bias=True,
            dropout=dropout,
            local_rank=local_rank,
            device=device,
            parallelism=parallelism,
            model_group=channel_group,
            sync_group=spatial_group,
            bayesian=bayesian,
        ).to(device)

    @torch.compile
    def forward(self, x: torch.Tensor, noise: torch.Tensor, mask=None):
        """
        Perform forward pass for DistribuedNICEWindowedTransformer.

        inputs
        ------
        x: torch.Tensor
        noise: torch.Tensor
        mask: Union[None, torch.Tensor]
        """
        x = self.attention_block(self.layer_norm1(x, noise), mask) + x
        x = self.mlp(self.layer_norm2(x, noise)) + x

        return x
