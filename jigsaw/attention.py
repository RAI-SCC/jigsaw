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

        self.softmax = norm.DistributedSoftMax(dim=-1, group=self.channel_group)
        self.xwt = linear.DistributedMatMulXWTranspose(
            self.local_rank,
            self.device,
            self.parallelism,
            model_group=self.channel_group,
            spatial_group=self.spatial_group,
            bayesian=bayesian,
        )
        self.xw = linear.DistributedMatMulXW(
            local_rank,
            device,
            self.parallelism,
            model_group=self.channel_group,
            bayesian=bayesian,
        )

        # TODO: check working
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

        self.heads = heads

        self.scale = (embed_dim * self.parallelism // self.heads) ** (-0.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for distributed multi-head attention.

        x: (n_batch, vars, n_lat, n_lon)
        """
        # x_local: [n_batch, n_lat, n_lon, variables]
        x_patches_local = x
        # shape: [n_batches, spatial_tokens, channel_tokens]
        qkv = self.qkv(x_patches_local)

        q, k, v = torch.tensor_split(qkv, 3, dim=-1)

        q = rearrange(q, "... n i (h d) -> ... n h i d", h=self.heads).contiguous()
        k = rearrange(k, "... n i (h d) -> ... n h i d", h=self.heads).contiguous()
        v = rearrange(v, "... n i (h d) -> ... n h i d", h=self.heads).contiguous()

        q = self.scale * q
        attention = self.xwt(q, k, None)
        if mask is not None:
            attention = attention + mask

        attention = self.softmax(attention)

        values = self.xw(attention, v, None)

        values = rearrange(values, "... n h s d -> ... n s (h d)").contiguous()
        values = self.concatenator(values)

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
    ):
        super().__init__()
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
        ).to(device)
        self.layer_norm1 = norm.DistributedLayerNorm(
            output_dim // 2, local_rank, device, spatial_group, channel_group
        ).to(device)
        self.layer_norm2 = norm.DistributedLayerNorm(
            output_dim // 2, local_rank, device, spatial_group, channel_group
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

        average_weights(self, spatial_group, device)

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
    ):
        """Initialize DistributedNICEWindowedTransformer."""
        super().__init__()
        self.noise_dim = noise_dim
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
