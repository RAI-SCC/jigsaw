import torch
import torch.distributed as dist
from einops import rearrange

from . import attention_ring as ring_attn
from . import linear
from . import hooks


class DistributedAttention(torch.nn.Module):
    """Distributed attention."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        out_dim: int,
        heads: int,
        dropout: float,
        local_rank: int,
        device: torch.device,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bayesian: bool = False,
        qk_norm: bool = False,
        ring_attention: bool = False,
        dtype: torch.dtype = torch.float32,
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
        self.spatial_group = spatial_group
        self.channel_group = channel_group

        self.embed_dim = embed_dim
        self.in_dim = in_dim
        self.qk_norm = qk_norm
        self.dtype = dtype
        self.dropout = dropout
        self.ring_attention = ring_attention
        
        assert heads % self.channel_group.size() == 0, "heads is not divisible by channel group size"
        
        if self.ring_attention:
            assert spatial_group.size() > 1, (
                "Cannot perform ring attention if spatial size is 1."
            )

        self.attn_core = self._ring_attn_core if self.ring_attention else self._attn_core
        print(f'attn core is {self.attn_core}')
        self.qkv = linear.DistributedXWT(
            in_dim,
            embed_dim * 3,
            bias=False,
            local_rank=self.local_rank,
            device=self.device,
            model_group=self.channel_group,
            spatial_group=self.spatial_group,
            channel_group=self.channel_group,
            bayesian=bayesian,
        )

        self.concatenator = linear.DistributedXWT(
            embed_dim,
            out_dim,
            bias=True,
            local_rank=self.local_rank,
            device=self.device,
            model_group=self.channel_group,
            spatial_group=self.spatial_group,
            channel_group=self.channel_group,
            bayesian=bayesian,
        )

        # sync up up spatial stuff 
        for p in self.parameters():
            dist.broadcast(p.data, group_src=0, group=self.spatial_group, async_op=False)
            p.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.spatial_group, async_op=True)
            )
        
        if self.qk_norm:
            self.q_norm, self.k_norm = (
                torch.nn.RMSNorm(embed_dim // heads),
                torch.nn.RMSNorm(embed_dim // heads),
            )

        self.heads = heads // self.channel_group.size()

    def _attn_core(self, q, k, v, dropout, **kwargs):

        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=dropout
            )
        return out

    def _ring_attention_core(self, q, k, v, x, **kwargs):

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
        )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for distributed multi-head attention.

        x: (n_batch, vars, n_lat, n_lon)
        """
        # x_local: [n_batch, n_lat, n_lon, variables]
        x_patches_local = x
        # shape: [n_batches, spatial_tokens, channel_tokens]
        qkv = self.qkv(x_patches_local)
        qkv = rearrange(
            qkv, "... n i (three h d) -> three ... n h i d", three=3, h=self.heads
        )

        if self.qk_norm:
            q, k, v = qkv.unbind(0)
            q = self.q_norm(q).to(torch.bfloat16)
            k = self.k_norm(k).to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        else:
            q, k, v = qkv.to(torch.bfloat16).unbind(0)

        out = self.attn_core(q, k, v, dropout=self.dropout, x=x)

        # out has shape [batch, local_heads, local_sequence length, head_dim]
        out = rearrange(out, "... n h s d -> ... n s (h d)").contiguous().to(self.dtype)
        values = self.concatenator(out)

        # Output: [n_batch, seq_length, output_dim]
        return values


class Attention(torch.nn.Module):
    def __init__(
        self, in_dim, embed_dim, out_dim, heads, device, dropout_p=0.0
    ):
        super().__init__()
        self.qkv = torch.nn.Linear(
            in_features=in_dim,
            out_features=3 * embed_dim,
            bias=False,
            device=device,
        )
        self.head_concat = torch.nn.Linear(
            in_features=embed_dim, out_features=out_dim, bias=False, device=device
        )
        self.heads = heads
        self.in_features = in_dim
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // self.heads
        self.out_features = out_dim
        self.dropout_p = dropout_p
        self.attn_core = self._attn_core

    def _attn_core(self, q, k, v, dropout, **kwargs):
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=self.dropout_p
            )

        return out

    def forward(self, x):
        """
        Forward pass of multi-headed attention.

        Automatically uses the torch SDPA backend.

        Parameters
        ----------
        x: torch.Tensor
            Tensor upon which to calculate attention.

        Returns
        -------
        out: torch.Tensor
            Result of multi-headed attention.

        """
        qkv = self.qkv(x)
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.heads, self.head_dim)

        q, k, v = qkv.permute(2, 0, 3, 1, 4).to(torch.bfloat16)
        out = self.attn_core(q, k, v, dropout=self.dropout_p)

        # out has shape [B, h, N, d]
        out = rearrange(out, "... h s d -> ... s (h d)").to(x.dtype)

        out = self.head_concat(out)
        return out

    def convert_attention_to_jigsaw(
        self,
        attention,
        process_mesh_dict: dict,
        device: torch.device,
        ring: bool = True,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ):
        """
        Convert attention module to jigsaw distributed module.

        Expects a 2D process mesh to be passed. The "sequence/token" dimension should
        be the rows, whereas the channel dimension is the columns.

        Parameters
        ----------
        attention: Attention
            Instance of (non-distributed) multi-headed attention module defined within jigsaw.
            This is the only supported version of Attention that can be auto-converted.
        process_mesh: dict
            Process mesh dict that comtains row, column, and model process groups.
        device: torch.device
            Device that computations are performed on.
        ring: bool
            Boolean flag for whether ring attention should be computed on the "row" dimension.
            If not, the "row" dimension will simply perform attention in parallel over the different
            domains. This is useful for, domain-parallel i.e., Windowed attention.
        row_kw: str
            Key for accessing row process groups.
        col_kw: str
            Key for accessing col process groups.
        model_kw: str
            Key for accessing model process groups.
            Model process group should contain the Union of the row and column groups.
        """
        channel_size = process_mesh_dict[col_kw].size()
        channel_rank = process_mesh_dict[col_kw].rank()
        spatial_size = process_mesh_dict[row_kw].size()

        assert attention.heads % channel_size == 0, (
            f"num_heads={attention.heads} not divisible by channel size {channel_size}"
        )
        assert attention.embed_dim % spatial_size == 0, (
            f"embed_dim={attention.embed_dim} not divisible by spatial size {spatial_size}"
        )

        heads_per_shard = attention.heads // channel_size

        # --- shard the fused QKV ---
        # W has shape [3*embed_dim, in_dim]
        # Now, W has shape [3, heads, head_dim, input_dim]
        w = attention.qkv.weight.data.view(
            3,
            attention.heads,
            attention.embed_dim // attention.heads,
            attention.in_features,
        ).clone()

        # Build the Distributed Linear and overwrite its weight
        attention.qkv = linear.DistributedXWT.convert_to_jigsaw(
            attention.qkv,
            process_mesh_dict,
            device,
            row_kw,
            col_kw,
            model_kw,
        )
        if ring:
            attention.attn_core = DistributedAttention._ring_attention_core

        # [3*embed_dim, input_dim]
        # Now, local W has shape [3*embed_dim, embed_dim//channel size]
        w_as_list = torch.tensor_split(w, channel_size, dim=1)
        # now a list of tensors: [3, :local_heads, head_dim, in_features]

        w_as_list = [w.reshape(-1, attention.in_features) for w in w_as_list]
        # global W has shape {768, 256}

        w_local_col_shard = 3 * attention.embed_dim // channel_size
        head_shard_size = w_as_list[0].shape[1] // channel_size

        # need to remap the data by swapping around head and final dimension
        for i in range(channel_size):
            attention.qkv.weight.data[
                i * w_local_col_shard : (i + 1) * w_local_col_shard
            ] = w_as_list[i][
                ...,
                channel_rank * head_shard_size : (channel_rank + 1) * head_shard_size,
            ]

        attention.head_concat = linear.DistributedXWT.convert_to_jigsaw(
            attention.head_concat,
            process_mesh_dict,
            device,
            row_kw,
            col_kw,
            model_kw,
        )

        # Update head count to reflect local shard
        attention.heads = heads_per_shard
        # embed_dim attribute: keep global value, since forward uses embed_dim // heads
        # which still gives head_dim correctly

        return attention
