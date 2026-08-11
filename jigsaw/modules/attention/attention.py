"""Distributed attention modules."""

from typing import Any, cast

import torch
import torch.distributed as dist
from einops import rearrange

from jigsaw import hooks
from jigsaw.modules import linear

from . import config, functional


class DistributedAttention(torch.nn.Module):
    """Distributed attention."""

    # type hints
    qkv: torch.nn.Linear | linear.DistributedXWT

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        out_dim: int,
        heads: int,
        dropout_p: float,
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
        self.dropout_p = dropout_p
        self.ring_attention = ring_attention

        assert heads % self.channel_group.size() == 0, (
            "heads is not divisible by channel group size"
        )

        if self.ring_attention:
            assert spatial_group.size() > 1, (
                "Cannot perform ring attention if spatial size is 1."
            )

        self.attn_core = (
            self._ring_attn_core if self.ring_attention else self._attn_core
        )

        if channel_group.size() == 1:
            # device= is required here as much as in the DistributedXWT branch below:
            # without it the layer is created on the default device, and the parameter
            # broadcast over the spatial group further down then hands NCCL a CPU tensor
            # ("No backend type associated with device type cpu"). Only reachable when
            # the channel group is trivial, i.e. pure spatial parallelism.
            self.qkv = torch.nn.Linear(
                in_dim, embed_dim * 3, bias=False, device=self.device
            )
        else:
            self.qkv = linear.DistributedXWT(
                in_dim,
                embed_dim * 3,
                bias=False,
                local_rank=self.local_rank,
                device=self.device,
                model_group=self.channel_group,
                row_group=self.spatial_group,
                column_group=self.channel_group,
                bayesian=bayesian,
            )

        self.head_concat = linear.DistributedXWT(
            embed_dim,
            out_dim,
            bias=True,
            local_rank=self.local_rank,
            device=self.device,
            model_group=self.channel_group,
            row_group=self.spatial_group,
            column_group=self.channel_group,
            bayesian=bayesian,
        )

        if self.qk_norm:
            # NOTE: these are normalized over head_dim (the global embed_dim // heads,
            # since `heads` here is the constructor argument rather than self.heads), so
            # unlike qkv and head_concat they are replicated across the *channel* group
            # as well as the spatial one. The loop below only broadcasts and registers a
            # grad-sync hook over spatial_group, so each channel rank keeps a gradient
            # covering only its own heads: the replicas drift apart after the first
            # optimizer step and each is individually missing the other ranks'
            # contributions. Left as is pending a decision by the authors -- see also the
            # RMSNorm/LayerNorm divergence against the serial Attention below, and the
            # fact that convert_to_jigsaw does not copy q_norm/k_norm parameters.
            self.q_norm, self.k_norm = (
                torch.nn.LayerNorm(embed_dim // heads, device=device),
                torch.nn.LayerNorm(embed_dim // heads, device=device),
            )

        # sync up up spatial parameters
        for p in self.parameters():
            dist.broadcast(
                p.data, group_src=0, group=self.spatial_group, async_op=False
            )
            p.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.spatial_group, async_op=True)
            )

        self.heads = heads // self.channel_group.size()
        self.scale = (embed_dim // heads) ** -0.5
        self.empty = torch.empty(0, device=device)
        if ring_attention:
            self.config = config.RingAttentionConfig(device=device)

    def _attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention computation.

        Parameters
        ----------
        q: torch.Tensor
            Query tensor.
        k: torch.Tensor
            Keys tensor.
        v: torch.Tensor
            Values tensor.
        kwargs:
            Additional optional parameters.

        Returns
        -------
        out: torch.Tensor
            Output of attention computation.

        """
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=self.dropout_p
            )
        return out

    def _ring_attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ring attention computation.

        Parameters
        ----------
        q: torch.Tensor
            Query tensor.
        k: torch.Tensor
            Keys tensor.
        v: torch.Tensor
            Values tensor.
        x: torch.Tensor
            Input data.
        kwargs:
            Additional optional parameters.

        Returns
        -------
        out: torch.Tensor
            Output of attention computation.

        """
        out = functional.RingAttentionFunction.apply(
            q,
            k,
            v,
            x,
            self.qkv.weight,
            self.scale,
            self.dropout_p,
            self.spatial_group,
            self.empty,
            self.local_rank,
            self.config,
        )
        # RingAttentionFunction.forward returns (out, lse, chunks_to_save); the extra
        # two are marked non-differentiable and only exist for the backward pass. The
        # previous cast() here was typing.cast, which does nothing at runtime, so the
        # whole tuple was handed to rearrange() in forward.
        out, _lse, _chunks = out
        return cast(torch.Tensor, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for distributed multi-head attention.

        x: (n_batch, vars, n_lat, n_lon)
        """
        # shape: [n_batches, spatial_tokens, channel_tokens]
        qkv = self.qkv(x)
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

        out = self.attn_core(q, k, v, x=x)
        # out has shape [batch, local_heads, local_sequence length, head_dim]
        out = rearrange(out, "... n h s d -> ... n s (h d)").contiguous().to(self.dtype)
        out = cast(torch.Tensor, self.head_concat(out))
        # Output: [n_batch, seq_length, output_dim]
        return out


class Attention(torch.nn.Module):
    """
    Multi-headed Attention module compatible with jigsaw conversion.

    Multi-attention, starting from qkv mapping and ending with head concatenation.
    Uses torch's SDPA for the attn_core calculation.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        out_dim: int,
        heads: int,
        device: torch.device,
        dropout_p: float = 0.0,
        qk_norm: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize Attention module.

        Parameters
        ----------
        in_dim: int
            Input dimension of the incoming data.
        embed_dim: int
            Embedding dimension to be shared amongst all heads (global, not per-head).
            Must be divisible by heads.
        out_dim: int
            Output dimension.
        heads: int
            Number of attention heads.
        device: torch.device
            Device upon which computations occur.
        dropout_p: float
            Dropout probability.
        qk_norm: bool
            Boolean flag for whether to apply qk_norm before attn_core.
        dtype: torch.dtype
            Data type of the model and data.
        """
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
        self.device = device
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // self.heads
        self.out_dim = out_dim
        self.dropout_p = dropout_p
        self.attn_core = self._attn_core
        self.qk_norm = qk_norm
        self.dtype = dtype
        if qk_norm:
            self.q_norm, self.k_norm = (
                # device= for the same reason as in DistributedAttention: without it these
                # land on the default device while qkv and head_concat above are on `device`.
                torch.nn.RMSNorm(self.head_dim, device=device),
                torch.nn.RMSNorm(self.head_dim, device=device),
            )

    def _attn_core(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: torch.Tensor
    ) -> torch.Tensor:
        """
        Attention computation.

        Parameters
        ----------
        q: torch.Tensor
            Query tensor.
        k: torch.Tensor
            Keys tensor.
        v: torch.Tensor
            Values tensor.
        kwargs:
            Additional optional parameters.

        Returns
        -------
        out: torch.Tensor
            Output of attention computation.

        """
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=self.dropout_p
            )

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        out = self.attn_core(q=q, k=k, v=v, x=x)

        # out has shape [B, h, N, d]
        out = rearrange(out, "... h s d -> ... s (h d)").to(x.dtype)

        out = cast(torch.Tensor, self.head_concat(out))

        return out

    @classmethod
    def convert_to_jigsaw(
        cls,
        attention: "Attention",
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
        ring: bool = True,
        **kwargs: Any,
    ) -> "DistributedAttention":
        """
        Convert attention module to jigsaw distributed module.

        Expects a 2D process mesh to be passed. The "sequence/token" dimension should
        be the rows, whereas the channel dimension is the columns.

        Parameters
        ----------
        attention: Attention
            Instance of (non-distributed) multi-headed attention module defined within jigsaw.
            This is the only supported version of Attention that can be auto-converted.
        process_mesh_dict: dict
            Process mesh dict that comtains row, column, and model process groups.
        device: torch.device
            Device that computations are performed on.
        row_kw: str
            Key for accessing row process groups.
        col_kw: str
            Key for accessing col process groups.
        model_kw: str
            Key for accessing model process groups.
            Model process group should contain the Union of the row and column groups.
        ring: bool
            Boolean flag for whether ring attention should be computed on the "row" dimension.
            If not, the "row" dimension will simply perform attention in parallel over the different
            domains. This is useful for, domain-parallel i.e., Windowed attention.
        kwargs: Any
            Additional inputs.
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
            attention.in_dim,
        ).clone()

        instance = DistributedAttention(
            in_dim=attention.in_dim,
            embed_dim=attention.embed_dim,
            out_dim=attention.out_dim,
            heads=attention.heads,
            dropout_p=attention.dropout_p,
            local_rank=process_mesh_dict[model_kw].rank(),
            device=attention.device,
            spatial_group=process_mesh_dict[row_kw],
            channel_group=process_mesh_dict[col_kw],
            bayesian=False,
            qk_norm=attention.qk_norm,
            ring_attention=ring,
            dtype=attention.dtype,
        )

        # Build the Distributed Linear and overwrite its weight
        instance.qkv = linear.DistributedXWT.convert_to_jigsaw(
            attention.qkv,
            process_mesh_dict,
            device,
            row_kw,
            col_kw,
            col_kw,
        )  # type: ignore[assignment]

        # attn_core is deliberately not reassigned here. __init__ already selected
        # self._ring_attn_core from the ring_attention=ring argument passed above, and it
        # bound it to the instance. Assigning DistributedAttention._ring_attn_core onto
        # the instance instead stored the plain function: instance attributes bypass the
        # descriptor protocol, so self was never bound and the call in forward shifted
        # every argument by one (q became self), failing with a missing 'v'.

        # [3*embed_dim, input_dim]
        # Now, local W has shape [3*embed_dim, embed_dim//channel size]
        w_as_list = list(torch.tensor_split(w, channel_size, dim=1))
        # now a list of tensors: [3, :local_heads, head_dim, in_features]

        w_as_list = [w.reshape(-1, instance.in_dim) for w in w_as_list]

        w_local_col_shard = 3 * attention.embed_dim // channel_size
        head_shard_size = w_as_list[0].shape[1] // channel_size

        # need to remap the data by swapping around head and final dimension
        for i in range(channel_size):
            instance.qkv.weight.data[
                i * w_local_col_shard : (i + 1) * w_local_col_shard
            ] = w_as_list[i][
                ...,
                channel_rank * head_shard_size : (channel_rank + 1) * head_shard_size,
            ]

        instance.head_concat = linear.DistributedXWT.convert_to_jigsaw(
            attention.head_concat,
            process_mesh_dict,
            device,
            row_kw,
            col_kw,
            col_kw,
        )  # type: ignore[assignment]

        # __init__ registered the spatial grad-sync hook on every parameter, but qkv and
        # head_concat have since been replaced by the converted submodules, whose
        # parameters carry no hook. Both are built with model_kw=col_kw, so their weights
        # are replicated across the spatial group: without the hook each spatial rank
        # would keep only its own local gradient contribution and the replicas would
        # drift apart during training.
        #
        # Only the weights are hooked here. DistributedXWT.__init__ already registers a
        # hook on bias over its row group -- which is this same spatial group -- so
        # hooking every parameter would all-reduce the bias gradient twice. That is
        # latent rather than active today only because both sublayers are built
        # bias=False.
        for module in (instance.qkv, instance.head_concat):
            module.weight.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(instance.spatial_group, async_op=True)
            )

        # Update head count to reflect local shard
        instance.heads = heads_per_shard
        # embed_dim attribute: keep global value, since forward uses embed_dim // heads
        # which still gives head_dim correctly

        return instance
