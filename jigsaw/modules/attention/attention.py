"""Attention modules."""

import warnings
from typing import Any, Callable, Optional, cast

import torch
import torch.distributed as dist
from einops import rearrange

from jigsaw import hooks
from jigsaw.modules import linear

from . import config as attention_config
from .config import AttnBackend
from .functional.functional import RingAttentionFunction

try:
    import transformer_engine.pytorch as te

    class TransformerEngineAttention(torch.autograd.Function):
        """Batching rule for ``te.DotProductAttention`` under ``torch.vmap``.

        TE's fused kernels have no batching rule, so the vmapped dimension is
        folded into the batch dimension. Only :func:`te_attention` should call
        this, and only under ``torch.vmap``: a custom ``autograd.Function`` is a
        graph boundary, so a plain call would discard TE's own graph. The
        ``vmap`` staticmethod runs below the vmap level instead, where TE
        records its graph as usual.
        """

        @staticmethod
        def forward(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn: "te.DotProductAttention",
        ) -> torch.Tensor:
            """Run TE dot-product attention.

            Parameters
            ----------
            q, k, v : torch.Tensor
                Query, key and value in ``bshd`` format.
            attn : te.DotProductAttention
                Attention module to call.

            Returns
            -------
            torch.Tensor
                Attention output in ``bshd`` format.
            """
            out: torch.Tensor = attn(q, k, v)
            return out

        @staticmethod
        def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
            """Save nothing; TE records its own graph in :meth:`vmap`."""
            return None

        @staticmethod
        def vmap(
            info: Any,
            in_dims: tuple[Optional[int], ...],
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn: "te.DotProductAttention",
        ) -> tuple[torch.Tensor, int]:
            """Fold the vmapped dimension into the batch dimension."""
            q_bdim, k_bdim, v_bdim = in_dims[0], in_dims[1], in_dims[2]

            # Move the vmapped dimension to the front for each tensor.
            q = (
                q.movedim(q_bdim, 0)
                if q_bdim is not None
                else q.expand(info.batch_size, *q.shape)
            )
            k = (
                k.movedim(k_bdim, 0)
                if k_bdim is not None
                else k.expand(info.batch_size, *k.shape)
            )
            v = (
                v.movedim(v_bdim, 0)
                if v_bdim is not None
                else v.expand(info.batch_size, *v.shape)
            )

            samples = q.shape[0]
            # q/k/v now have shape [samples, batch, seq, heads, head_dim].
            # Fold samples into the batch dimension so TE sees a plain bshd tensor.
            q = q.flatten(0, 1)
            k = k.flatten(0, 1)
            v = v.flatten(0, 1)

            out = attn(q, k, v)

            # Unfold the samples dimension back out to the front.
            out = out.unflatten(0, (samples, -1))
            return out, 0

    def te_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn: "te.DotProductAttention",
    ) -> torch.Tensor:
        """Run TE dot-product attention, with TE's own backward.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Query, key and value in ``bshd`` format.
        attn : te.DotProductAttention
            Attention module to call.

        Returns
        -------
        torch.Tensor
            Attention output in ``bshd`` format.
        """
        if any(torch._C._functorch.is_functorch_wrapped_tensor(t) for t in (q, k, v)):
            out: torch.Tensor = TransformerEngineAttention.apply(q, k, v, attn)
        else:
            out = attn(q, k, v)
        return out

    JIGSAW_TE_AVAILABLE = True
except ModuleNotFoundError:
    JIGSAW_TE_AVAILABLE = False


def attn_core_dtype(x: torch.Tensor) -> torch.dtype:
    """Return the dtype of the attention core.

    This is the autocast dtype under autocast, bf16 on CUDA and the input dtype
    otherwise.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.dtype
        Dtype of the attention core.
    """
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    if x.is_cuda:
        return torch.bfloat16
    return x.dtype


class DistributedAttention(torch.nn.Module):
    """Distributed multi-head attention."""

    # type hints
    qkv: torch.nn.Linear | linear.DistributedXWT
    head_concat: torch.nn.Linear | linear.DistributedXWT

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
        ring: bool = False,
        attn_backend: AttnBackend | str = AttnBackend.SDPA,
        dtype: torch.dtype = torch.float32,
        config: Optional[attention_config.RingAttentionConfig] = None,
        out_bias: bool = True,
    ):
        """Initialize the module.

        The weights are replicated over ``spatial_group``, which splits the
        sequence, and sharded over ``channel_group``, which splits the heads.
        Their gradients are summed over ``spatial_group``.

        Parameters
        ----------
        in_dim : int
            Global input dimension.
        embed_dim : int
            Global embedding dimension, shared by all heads.
        out_dim : int
            Global output dimension.
        heads : int
            Global number of heads.
        dropout_p : float
            Dropout probability.
        local_rank : int
            Rank within the channel group. Kept for compatibility: the module
            reads the channel rank and the ring rank from the groups.
        device : torch.device
            Device of the module.
        spatial_group : dist.ProcessGroup
            Process group along the sequence dimension.
        channel_group : dist.ProcessGroup
            Process group along the channel dimension.
        bayesian : bool
            Whether the linear layers are Bayesian.
        qk_norm : bool
            Whether to normalize queries and keys.
        ring : bool
            Whether to use ring attention over ``spatial_group``. By default
            the ring recomputes the keys and values of remote chunks from the
            remote input with the local QKV weight, which needs an unsharded
            channel dimension: ``channel_group`` must have size 1, unless
            ``config`` makes the ring send and save the keys and values
            themselves (``send_in_forward=SendForward.KV`` and
            ``save_for_backward=SaveBackward.KV``).
        attn_backend : AttnBackend | str
            Attention core, ``"te"``, ``"sdpa"`` or ``"math"``.
        dtype : torch.dtype
            Data type of the parameters.
        config : RingAttentionConfig, optional
            Ring-attention configuration. Defaults to a configuration for
            ``attn_backend``.
        out_bias : bool
            Whether the output projection has a bias.
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
        self.ring = ring

        self.attn_backend = self._resolve_backend(AttnBackend(attn_backend))
        self.te = self.attn_backend == AttnBackend.TE
        # Only the non-ring TE core consumes/produces a bshd tensor; the ring
        # cores (functional.cores.te included) work in the ring's bhsd
        # convention and transpose internally, so forward() keeps the bhsd
        # rearrange for it.
        self._te_bshd = self.te and not ring

        self.heads = heads // self.channel_group.size()
        self.scale = (embed_dim // heads) ** -0.5
        self.empty = torch.empty(0, device=device)

        assert heads % self.channel_group.size() == 0, (
            "heads is not divisible by channel group size"
        )

        if self.ring:
            assert spatial_group.size() > 1, (
                "Cannot perform ring attention if spatial size is 1."
            )
            # With a channel split the local QKV weight is only a shard, too
            # little to rebuild a remote chunk's keys and values from its input:
            # the ring then has to send and save the keys and values themselves.
            sends_and_saves_kv = (
                config is not None
                and config.send_in_forward == attention_config.SendForward.KV
                and config.save_for_backward == attention_config.SaveBackward.KV
            )
            assert channel_group.size() == 1 or sends_and_saves_kv, (
                "Ring attention needs an unsharded channel dimension, but the "
                f"channel group has size {channel_group.size()}. With a channel "
                "split, pass a config with send_in_forward=SendForward.KV and "
                "save_for_backward=SaveBackward.KV."
            )
            self.attn_core = self._ring_attn_core
        else:
            self.attn_core = self._select_attn_core(embed_dim, heads)

        if channel_group.size() == 1:
            self.qkv = torch.nn.Linear(
                in_dim, embed_dim * 3, bias=False, device=device, dtype=self.dtype
            )
        else:
            self.qkv = linear.DistributedXWT(
                in_dim,
                embed_dim * 3,
                bias=False,
                local_rank=self.channel_group.rank(),
                device=self.device,
                model_group=self.channel_group,
                row_group=self.spatial_group,
                column_group=self.channel_group,
                bayesian=bayesian,
                dtype=self.dtype,
            )

        if channel_group.size() == 1:
            self.head_concat = torch.nn.Linear(
                embed_dim, out_dim, bias=out_bias, device=device, dtype=self.dtype
            )
        else:
            self.head_concat = linear.DistributedXWT(
                embed_dim,
                out_dim,
                bias=out_bias,
                local_rank=self.channel_group.rank(),
                device=self.device,
                model_group=self.channel_group,
                row_group=self.spatial_group,
                column_group=self.channel_group,
                bayesian=bayesian,
                dtype=self.dtype,
            )

        # The weights are replicated over the spatial group: sync them up and
        # sum their gradients. ``DistributedXWT`` already sums its own bias
        # gradient over its row group, which is the spatial group here.
        already_synced = (
            {id(self.head_concat.bias)}
            if isinstance(self.head_concat, linear.DistributedXWT)
            and self.head_concat.bias is not None
            else set()
        )
        for p in self.parameters():
            dist.broadcast(
                p.data, group_src=0, group=self.spatial_group, async_op=False
            )
            if id(p) not in already_synced:
                p.register_post_accumulate_grad_hook(
                    hooks.make_grad_sync_hook(self.spatial_group, async_op=True)
                )

        if self.qk_norm:
            self.q_norm, self.k_norm = (
                torch.nn.LayerNorm(embed_dim // heads, device=device),
                torch.nn.LayerNorm(embed_dim // heads, device=device),
            )

        self.heads = heads // self.channel_group.size()
        self.scale = (embed_dim // heads) ** -0.5

        self.config = (
            attention_config.RingAttentionConfig(
                device=device, attn_backend=self.attn_backend
            )
            if ring and config is None
            else config
        )

    @staticmethod
    def _resolve_backend(attn_backend: AttnBackend) -> AttnBackend:
        """Fall back from ``TE`` to ``SDPA`` if transformer_engine is unavailable."""
        if attn_backend is AttnBackend.TE and not JIGSAW_TE_AVAILABLE:
            warnings.warn(
                "attn_backend='te' but transformer_engine is unavailable; "
                "falling back to 'sdpa'. Install it or update your container."
            )
            return AttnBackend.SDPA
        return attn_backend

    def _select_attn_core(
        self, embed_dim: int, heads: int
    ) -> Callable[..., torch.Tensor]:
        """Return the non-ring attention core of ``self.attn_backend``."""
        match self.attn_backend:
            case AttnBackend.TE:
                self.attn = te.DotProductAttention(
                    num_attention_heads=self.heads,
                    kv_channels=embed_dim // heads,
                    num_gqa_groups=self.heads,
                    attention_dropout=self.dropout_p,
                    qkv_format="bshd",
                    attn_mask_type="no_mask",
                )
                return self._transformer_attn_core
            case AttnBackend.MATH:
                return self._attn_core_math
            case AttnBackend.SDPA:
                return self._attn_core
            case other:
                raise ValueError(f"unhandled attn_backend {other}")

    def _attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Query, key and value.
        **kwargs : torch.Tensor
            Unused.

        Returns
        -------
        torch.Tensor
            Attention output.
        """
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        return out

    def _attn_core_math(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention with the unfused math backend."""
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        return out

    def _transformer_attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Query, key and value.
        **kwargs : torch.Tensor
            Unused.

        Returns
        -------
        torch.Tensor
            Attention output.
        """
        # `self.attn` (a real nn.Module) applies its own training-mode gate
        # to `attention_dropout` internally, same as any other dropout module.
        return te_attention(q, k, v, self.attn)

    def _ring_attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ring attention.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Query, key and value.
        x : torch.Tensor
            Local input chunk.
        **kwargs : torch.Tensor
            Unused.

        Returns
        -------
        torch.Tensor
            Attention output.
        """
        out = RingAttentionFunction.apply(
            q,
            k,
            v,
            x,
            self.qkv.weight,
            self.scale,
            self.dropout_p if self.training else 0.0,
            self.spatial_group,
            self.spatial_group.rank(),
            self.config,
        )[0]
        out = cast(torch.Tensor, out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute distributed multi-head attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Attention output.
        """
        # shape: [n_batches, spatial_tokens, channel_tokens]
        qkv = self.qkv(x)

        if not self._te_bshd:
            qkv = rearrange(
                qkv, "... n s (three h d) -> three ... n h s d", three=3, h=self.heads
            )
        else:
            qkv = rearrange(
                qkv, "... n s (three h d) -> three ... n s h d", three=3, h=self.heads
            )

        core_dtype = attn_core_dtype(x)
        if self.qk_norm:
            q, k, v = qkv.unbind(0)
            q = self.q_norm(q).to(core_dtype)
            k = self.k_norm(k).to(core_dtype)
            v = v.to(core_dtype)
        else:
            q, k, v = qkv.to(core_dtype).unbind(0)

        out = self.attn_core(q, k, v, x=x)
        # The core runs in ``core_dtype`` (bf16 on CUDA, see ``attn_core_dtype``);
        # ``head_concat`` holds the model-dtype weights, so fold ``out`` back to
        # ``x.dtype`` before it. The non-bshd path also collapses heads here; the
        # ``_te_bshd`` core already returns ``... n s (h d)``.
        if not self._te_bshd:
            out = rearrange(out, "... n h s d -> ... n s (h d)").contiguous()
        out = out.to(x.dtype)

        out = cast(torch.Tensor, self.head_concat(out))
        # Output: [n_batch, seq_length, output_dim]
        return out


class Attention(torch.nn.Module):
    """Multi-head attention that can be converted with :meth:`convert_to_jigsaw`."""

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
        """Initialize the module.

        Parameters
        ----------
        in_dim : int
            Input dimension.
        embed_dim : int
            Embedding dimension shared by all heads. Must be divisible by
            ``heads``.
        out_dim : int
            Output dimension.
        heads : int
            Number of heads.
        device : torch.device
            Device of the module.
        dropout_p : float
            Dropout probability.
        qk_norm : bool
            Whether to normalize queries and keys.
        dtype : torch.dtype
            Data type of the parameters.
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
                torch.nn.RMSNorm(self.head_dim),
                torch.nn.RMSNorm(self.head_dim),
            )

    def _attn_core(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Query, key and value.
        **kwargs : torch.Tensor
            Unused.

        Returns
        -------
        torch.Tensor
            Attention output.
        """
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Attention output.
        """
        qkv = self.qkv(x)
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.heads, self.head_dim)

        q, k, v = qkv.permute(2, 0, 3, 1, 4).to(attn_core_dtype(x))
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
        attn_backend: AttnBackend | str = AttnBackend.SDPA,
        config: Optional[attention_config.RingAttentionConfig] = None,
        **kwargs: Any,
    ) -> "DistributedAttention":
        """Convert an attention module to a :class:`DistributedAttention`.

        The rows of the process mesh split the sequence and the columns split
        the heads. The weights are replicated over the rows and sharded over
        the columns.

        Parameters
        ----------
        attention : Attention
            Module to convert.
        process_mesh_dict : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        device : torch.device
            Device of the converted module.
        row_kw : str
            Key of the row process group.
        col_kw : str
            Key of the column process group.
        model_kw : str
            Key of the model process group. Unused, the ranks come from the
            row and column groups.
        ring : bool
            Whether to use ring attention over the rows. With its default
            ``config`` that needs a column group of size 1, see
            :class:`DistributedAttention`. Otherwise, each row attends within
            its own chunk.
        attn_backend : AttnBackend | str
            Attention core, ``"te"``, ``"sdpa"`` or ``"math"``.
        config : RingAttentionConfig, optional
            Ring-attention configuration. Defaults to a configuration for
            ``attn_backend``.
        **kwargs : Any
            Unused.

        Returns
        -------
        DistributedAttention
            Converted module.
        """
        channel_size = process_mesh_dict[col_kw].size()
        channel_rank = process_mesh_dict[col_kw].rank()
        spatial_group = process_mesh_dict[row_kw]

        assert attention.heads % channel_size == 0, (
            f"num_heads={attention.heads} not divisible by channel size {channel_size}"
        )

        instance = DistributedAttention(
            in_dim=attention.in_dim,
            embed_dim=attention.embed_dim,
            out_dim=attention.out_dim,
            heads=attention.heads,
            dropout_p=attention.dropout_p,
            local_rank=channel_rank,
            device=device,
            spatial_group=spatial_group,
            channel_group=process_mesh_dict[col_kw],
            bayesian=False,
            qk_norm=attention.qk_norm,
            ring=ring,
            attn_backend=attn_backend,
            dtype=attention.dtype,
            out_bias=attention.head_concat.bias is not None,
            config=config,
        )

        # The weights are replicated over the spatial group and sharded over the
        # channel group, so only the channel rank picks a slice.
        # QKV: the serial weight has shape [3 * embed_dim, in_dim], read as
        # [3, heads, head_dim, in_dim]. Block ``i`` of the local weight holds the
        # rows of head shard ``i``; the local columns are this rank's input shard.
        head_shards = torch.tensor_split(
            attention.qkv.weight.data.view(
                3, attention.heads, attention.head_dim, attention.in_dim
            ),
            channel_size,
            dim=1,
        )
        in_shard = attention.in_dim // channel_size
        qkv_blocks = [
            shard.reshape(-1, attention.in_dim)[
                :, channel_rank * in_shard : (channel_rank + 1) * in_shard
            ]
            for shard in head_shards
        ]
        instance.qkv.weight.data.copy_(torch.cat(qkv_blocks, dim=0))

        # Head concat: columns are the heads of this channel rank.
        embed_shard = attention.embed_dim // channel_size
        instance.head_concat.weight.data.copy_(
            attention.head_concat.weight.data[
                :, channel_rank * embed_shard : (channel_rank + 1) * embed_shard
            ]
        )
        serial_bias, local_bias = attention.head_concat.bias, instance.head_concat.bias
        if serial_bias is not None and local_bias is not None:
            out_shard = attention.out_dim // channel_size
            local_bias.data.copy_(
                serial_bias.data[
                    channel_rank * out_shard : (channel_rank + 1) * out_shard
                ]
            )

        # The serial weights need not agree across the spatial group.
        for p in instance.parameters():
            dist.broadcast(p.data, group_src=0, group=spatial_group)

        # Update head count to reflect local shard
        instance.heads = attention.heads // channel_size
        # embed_dim attribute: keep global value, since forward uses embed_dim // heads
        # which still gives head_dim correctly

        return instance
