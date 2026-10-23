"""Distributed attention modules."""

import warnings
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.distributed as dist
from einops import rearrange

from jigsaw import parallel
from jigsaw.modules import linear

from . import _config as attention_config
from ._config import AttnBackend
from ._functional.cores import te as te_core
from ._functional.ring import _RingAttention, get_hybrid_topology

#: Whether ``attn_backend="te"`` can run (cuDNN or flash core importable).
JIGSAW_TE_AVAILABLE = te_core.TE_CORE_AVAILABLE

if JIGSAW_TE_AVAILABLE:

    class _TransformerEngineAttention(torch.autograd.Function):
        """Autograd wrapper around the TransformerEngine attention kernels.

        Runs the same :mod:`._functional.cores.te` kernels as the ring's inner
        step (cuDNN on Hopper+, ``flash_attn`` on Ampere), in ``bshd`` layout,
        with their native backward and ``rng_state``. The softmax LSE is dropped.

        Unsupported: FP8, bias, mask and GQA. A kernel that cannot serve a shape
        raises rather than falling back. ``torch.vmap`` is supported via
        :meth:`vmap`.
        """

        @staticmethod
        def forward(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            scale: float,
            dropout_p: float,
            core: Any,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            """Run TE attention on ``core``.

            Parameters
            ----------
            q, k, v : torch.Tensor
                Query/key/value tensors in ``bshd`` format
                (``[batch, seq, heads, head_dim]``), fp16 or bf16.
            scale : float
                Softmax scale.
            dropout_p : float
                Dropout probability, already gated on training mode by the caller.
            core : module
                :mod:`._functional.cores.te.cudnn` or :mod:`._functional.cores.te.flash`.

            Returns
            -------
            tuple
                ``(out, logsumexp, rng_state)`` with ``out`` of shape
                ``[batch, seq, heads * head_dim]``. Callers take ``[0]``; the rest
                is for :meth:`backward`.
            """
            out, lse, rng_state = core.forward(q, k, v, scale, dropout_p, layout="bshd")
            return out.flatten(-2), lse, rng_state

        @staticmethod
        def setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
            """Save what the backward kernel needs: q/k/v, out, LSE and the RNG state."""
            _q, _k, _v, scale, dropout_p, core = inputs
            out, lse, rng_state = output
            ctx.save_for_backward(*inputs[:3], out, lse, rng_state)
            ctx.scale = scale
            ctx.dropout_p = dropout_p
            ctx.core = core
            ctx.mark_non_differentiable(lse)

        @staticmethod
        def backward(  # type: ignore[override]
            ctx: Any,
            grad_out: torch.Tensor,
            _grad_lse: torch.Tensor | None,
            _grad_rng: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            """dq/dk/dv straight from the kernel's backward, with no forward recompute."""
            q, k, v, out, lse, rng_state = ctx.saved_tensors
            bshd = q.shape  # [b, s, h, d]; forward merged (h, d) in its output
            dq, dk, dv = ctx.core.backward(
                grad_out.reshape(bshd),
                q,
                k,
                v,
                out.reshape(bshd),
                lse,
                ctx.scale,
                dropout_p=ctx.dropout_p,
                rng_state=rng_state,
                layout="bshd",
            )
            return dq, dk, dv, None, None, None

        @staticmethod
        def vmap(
            info: Any,
            in_dims: tuple[int | None, ...],
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            scale: float,
            dropout_p: float,
            core: Any,
        ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None], tuple]:
            """Fold the vmapped (samples) axis into the batch, run one plain call, unfold.

            The TE kernels have no batching rule; folding is exact because
            attention is independent across the batch axis.
            """
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

            out, lse, rng_state = _TransformerEngineAttention.apply(
                q, k, v, scale, dropout_p, core
            )

            # Unfold the samples dimension back out to the front.
            out = out.unflatten(0, (samples, -1))
            lse = lse.unflatten(0, (samples, -1))
            return (out, lse, rng_state), (0, 0, None)


def attn_core_dtype(x: torch.Tensor) -> torch.dtype:
    """Return the dtype of the attention core.

    The autocast dtype under autocast, else the input dtype, except fp32/fp64 on
    CUDA, which runs in bf16 (flash rejects fp32; fp16 would need loss scaling).

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
    if x.is_cuda and x.dtype not in (torch.float16, torch.bfloat16):
        return torch.bfloat16
    return x.dtype


def _mask_to(attn_mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    """Cast an additive (float) mask to the core dtype; SDPA rejects a mismatch."""
    if attn_mask is not None and attn_mask.is_floating_point():
        return attn_mask.to(dtype)
    return attn_mask


class DistributedAttention(torch.nn.Module):
    """Distributed multi-head attention, parallelized over ``spatial_group``.

    Three mutually exclusive schemes, selected by ``ring`` / ``ulysses``: plain
    (each rank attends to its own local sequence shard), ring (the input ``x``
    circulates around ``spatial_group``, or K/V with a channel group > 1 or
    ``qk_norm``), or Ulysses (a single all-to-all trades sequence
    shard for head shard; flat, or hierarchical when ``ulysses_intra_size``
    is also set: Ulysses within groups of ranks, nested in a ring across them).

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
    local_rank : int
        Rank within the channel group. Kept for compatibility: the module
        reads the channel rank and the ring rank from the groups.
    device : torch.device
        Device of the module.
    spatial_group : dist.ProcessGroup
        Process group along the sequence dimension.
    channel_group : dist.ProcessGroup
        Process group along the channel dimension.
    dropout_p : float
        Dropout probability.
    qk_norm : bool
        Whether to normalize queries and keys.
    ring : bool
        Run ring (sequence-parallel) attention over ``spatial_group``. The
        inner per-chunk attention still uses ``attn_backend``.
    ulysses : bool
        Run Ulysses (all-to-all, head-parallel) attention over
        ``spatial_group``. Mutually exclusive with ``ring``; needs
        ``heads // channel_group.size()`` divisible by ``spatial_group.size()``.
        See :mod:`._functional.ulysses`.
    ulysses_intra_size : int
        If ``> 0`` (with ``ulysses``), run hierarchical Ulysses: all-to-all
        within groups of this many spatial ranks, nested in a ring across
        them. ``0`` (default) runs flat Ulysses.
    ulysses_inter_save_scope : SaveScope | str
        ``save_scope`` of the hierarchical scheme's outer ring.
    ulysses_inter_overlap : bool
        ``overlap_comm`` of the hierarchical scheme's outer ring. The
        backward overlaps only with ``ulysses_inter_save_scope=LOCAL``.
    ulysses_inter_offload : Offload | str
        ``offload`` of the hierarchical scheme's outer ring.
    attn_backend : AttnBackend | str
        Attention core implementation: ``"te"``, ``"sdpa"`` (default) or
        ``"math"``. See :class:`._config.AttnBackend`.
    dtype : torch.dtype
        Data type of the parameters.
    config : RingAttentionConfig, optional
        Ring-attention configuration. Defaults to a configuration for
        ``attn_backend``.
    out_bias : bool
        Whether the output projection has a bias.
    """

    qkv: torch.nn.Linear | linear.DistributedLinear
    head_concat: torch.nn.Linear | linear.DistributedLinear

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        out_dim: int,
        heads: int,
        local_rank: int,
        device: torch.device,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        dropout_p: float = 0.0,
        qk_norm: bool = False,
        ring: bool = False,
        ulysses: bool = False,
        ulysses_intra_size: int = 0,
        ulysses_inter_save_scope: attention_config.SaveScope | str = (
            attention_config.SaveScope.LOCAL
        ),
        ulysses_inter_overlap: bool = False,
        ulysses_inter_offload: attention_config.Offload | str = (
            attention_config.Offload.DISABLED
        ),
        attn_backend: AttnBackend | str = AttnBackend.SDPA,
        dtype: torch.dtype = torch.float32,
        config: attention_config.RingAttentionConfig | None = None,
        out_bias: bool = True,
    ) -> None:
        """Initialize the module.

        The weights are replicated over ``spatial_group``, which splits the
        sequence, and sharded over ``channel_group``, which splits the heads.
        Their gradients are summed over ``spatial_group``.
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
        self.ulysses = ulysses
        self.ulysses_intra_size = ulysses_intra_size if ulysses else 0
        self.ulysses_inter_save_scope = (
            ulysses_inter_save_scope
            if isinstance(ulysses_inter_save_scope, attention_config.SaveScope)
            else attention_config.SaveScope[ulysses_inter_save_scope]
        )
        self.ulysses_inter_overlap = bool(ulysses_inter_overlap)
        self.ulysses_inter_offload = (
            ulysses_inter_offload
            if isinstance(ulysses_inter_offload, attention_config.Offload)
            else attention_config.Offload[ulysses_inter_offload]
        )
        if (
            self.ulysses_inter_overlap
            and self.ulysses_intra_size
            and self.ulysses_inter_save_scope == attention_config.SaveScope.GLOBAL
        ):
            warnings.warn(
                "ulysses_inter_overlap with ulysses_inter_save_scope=GLOBAL "
                "overlaps only the forward outer K/V ring: GLOBAL keeps "
                "every arrived chunk and all-reduces dK/dV, so the backward has "
                "no re-ringing relay to hide. Use ulysses_inter_save_scope=LOCAL "
                "to overlap the backward too.",
                stacklevel=2,
            )
        if ring and ulysses:
            raise ValueError("ring and ulysses are mutually exclusive")

        self.attn_backend = self._resolve_backend(
            attention_config._coerce_enum(AttnBackend, attn_backend, "attn_backend")
        )
        self.te = self.attn_backend == AttnBackend.TE
        # ``_te_bshd``: the plain TE core returns heads already folded.
        self._bshd_qkv = self.te or self.ulysses
        self._te_bshd = self.te and not ring and not ulysses

        assert heads % self.channel_group.size() == 0, (
            "heads is not divisible by channel group size"
        )
        assert embed_dim % heads == 0, "embed_dim is not divisible by heads"

        self.heads = heads // self.channel_group.size()
        self.scale = (embed_dim // heads) ** -0.5
        self.config = self._resolve_ring_config(ring, config, device)

        if self.ring:
            assert spatial_group.size() > 1, (
                "Cannot perform ring attention if spatial size is 1."
            )
            self.attn_core: Callable[..., torch.Tensor] = self._ring_attn_core
        elif self.ulysses:
            assert spatial_group.size() > 1, (
                "Cannot perform ulysses attention if spatial size is 1."
            )
            if self.ulysses_intra_size:
                a = self.ulysses_intra_size
                assert (
                    spatial_group.size() % a == 0 and spatial_group.size() // a >= 2
                ), (
                    f"hierarchical ulysses needs spatial ({spatial_group.size()}) "
                    f"to split into all-to-all intra groups of {a} and an "
                    f"outer ring of >= 2; use ulysses_intra_size=0 for "
                    f"the flat scheme"
                )
                assert self.heads % a == 0, (
                    f"hierarchical ulysses scatters heads over the intra "
                    f"group: heads // channel ({self.heads}) must be divisible by "
                    f"ulysses_intra_size ({a})"
                )
                topo = get_hybrid_topology(spatial_group, a)
                self._uly_intra_group = topo.intra_group
                self._uly_inter_group = topo.inter_group
                self._uly_ring_cfg = attention_config.RingAttentionConfig(
                    device=self.device,
                    send_in_forward=attention_config.SendForward.KV,
                    save_for_backward=attention_config.SaveBackward.KV,
                    save_scope=self.ulysses_inter_save_scope,
                    send_op=attention_config.SendOp.ALLTOALL,
                    # Pinned: backward overlap needs LOCAL + SEND_DKV.
                    send_backward=attention_config.SendBackward.SEND_DKV,
                    attn_backend=self.attn_backend,
                    overlap_comm=self.ulysses_inter_overlap,
                    offload=self.ulysses_inter_offload,
                )
            else:
                assert self.heads % spatial_group.size() == 0, (
                    f"ulysses attention needs heads // channel ({self.heads}) "
                    f"divisible by spatial group size ({spatial_group.size()})"
                )
            if self.te:
                # Same per-GPU-arch kernel the ring's inner core resolves; the
                # Ulysses path calls it directly on the gathered full sequence.
                self.te_core = te_core.select_core(self.device)
            self.attn_core = self._ulysses_attn_core
        else:
            self.attn_core = self._select_attn_core()

        if channel_group.size() == 1:
            self.qkv = torch.nn.Linear(
                in_dim, embed_dim * 3, bias=False, device=device, dtype=self.dtype
            )
        else:
            self.qkv = linear.DistributedLinear(
                in_dim,
                embed_dim * 3,
                local_rank=self.channel_group.rank(),
                device=self.device,
                model_group=self.channel_group,
                row_group=self.spatial_group,
                col_group=self.channel_group,
                bias=False,
                dtype=self.dtype,
            )

        if channel_group.size() == 1:
            # No channel parallelism: plain local matmul (also allows a pure ring).
            self.head_concat = torch.nn.Linear(
                embed_dim, out_dim, bias=out_bias, device=device, dtype=self.dtype
            )
        else:
            self.head_concat = linear.DistributedLinear(
                embed_dim,
                out_dim,
                local_rank=self.channel_group.rank(),
                device=self.device,
                model_group=self.channel_group,
                row_group=self.spatial_group,
                col_group=self.channel_group,
                bias=out_bias,
                dtype=self.dtype,
            )

        if self.qk_norm:
            self.q_norm, self.k_norm = (
                torch.nn.LayerNorm(embed_dim // heads, device=device, dtype=dtype),
                torch.nn.LayerNorm(embed_dim // heads, device=device, dtype=dtype),
            )
            # The norms act on the unsharded head dimension, so they are
            # replicated over the spatial and the channel group: sum their
            # gradients over both.
            norm_sync_groups = [
                g for g in (self.spatial_group, self.channel_group) if g.size() > 1
            ]
            for norm in (self.q_norm, self.k_norm):
                for p in norm.parameters():
                    parallel.mark_replicated(p, norm_sync_groups)

        for p in self.parameters():
            # Skip already-marked params; a duplicate mark would otherwise
            # double-sum the grad.
            if parallel.replica_groups(p):
                continue
            parallel.mark_replicated(p, self.spatial_group)
        # Make replicas identical from the start, so plain DDP needs no broadcast.
        parallel.broadcast_replicas(self)

    def _resolve_ring_config(
        self,
        ring: bool,
        config: attention_config.RingAttentionConfig | None,
        device: torch.device,
    ) -> attention_config.RingAttentionConfig | None:
        """Pick the ``RingAttentionConfig`` for this module.

        The ring rebuilds K/V from a received ``x`` with a plain local matmul.
        That would skip the communication of a sharded ``qkv`` projection
        (channel group > 1) and ``k_norm``, so either one needs
        ``send_in_forward=KV`` and ``save_for_backward=KV``. The default config
        is built that way; an explicit config without them raises
        :class:`ValueError`.
        """
        if not ring:
            return config

        needs_kv = self.channel_group.size() > 1 or self.qk_norm
        if config is None:
            kwargs: dict = dict(device=device, attn_backend=self.attn_backend)
            if needs_kv:
                kwargs.update(
                    send_in_forward=attention_config.SendForward.KV,
                    save_for_backward=attention_config.SaveBackward.KV,
                )
            return attention_config.RingAttentionConfig(**kwargs)

        if needs_kv and (
            config.send_in_forward != attention_config.SendForward.KV
            or config.save_for_backward != attention_config.SaveBackward.KV
        ):
            raise ValueError(
                "ring attention with channel/tensor parallelism or qk_norm "
                f"(channel_group.size()={self.channel_group.size()}, "
                f"qk_norm={self.qk_norm}) needs send_in_forward=KV and "
                "save_for_backward=KV: the qkv projection is sharded across "
                "the channel group and/or the normalized keys only exist "
                "after the local projection, so the ring cannot recompute "
                "K/V from a circulated x chunk. Got "
                f"send_in_forward={config.send_in_forward.name}, "
                f"save_for_backward={config.save_for_backward.name}."
            )
        return config

    @staticmethod
    def _resolve_backend(attn_backend: AttnBackend) -> AttnBackend:
        """Fall back from ``TE`` to ``SDPA`` if neither TE core is importable."""
        if attn_backend is AttnBackend.TE and not JIGSAW_TE_AVAILABLE:
            warnings.warn(
                "attn_backend='te' but neither transformer_engine nor flash_attn "
                "is importable; "
                "falling back to 'sdpa'. Install one or update your container.",
                stacklevel=3,  # past __init__, to the module's construction
            )
            return AttnBackend.SDPA
        return attn_backend

    def _select_attn_core(self) -> Callable[..., torch.Tensor]:
        """Resolve ``self.attn_backend`` to its non-ring attention core."""
        match self.attn_backend:
            case AttnBackend.TE:
                # Resolved once here so nothing branches per call; see
                # `_functional.cores.te.select_core`.
                self.te_core = te_core.select_core(self.device)
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
        attn_mask: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Query, key and value.
        attn_mask : torch.Tensor, optional
            Additive (float) or keep-true (bool) mask broadcastable to
            ``(..., heads, seq_q, seq_kv)``, e.g. a per-window Swin shift
            mask. Flash rejects a non-null mask, so a masked call instead
            runs on the memory-efficient or cuDNN ``sdpa`` backend, fused
            rather than falling back to unfused math.
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
                torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        return out

    def _attn_core_math(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention with the unfused math backend.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Query, key and value.
        attn_mask : torch.Tensor, optional
            Same convention as :meth:`_attn_core`'s.
        **kwargs : torch.Tensor
            Unused.
        """
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        return out

    def _transformer_attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Attention computation on the TransformerEngine kernels.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor, ``bshd`` (``[batch, seq, heads, head_dim]``).
        k : torch.Tensor
            Keys tensor, ``bshd``.
        v : torch.Tensor
            Values tensor, ``bshd``.
        attn_mask : torch.Tensor, optional
            Not supported; a non-``None`` mask raises :class:`ValueError`. Use
            ``attn_backend="sdpa"`` (memory-efficient kernel) for masked attention.
        **kwargs : torch.Tensor
            Unused.

        Returns
        -------
        torch.Tensor
            Output of attention computation, ``[batch, seq, heads * head_dim]``.
        """
        if attn_mask is not None:
            raise ValueError(
                "attn_backend='te' does not support attn_mask: the TE kernels "
                "this core calls (fused_attn_fwd/bwd) have no fused path for an "
                "arbitrary mask, only no_mask/causal/padding attn_mask_types. "
                "Use attn_backend='sdpa' or 'math' for masked (e.g. shifted-"
                "window) attention."
            )
        # There is no dropout nn.Module here to apply its own training-mode
        # gate, so gate it the way the sdpa/math cores do.
        out, _lse, _rng_state = _TransformerEngineAttention.apply(
            q, k, v, self.scale, self.dropout_p if self.training else 0.0, self.te_core
        )
        return cast(torch.Tensor, out)

    def _ring_attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ring attention.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Query, key and value.
        x : torch.Tensor
            Local input chunk.
        attn_mask : torch.Tensor, optional
            Not supported; a non-``None`` mask raises :class:`ValueError`. Use
            ``ring=False`` for masked attention.
        **kwargs : torch.Tensor
            Unused.

        Returns
        -------
        torch.Tensor
            Attention output.
        """
        if attn_mask is not None:
            raise ValueError(
                "ring attention does not support attn_mask: K/V circulate "
                "chunk-by-chunk around the ring and _RingAttention has "
                "no per-step masking. Use ring=False for masked attention."
            )
        out = _RingAttention.apply(
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

    def _dense_bshd_core(
        self, dropout_p: float
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Build a single dense-attention call on ``bshd`` ``q``/``k``/``v``.

        The resolved TE kernel (``bshd``-native) or torch SDPA (transposed to
        ``bhsd`` around the call and back). Used as the inner core of flat
        Ulysses, over the whole gathered sequence.
        """
        if self.te:

            def core(
                qc: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor
            ) -> torch.Tensor:
                out, _lse, _rng = _TransformerEngineAttention.apply(
                    qc, kc, vc, self.scale, dropout_p, self.te_core
                )  # [b, s, h * d]
                out = cast(torch.Tensor, out)
                return cast(
                    torch.Tensor, out.unflatten(-1, (qc.shape[-2], qc.shape[-1]))
                )
        else:

            def core(
                qc: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor
            ) -> torch.Tensor:
                with torch.nn.attention.sdpa_kernel(
                    [
                        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                        torch.nn.attention.SDPBackend.MATH,
                    ]
                ):
                    out = torch.nn.functional.scaled_dot_product_attention(
                        qc.transpose(1, 2),
                        kc.transpose(1, 2),
                        vc.transpose(1, 2),
                        is_causal=False,
                        dropout_p=dropout_p,
                        scale=self.scale,
                    )
                return out.transpose(1, 2)

        return core

    def _inter_ring_core(
        self, dropout_p: float
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Build the outer ring of hierarchical Ulysses on ``bshd`` ``q``/``k``/``v``.

        Rings the projected K/V across the all-to-all groups, after the
        all-to-all within each group.
        """
        layout = self._uly_ring_cfg.core_layout
        inter_rank = dist.get_rank(self._uly_inter_group)

        def core(qc: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor) -> torch.Tensor:
            if layout == "bhsd":
                qc, kc, vc = (t.transpose(1, 2).contiguous() for t in (qc, kc, vc))
            out = _RingAttention.apply(
                qc,
                kc,
                vc,
                qc.detach(),  # dummy x: unused with send_in_forward=KV
                self.qkv.weight,  # dummy w_qkv: unused with send_in_forward=KV
                self.scale,
                dropout_p,
                self._uly_inter_group,
                inter_rank,
                self._uly_ring_cfg,
            )[0]
            out = cast(torch.Tensor, out)
            if layout == "bhsd":
                out = out.transpose(1, 2).contiguous()
            return out

        return core

    def _ulysses_attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Ulysses (all-to-all, head-parallel) sequence-parallel attention.

        ``q``/``k``/``v`` are ``bshd`` ``[b, seq // spatial, local_heads, d]``.
        Flat when ``ulysses_intra_size == 0``, hierarchical otherwise. A
        non-``None`` ``attn_mask`` raises :class:`ValueError`.
        """
        if attn_mask is not None:
            raise ValueError(
                "ulysses attention does not support attn_mask: the inner "
                "dense core built by _dense_bshd_core/_inter_ring_core "
                "takes no mask parameter. Use ulysses=False for masked "
                "attention."
            )
        from ._functional.ulysses import (
            hierarchical_ulysses_attention,
            ulysses_attention,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        qkv_stack = kwargs.get("qkv_stack")

        if self.ulysses_intra_size:
            return hierarchical_ulysses_attention(
                q,
                k,
                v,
                self._uly_intra_group,
                self._inter_ring_core(dropout_p),
                qkv_stack=qkv_stack,
            )
        return ulysses_attention(
            q,
            k,
            v,
            self.spatial_group,
            self._dense_bshd_core(dropout_p),
            qkv_stack=qkv_stack,
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for distributed multi-head attention.

        Parameters
        ----------
        x : torch.Tensor
            ``(n_batch, vars, n_lat, n_lon)``, already sharded per
            ``spatial_group`` / ``channel_group``. Extra axes between batch and
            sequence (e.g. Swin windows) are kept.
        attn_mask : torch.Tensor, optional
            Additive (float) or keep-true (bool) mask broadcastable to
            ``(..., heads, seq, seq)``. Only supported by the plain core with
            ``attn_backend`` ``"sdpa"`` or ``"math"``; otherwise raises
            :class:`ValueError`.

        Returns
        -------
        torch.Tensor
            ``(n_batch, seq_length, output_dim)``.
        """
        # Swap in the in-graph-synced replicated parameters for the whole forward;
        # they carry no mark, so the re-entered forward falls through.
        synced = parallel.synced_parameters(self)
        if synced:
            return cast(
                torch.Tensor,
                torch.func.functional_call(
                    self, synced, (x,), {"attn_mask": attn_mask}
                ),
            )

        # shape: [n_batches, spatial_tokens, channel_tokens]
        qkv = self.qkv(x)

        # TE and Ulysses take bshd; SDPA / math take bhsd.
        if self._bshd_qkv:
            qkv = rearrange(
                qkv, "... n s (three h d) -> three ... n s h d", three=3, h=self.heads
            )
        else:
            qkv = rearrange(
                qkv, "... n s (three h d) -> three ... n h s d", three=3, h=self.heads
            )

        core_dtype = attn_core_dtype(x)
        # Ulysses reuses ``qkv_stack`` to skip a re-stack; invalid after qk_norm.
        if self.qk_norm:
            q, k, v = qkv.unbind(0)
            q = self.q_norm(q).to(core_dtype)
            k = self.k_norm(k).to(core_dtype)
            v = v.to(core_dtype)
            qkv_stack = None
        else:
            qkv = qkv.to(core_dtype)
            q, k, v = qkv.unbind(0)
            qkv_stack = qkv

        out = self.attn_core(
            q, k, v, x=x, qkv_stack=qkv_stack, attn_mask=_mask_to(attn_mask, core_dtype)
        )
        # Fold ``out`` to ``... n s (h d)`` in ``x.dtype`` for ``head_concat``.
        if self._te_bshd:
            pass
        elif self.te or self.ulysses:
            out = rearrange(out, "... n s h d -> ... n s (h d)").contiguous()
        else:
            out = rearrange(out, "... n h s d -> ... n s (h d)").contiguous()
        out = out.to(x.dtype)

        out = cast(torch.Tensor, self.head_concat(out))
        # Output: [n_batch, seq_length, output_dim]
        return out

    def get_parameter_count(self) -> int:
        """Return the global number of parameters of this layer.

        Returns
        -------
        int
            Global (unsharded) count of this layer's parameters, including plain
            ``torch.nn`` children; children with ``get_parameter_count`` report
            their own. No communication.

        Notes
        -----
        This method calculates the number of parameters instead of counting the number of elements of the local parameters.
        """
        if not self.qk_norm:
            return 0

        global_heads = self.heads * self.channel_group.size()

        return 4 * (
            self.embed_dim // global_heads
        )  # 2 norms (q and k) * 2 parameters (scale and bias) * normalized_dim


class Attention(torch.nn.Module):
    """Non-distributed multi-head attention, convertible with :meth:`convert_to_jigsaw`.

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
    out_bias : bool
        Whether the output projection has a bias.
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
        out_bias: bool = True,
    ):
        """Initialize the module."""
        super().__init__()
        self.qkv = torch.nn.Linear(
            in_features=in_dim,
            out_features=3 * embed_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.head_concat = torch.nn.Linear(
            in_features=embed_dim,
            out_features=out_dim,
            bias=out_bias,
            device=device,
            dtype=dtype,
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
                torch.nn.LayerNorm(self.head_dim, device=device, dtype=dtype),
                torch.nn.LayerNorm(self.head_dim, device=device, dtype=dtype),
            )

    def _attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Query, key and value.
        attn_mask : torch.Tensor, optional
            Additive (float) or keep-true (bool) mask broadcastable to
            ``(..., heads, seq_q, seq_kv)``. See
            :meth:`DistributedAttention._attn_core` for why a masked call
            still lands on a fused backend rather than unfused math.
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
                torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

        return out

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        attn_mask : torch.Tensor, optional
            Additive (float) or keep-true (bool) mask broadcastable to
            ``q``'s ``(batch, heads, seq, seq)`` shape, e.g. a per-window
            Swin shift mask (batch here would be ``batch * windows``,
            flattened by the caller).

        Returns
        -------
        torch.Tensor
            Attention output.
        """
        qkv = self.qkv(x)
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.heads, self.head_dim)

        core_dtype = attn_core_dtype(x)
        if self.qk_norm:
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            q = self.q_norm(q).to(core_dtype)
            k = self.k_norm(k).to(core_dtype)
            v = v.to(core_dtype)
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).to(core_dtype)
        out = self.attn_core(
            q=q, k=k, v=v, x=x, attn_mask=_mask_to(attn_mask, core_dtype)
        )

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
        ulysses: bool = False,
        ulysses_intra_size: int = 0,
        ulysses_inter_save_scope: attention_config.SaveScope | str = (
            attention_config.SaveScope.LOCAL
        ),
        ulysses_inter_overlap: bool = False,
        ulysses_inter_offload: attention_config.Offload | str = (
            attention_config.Offload.DISABLED
        ),
        attn_backend: AttnBackend | str = AttnBackend.SDPA,
        config: attention_config.RingAttentionConfig | None = None,
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
            Whether to use ring attention over the rows; if ``False``, each row
            attends within its own chunk. With a column group larger than 1 or
            ``qk_norm``, the default ``config`` circulates K/V instead of ``x``.
        ulysses : bool
            Run Ulysses (all-to-all, head-parallel) sequence parallelism on the
            "row" dimension instead of the ring. Mutually exclusive with ``ring``.
        ulysses_intra_size : int
            Forwarded to :class:`DistributedAttention`.
        ulysses_inter_save_scope : SaveScope | str
            Forwarded to :class:`DistributedAttention`.
        ulysses_inter_overlap : bool
            Forwarded to :class:`DistributedAttention`.
        ulysses_inter_offload : Offload | str
            Forwarded to :class:`DistributedAttention`.
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
            qk_norm=attention.qk_norm,
            ring=ring,
            ulysses=ulysses,
            ulysses_intra_size=ulysses_intra_size,
            ulysses_inter_save_scope=ulysses_inter_save_scope,
            ulysses_inter_overlap=ulysses_inter_overlap,
            ulysses_inter_offload=ulysses_inter_offload,
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

        # The query/key norms act on the unsharded head dimension.
        if attention.qk_norm:
            instance.q_norm.load_state_dict(attention.q_norm.state_dict())
            instance.k_norm.load_state_dict(attention.k_norm.state_dict())

        # The serial weights need not agree across the spatial group.
        for p in instance.parameters():
            dist.broadcast(p.data, group_src=0, group=spatial_group)

        # Update head count to reflect local shard
        instance.heads = attention.heads // channel_size
        # embed_dim attribute: keep global value, since forward uses embed_dim // heads
        # which still gives head_dim correctly

        return instance
