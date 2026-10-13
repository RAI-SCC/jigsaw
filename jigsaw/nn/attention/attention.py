"""Distributed attention modules."""

import warnings
from typing import Any, Callable, Optional, cast

import torch
import torch.distributed as dist
from einops import rearrange

from jigsaw import parallel
from jigsaw.nn import linear

from . import config as attention_config
from .config import AttnBackend
from .functional.cores import te as te_core
from .functional.ring import _RingAttention, get_hybrid_topology

#: Whether ``attn_backend="te"`` can be run at all. Both TE paths -- the ring's
#: inner core and :class:`_TransformerEngineAttention` below -- go through
#: :mod:`.functional.cores.te`, so one probe covers both: it is false only when
#: *neither* arm is importable (no ``transformer_engine`` for cuDNN, no
#: ``flash_attn`` for flash).
JIGSAW_TE_AVAILABLE = te_core.TE_CORE_AVAILABLE

if JIGSAW_TE_AVAILABLE:

    class _TransformerEngineAttention(torch.autograd.Function):
        """Autograd wrapper around the TransformerEngine attention kernels.

        Drives :mod:`.functional.cores.te` -- the same cores the ring's inner
        step uses -- in the non-ring, ``bshd`` configuration, so
        ``attn_backend="te"`` means the same kernel whether or not ``ring`` is
        set. Which kernel that is (cuDNN on Hopper+, ``flash_attn`` on Ampere)
        is resolved once into ``self.te_core`` by
        :func:`.functional.cores.te.select_core`, then passed in as ``core``.

        Replaces a wrapper around ``te.DotProductAttention``, which is itself an
        ``autograd.Function`` -- a graph boundary that forced the old wrapper to
        recompute the whole forward under ``enable_grad`` just to get something
        to differentiate, plus an RNG snapshot/restore dance to redraw the same
        dropout mask. Calling the kernels directly gives their real backward and
        own ``rng_state`` instead: no recompute, no RNG bookkeeping, measurably
        faster fwd+bwd, with the biggest wins at short sequences where
        ``DotProductAttention``'s per-call backend-selection overhead also goes
        away.

        The softmax LSE the cores return is dropped here (the ring needs it for
        its online-softmax merge; a single un-chunked call does not) -- for free,
        since both kernels compute it regardless as their own backward's
        normaliser.

        Gives up relative to ``DotProductAttention``: FP8, bias, mask, GQA, and
        its per-shape backend dispatch -- the last costs little since jigsaw's
        problem class (non-FP8, no mask, no bias, no GQA) gets the same verdict
        from TE's selector across shapes, which is what makes the coarse
        per-architecture rule in :func:`.functional.cores.te.select_core` a
        faithful stand-in. A kernel that cannot serve a shape raises rather than
        falling back silently.

        ``torch.vmap`` support (``torch_blue``'s Bayesian ensembles) survives the
        move: see the :meth:`vmap` staticmethod.
        """

        @staticmethod
        def forward(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            scale: float,
            dropout_p: float,
            core: Any,
        ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            """Run TE attention on ``core``.

            Parameters
            ----------
            q, k, v : torch.Tensor
                Query/key/value tensors in ``bshd`` format
                (``[batch, seq, heads, head_dim]``), fp16 or bf16.
            scale : float
                Softmax scale (``head_dim ** -0.5`` for the callers here --
                ``te.DotProductAttention``'s default).
            dropout_p : float
                Attention dropout probability. Already gated on training mode by
                the caller; this function has no ``self.training`` to consult.
            core : module
                The TE core to run -- :mod:`.functional.cores.te.cudnn` or
                :mod:`.functional.cores.te.flash`, resolved once per module by
                :func:`.functional.cores.te.select_core`.

            Returns
            -------
            tuple
                ``(out, logsumexp, rng_state)``. ``out`` is
                ``[batch, seq, heads * head_dim]`` -- heads merged, matching what
                ``te.DotProductAttention(qkv_format="bshd")`` returned.
                ``logsumexp`` ``(b, h, s)`` and ``rng_state`` exist only so
                :meth:`setup_context` can hand them to :meth:`backward`; callers
                take ``[0]``.
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
            _grad_lse: Optional[torch.Tensor],
            _grad_rng: Optional[torch.Tensor],
        ) -> tuple[Optional[torch.Tensor], ...]:
            """dq/dk/dv straight from the kernel's backward -- no forward recompute."""
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
            in_dims: tuple[Optional[int], ...],
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            scale: float,
            dropout_p: float,
            core: Any,
        ) -> tuple[tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], tuple]:
            """Fold the vmapped (samples) axis into the batch, run one plain call, unfold.

            ``transformer_engine``'s kernels are opaque C++/CUDA extensions with no
            batching rule, so calling them under ``torch.vmap`` (as ``torch_blue``
            does for Bayesian sampling) fails; having a ``vmap`` staticmethod lets
            us decide what happens instead. Since attention is independent across
            the batch axis, ``[samples, batch, ...] -> [samples * batch, ...]`` is
            exact.

            It re-enters :meth:`forward` via ``apply``, so gradients route
            through :meth:`backward` -- the folded ``q/k/v`` produce the matching
            folded LSE / ``rng_state``, which is what ``fused_attn_bwd`` needs.
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

    This is the autocast dtype under autocast. Otherwise it is the input dtype, except
    that fp32 (or fp64) inputs on CUDA run the core in bf16: the flash kernels reject
    fp32, and fp16 would need loss scaling -- without it, the small gradients of a loss
    averaged over many outputs underflow to zero in the attention backward.

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


def _mask_to(
    attn_mask: Optional[torch.Tensor], dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """Cast an additive (float) mask to the core dtype; SDPA rejects a mismatch."""
    if attn_mask is not None and attn_mask.is_floating_point():
        return attn_mask.to(dtype)
    return attn_mask


class DistributedAttention(torch.nn.Module):
    """Distributed multi-head attention, parallelised over ``spatial_group``.

    Three mutually exclusive schemes, selected by ``ring`` / ``ulysses``: plain
    (each rank attends to its own local sequence shard), ring (K/V circulate
    around ``spatial_group``), or Ulysses (a single all-to-all trades sequence
    shard for head shard -- flat, or hierarchical when ``ulysses_gpus_per_node``
    is also set: intra-node Ulysses nested in an inter-node ring). See
    :meth:`__init__` for the full parameter reference.
    """

    # type hints
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
        bayesian: bool = False,
        qk_norm: bool = False,
        ring: bool = False,
        ulysses: bool = False,
        ulysses_gpus_per_node: int = 0,
        ulysses_inter_save_scope: attention_config.SaveScope | str = (
            attention_config.SaveScope.LOCAL
        ),
        ulysses_inter_overlap: bool = False,
        ulysses_inter_offload: attention_config.Offload | str = (
            attention_config.Offload.DISABLED
        ),
        attn_backend: AttnBackend | str = AttnBackend.SDPA,
        dtype: torch.dtype = torch.float32,
        config: Optional[attention_config.RingAttentionConfig] = None,
        out_bias: bool = True,
    ) -> None:
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
        bayesian : bool
            Whether the linear layers are Bayesian.
        qk_norm : bool
            Whether to normalize queries and keys.
        ring : bool
            Run ring (sequence-parallel) attention over ``spatial_group``. A
            separate switch from ``attn_backend``: the ring's inner per-chunk
            attention still uses the chosen backend.
        ulysses : bool
            Run Ulysses (all-to-all, head-parallel) sequence parallelism over
            ``spatial_group`` instead of the ring: one all-to-all trades the
            local sequence shard for a head shard, a single dense attention runs
            over the full sequence on ``heads // (channel * spatial)`` heads, and
            an all-to-all brings the result back. Mutually exclusive with
            ``ring``; needs ``heads // channel_group.size()`` divisible by
            ``spatial_group.size()``. See :mod:`.functional.ulysses`.
        ulysses_gpus_per_node : int
            When ``> 0`` and ``ulysses`` is set, run the *hierarchical* scheme
            (Transformer Engine's ``cp_comm_type="a2a+p2p"`` analogue): an
            all-to-all over an intra-node sub-group of this many consecutive
            spatial ranks, nested in a P2P ring over the inter-node sub-group
            (``spatial // ulysses_gpus_per_node`` ranks), so only ``n_nodes``
            K/V chunks ever ring over the slow fabric. Needs ``spatial``
            divisible by this into a ring of ``>= 2``, and ``heads // channel``
            divisible by it. ``0`` (default) keeps the flat single-level
            Ulysses. ``ulysses_inter_save_scope``/``ulysses_inter_overlap`` tune
            the inter-node ring.
        ulysses_inter_save_scope : SaveScope | str
            ``save_scope`` for the hierarchical scheme's inter-node K/V ring
            (only read when ``ulysses_gpus_per_node > 0``). ``LOCAL`` (default)
            keeps only this rank's own chunk and re-rings the rest over the
            slow fabric in the backward. ``GLOBAL`` keeps every K/V chunk the
            forward ring received so the backward reads them from memory, at
            one extra ``n_nodes``-chunk buffer per rank, but routes dK/dV
            through an all-reduce per backward step instead -- worthwhile only
            once ``n_nodes`` is large enough that ``LOCAL``'s re-ringing costs
            more than that all-reduce.
        ulysses_inter_overlap : bool
            Put the hierarchical scheme's inter-node K/V ring (and its backward
            dK/dV relay) on a dedicated CUDA stream with double buffering, so
            each inter-node transfer overlaps the node-local attention core --
            the ``cp_stream`` trick Transformer Engine's ``a2a+p2p`` uses. Only
            read when ``ulysses_gpus_per_node > 0``; forwarded to the inter-node
            ring's :class:`.config.RingAttentionConfig` as ``overlap_comm``
            (CUDA-only, inert otherwise). Requires
            ``ulysses_inter_save_scope=LOCAL`` to also overlap the *backward*
            (the re-ringing dK/dV relay the overlap targets); paired with
            ``GLOBAL`` only the forward K/V ring overlaps and a warning is
            emitted.
        ulysses_inter_offload : Offload | str
            ``offload`` for the hierarchical scheme's inter-node K/V ring (only
            read when ``ulysses_gpus_per_node > 0``): forwarded as-is to the
            inter-node ring's :class:`.config.RingAttentionConfig`, which is the
            same ``RingAttentionConfig`` the plain ``ring=True`` path uses --
            see :class:`.offload.NVLinkActivationOffloader`. ``CPU`` moves the
            chunk kept for backward (per ``ulysses_inter_save_scope``) to
            pinned host memory between forward and backward, trading GPU HBM
            for PCIe/NVLink traffic that the offloader's side streams overlap
            with compute. ``DISABLED`` (default) keeps it on-device.
        attn_backend : AttnBackend | str
            Attention core implementation -- ``"te"``, ``"sdpa"`` (default) or
            ``"math"``. See :class:`.config.AttnBackend`.
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
        self.ulysses = ulysses
        self.ulysses_gpn = ulysses_gpus_per_node if ulysses else 0
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
            and self.ulysses_gpn
            and self.ulysses_inter_save_scope == attention_config.SaveScope.GLOBAL
        ):
            warnings.warn(
                "ulysses_inter_overlap with ulysses_inter_save_scope=GLOBAL "
                "overlaps only the forward inter-node K/V ring: GLOBAL keeps "
                "every arrived chunk and all-reduces dK/dV, so the backward has "
                "no re-ringing relay to hide. Use ulysses_inter_save_scope=LOCAL "
                "to overlap the backward too."
            )
        if ring and ulysses:
            raise ValueError("ring and ulysses are mutually exclusive")

        self.attn_backend = self._resolve_backend(AttnBackend(attn_backend))
        self.te = self.attn_backend == AttnBackend.TE
        # Both TE paths run bshd q/k/v (``self.te`` gates the rearrange in
        # ``forward``); the Ulysses path is bshd for every backend. They differ
        # only in the *output*: the non-ring, non-ulysses TE core returns
        # ``... n s (h d)`` (heads already folded), the ring and Ulysses cores
        # return ``... n s h d``. ``_te_bshd`` marks the folded case.
        self._bshd_qkv = self.te or self.ulysses
        self._te_bshd = self.te and not ring and not ulysses

        self.heads = heads // self.channel_group.size()
        self.scale = (embed_dim // heads) ** -0.5
        self.empty = torch.empty(0, device=device)

        assert heads % self.channel_group.size() == 0, (
            "heads is not divisible by channel group size"
        )

        # Resolved here (rather than at its usual place below) so the
        # explicit-config validation inside it -- channel split and/or
        # qk_norm need send_in_forward=KV and save_for_backward=KV -- runs
        # before anything else gets built from a config that would be
        # rejected anyway.
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
            if self.ulysses_gpn:
                a = self.ulysses_gpn
                assert (
                    spatial_group.size() % a == 0 and spatial_group.size() // a >= 2
                ), (
                    f"hierarchical ulysses needs spatial ({spatial_group.size()}) "
                    f"to split into an intra-node a2a group of {a} and an "
                    f"inter-node ring of >= 2; use ulysses_gpus_per_node=0 for "
                    f"the flat scheme"
                )
                assert self.heads % a == 0, (
                    f"hierarchical ulysses scatters heads over the intra-node "
                    f"group: heads // channel ({self.heads}) must be divisible by "
                    f"ulysses_gpus_per_node ({a})"
                )
                topo = get_hybrid_topology(spatial_group, a)
                self._uly_intra_group = topo.intranode_group
                self._uly_inter_group = topo.internode_group
                self._uly_ring_cfg = attention_config.RingAttentionConfig(
                    device=self.device,
                    send_in_forward=attention_config.SendForward.KV,
                    save_for_backward=attention_config.SaveBackward.KV,
                    # LOCAL (default): keep only this rank's chunk; the backward
                    # re-rings the rest. At n_nodes == 2 that is one inter-node
                    # hop, same as GLOBAL would cost (GLOBAL keeps every arrived
                    # K/V chunk but then all-reduces dK/dV each backward step) --
                    # measured a wash there; GLOBAL is expected to pay off only
                    # at n_nodes >= 4. SendOp.ALLTOALL still runs the same p2p
                    # ring exchange, just issued as an all-to-all collective
                    # instead of individual send/recv -- measured faster here.
                    save_scope=self.ulysses_inter_save_scope,
                    send_op=attention_config.SendOp.ALLTOALL,
                    # Relay dK/dV P2P around the inter-node ring (not a global
                    # buffer + final all-reduce). Pinned, not left to the config
                    # default, because ``_ring_backward`` only takes the
                    # comm-stream backward-overlap path for LOCAL + SEND_DKV --
                    # see ``overlap_comm`` below and ``backward._ring_backward``.
                    send_backward=attention_config.SendBackward.SEND_DKV,
                    attn_backend=self.attn_backend,
                    # cp_stream-style overlap of the inter-node K/V + dK/dV
                    # transfers with the node-local cores (CUDA-only; inert
                    # otherwise). The backward half of this engages only with
                    # save_scope=LOCAL (the __init__ warning covers GLOBAL).
                    # See `RingAttentionConfig.overlap_comm`.
                    overlap_comm=self.ulysses_inter_overlap,
                    # CPU-offload the chunk `save_for_backward` keeps (per
                    # ulysses_inter_save_scope) between forward and backward --
                    # same mechanism the plain ring path uses. See
                    # NVLinkActivationOffloader.
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
                bayesian=bayesian,
                dtype=self.dtype,
            )

        if channel_group.size() == 1:
            # No channel/tensor parallelism: the output projection is a plain
            # local matmul, same as ``qkv`` above. ``DistributedLinear`` only has
            # kernels for (row, col) in {(1,1),(1,2),(2,2),(4,2)}, so this also
            # unblocks a pure ring (spatial=N, channel=1).
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
                bayesian=bayesian,
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
            # A duplicate mark would sum that gradient twice, scaling it by
            # ``spatial_group.size()``. DistributedLinear already marks its own
            # ``bias`` over its ``row_group``, which is ``spatial_group`` here,
            # so skip anything already claimed.
            if parallel.replica_groups(p):
                continue
            parallel.mark_replicated(p, self.spatial_group)
        # Make every replica identical from the start, over each group it is
        # marked with (the qk norms over the channel group too), so a plain
        # DistributedDataParallel needs no extra broadcast: one coalesced
        # broadcast per group and dtype. head_concat's bias gets broadcast twice
        # (DistributedLinear.__init__ already does its own) -- harmless.
        parallel.broadcast_replicas(self)

    def _resolve_ring_config(
        self,
        ring: bool,
        config: Optional[attention_config.RingAttentionConfig],
        device: torch.device,
    ) -> Optional[attention_config.RingAttentionConfig]:
        """Pick the ``RingAttentionConfig`` for this module.

        With channel/tensor parallelism the fused ``qkv`` projection is split
        across the channel group (a distributed ``DistributedLinear`` matmul), so
        the ring cannot rebuild K/V from a circulated ``x`` chunk with the plain
        local matmul in :func:`..functional.ring._compute_kv`. ``qk_norm`` needs
        the same thing for a different reason: the key norm is a separate
        module step applied after the projection, not fused into it, so a
        remote chunk's K/V rebuilt from its circulated ``x`` would miss it.
        Either case needs the ring to circulate the already-projected K/V
        directly -- i.e. ``send_in_forward=KV`` *and* ``save_for_backward=KV``
        (the only pairing that never calls ``_compute_kv``, forward or
        backward). A default config is built that way; an explicit one is
        required to already be.
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
                "save_for_backward=KV -- the qkv projection is sharded across "
                "the channel group and/or the normalized keys only exist "
                "after the local projection, so the ring cannot recompute "
                "K/V from a circulated x chunk. Got "
                f"send_in_forward={config.send_in_forward.name}, "
                f"save_for_backward={config.save_for_backward.name}."
            )
        return config

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

    def _select_attn_core(self) -> Callable[..., torch.Tensor]:
        """Resolve ``self.attn_backend`` to its non-ring attention core.

        The ``TE`` arm resolves *which* TE kernel this GPU gets (``self.te_core``);
        it needs no other per-module state, going straight to that core via
        :class:`_TransformerEngineAttention`, parameterised per call by
        ``self.scale`` / ``self.dropout_p``.
        """
        match self.attn_backend:
            case AttnBackend.TE:
                # cuDNN on Hopper+, flash_attn on Ampere -- resolved once here so
                # nothing branches per call. See `functional.cores.te`.
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
        attn_mask: Optional[torch.Tensor] = None,
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
        attn_mask: Optional[torch.Tensor] = None,
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
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Attention computation on the TransformerEngine kernels.

        Runs ``self.te_core`` -- cuDNN on Hopper+, ``flash_attn`` on Ampere --
        shared with the ring's inner core; see
        :class:`_TransformerEngineAttention` for why the kernels are called
        directly rather than through ``te.DotProductAttention``.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor, ``bshd`` (``[batch, seq, heads, head_dim]``).
        k : torch.Tensor
            Keys tensor, ``bshd``.
        v : torch.Tensor
            Values tensor, ``bshd``.
        attn_mask : torch.Tensor, optional
            Not supported: ``fused_attn_fwd``/``fused_attn_bwd`` (what
            :mod:`.functional.cores.te` calls) only implement
            ``attn_mask_type`` in ``{no_mask, causal, padding, ...}``, no
            arbitrary mask. Raises rather than silently taking TE's unfused
            detour; use ``attn_backend="sdpa"`` for masked attention instead.
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
                "arbitrary mask -- only no_mask/causal/padding attn_mask_types. "
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
        attn_mask: Optional[torch.Tensor] = None,
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
            Not supported: K/V circulate chunk-by-chunk around the ring and
            :class:`.functional.ring._RingAttention` has no per-step
            masking. Raises rather than silently ignoring it; use
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
        Ulysses -- over the whole gathered sequence.
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

    def _inter_node_ring_core(
        self, dropout_p: float
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Build a P2P ring over the inter-node group on ``bshd`` ``q``/``k``/``v``.

        The inner core of the hierarchical Ulysses scheme: after the intra-node
        all-to-all, ``q``/``k``/``v`` cover one node's worth of sequence on
        ``local_heads // A`` heads, and this rings the (already-projected) K/V
        around the ``n_nodes`` inter-node peers via :class:`_RingAttention`
        with ``send_in_forward = save_for_backward = KV`` (so it never rebuilds
        K/V from ``x`` -- the dummy ``x`` / ``w_qkv`` below are unused).
        ``_uly_ring_cfg.core_layout`` is ``bshd`` for the TE core, ``bhsd``
        otherwise; transpose around the call for the latter.
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
                qc.detach(),  # dummy x -- unused with send_in_forward=KV
                self.qkv.weight,  # dummy w_qkv -- unused with send_in_forward=KV
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
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Ulysses (all-to-all, head-parallel) sequence-parallel attention.

        ``q``/``k``/``v`` are ``bshd`` ``[b, seq // spatial, local_heads, d]``.

        Flat (``ulysses_gpus_per_node == 0``): one all-to-all over the whole
        spatial group -> ``[b, seq, local_heads // spatial, d]``, one dense
        attention, one all-to-all back.

        Hierarchical (``> 0``, the ``a2a+p2p`` analogue): an all-to-all over the
        intra-node sub-group -> ``[b, seq // n_nodes, local_heads // A, d]``,
        then a P2P ring over the inter-node sub-group, then an all-to-all back.

        ``attn_mask`` is not supported: the inner dense core takes no mask
        parameter. Raises rather than silently ignoring it.
        """
        if attn_mask is not None:
            raise ValueError(
                "ulysses attention does not support attn_mask: the inner "
                "dense core built by _dense_bshd_core/_inter_node_ring_core "
                "takes no mask parameter. Use ulysses=False for masked "
                "attention."
            )
        from .functional.ulysses import (
            hierarchical_ulysses_attention,
            ulysses_attention,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        qkv_stack = kwargs.get("qkv_stack")

        if self.ulysses_gpn:
            return hierarchical_ulysses_attention(
                q,
                k,
                v,
                self._uly_intra_group,
                self._inter_node_ring_core(dropout_p),
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
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for distributed multi-head attention.

        Projects ``x`` to Q/K/V, dispatches to the plain, ring, or Ulysses
        attention core per the constructor's config, and concatenates heads
        back to the output projection.

        Parameters
        ----------
        x : torch.Tensor
            ``(n_batch, vars, n_lat, n_lon)``, already sharded per
            ``spatial_group`` / ``channel_group``. A leading extra axis
            between the batch and the sequence (e.g. Swin's window axis) is
            fine too -- ``qkv``'s rearrange keeps whatever sits between the
            leading ``...`` batch dims and the trailing ``(three h d)``.
        attn_mask : torch.Tensor, optional
            Additive (float) or keep-true (bool) mask broadcastable to
            ``q``'s ``(..., heads, seq, seq)`` shape, e.g. a per-window Swin
            shift mask. Only the plain (non-ring, non-Ulysses) core supports
            it, and only with ``attn_backend`` in ``{"sdpa", "math"}``;
            ``"te"``, ``ring=True`` and ``ulysses=True`` raise ``ValueError``
            instead -- see each core's docstring.

        Returns
        -------
        torch.Tensor
            ``(n_batch, seq_length, output_dim)``.
        """
        # The replicated parameters -- including those of the plain
        # ``nn.Linear`` / ``nn.LayerNorm`` submodules, whose forward jigsaw does
        # not own -- are summed over their groups in-graph (see
        # :mod:`jigsaw.parallel`). ``functional_call`` swaps the synced tensors
        # in for the whole forward, so every use (``self.qkv(x)``, the ring
        # core's ``self.qkv.weight``) sees them. They carry no mark, so the
        # re-entered ``forward`` finds nothing left to sync and falls through,
        # and a DistributedLinear submodule does not sync its bias a second time.
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

        # TE cores (ring and non-ring) are bshd-native -- feed them ``n s h d`` so
        # nothing transposes per chunk; the Ulysses path is bshd end to end and
        # transposes only around a non-TE inner kernel. SDPA / math want ``n h s d``.
        if self._bshd_qkv:
            qkv = rearrange(
                qkv, "... n s (three h d) -> three ... n s h d", three=3, h=self.heads
            )
        else:
            qkv = rearrange(
                qkv, "... n s (three h d) -> three ... n h s d", three=3, h=self.heads
            )

        core_dtype = attn_core_dtype(x)
        # ``qkv_stack`` -- the contiguous ``[3, ... n s h d]`` tensor ``q/k/v`` are
        # unbound from -- is handed to the Ulysses core so it feeds the
        # sequence<->head all-to-all directly instead of re-``torch.stack``-ing
        # its three views (a full [3, b, s, h, d] copy per forward, and its
        # transpose in the backward). Only when ``qk_norm`` is off; the norm
        # rewrites q/k after the unbind, so the stack no longer matches.
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
        # Fold ``out`` to ``... n s (h d)`` in ``x.dtype`` before ``head_concat``
        # (which holds model-dtype weights). The non-ring TE core already returns
        # ``... n s (h d)``; the ring TE core and the Ulysses core return bshd
        # ``... n s h d``; every other core returns bhsd ``... n h s d``.
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
            dtype=dtype,
        )
        self.head_concat = torch.nn.Linear(
            in_features=embed_dim,
            out_features=out_dim,
            bias=False,
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
        attn_mask: Optional[torch.Tensor] = None,
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
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
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
        ulysses_gpus_per_node: int = 0,
        ulysses_inter_save_scope: attention_config.SaveScope | str = (
            attention_config.SaveScope.LOCAL
        ),
        ulysses_inter_overlap: bool = False,
        ulysses_inter_offload: attention_config.Offload | str = (
            attention_config.Offload.DISABLED
        ),
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
            ``config`` that needs a column group of size 1 and no ``qk_norm``,
            see :class:`DistributedAttention`. Otherwise, each row attends within
            its own chunk.
        ulysses : bool
            Run Ulysses (all-to-all, head-parallel) sequence parallelism on the
            "row" dimension instead of the ring. Mutually exclusive with ``ring``.
        ulysses_gpus_per_node, ulysses_inter_save_scope, ulysses_inter_overlap, ulysses_inter_offload
            Forwarded to :class:`DistributedAttention`. See its docstring.
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
            ulysses=ulysses,
            ulysses_gpus_per_node=ulysses_gpus_per_node,
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
