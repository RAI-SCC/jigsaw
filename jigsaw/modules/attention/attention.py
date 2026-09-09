"""Distributed attention modules."""
import warnings
from typing import Any, Callable, Optional, cast

import torch
import torch.distributed as dist
from einops import rearrange

from jigsaw import hooks
from jigsaw.modules import linear

from . import config as attention_config
from .config import AttnBackend
from .functional.cores import te as te_core
from .functional.functional import RingAttentionFunction

#: Whether ``attn_backend="te"`` can be run at all. Both TE paths -- the ring's
#: inner core and :class:`TransformerEngineAttention` below -- go through
#: :mod:`.functional.cores.te`, so one probe covers both: it is false only when
#: *neither* arm is importable (no ``transformer_engine`` for cuDNN, no
#: ``flash_attn`` for flash).
JIGSAW_TE_AVAILABLE = te_core.TE_CORE_AVAILABLE

if JIGSAW_TE_AVAILABLE:

    class TransformerEngineAttention(torch.autograd.Function):
        """Autograd wrapper around the TransformerEngine attention kernels.

        Drives :mod:`.functional.cores.te` -- the same cores the ring's inner
        step uses -- in the non-ring, ``bshd`` configuration, so
        ``attn_backend="te"`` means the same kernel whether or not ``ring`` is
        set. Which kernel that is depends on the GPU (cuDNN on Hopper+,
        ``flash_attn`` on Ampere) and is resolved once into ``self.te_core`` by
        :func:`.functional.cores.te.select_core`, then passed in as ``core``.

        Replaces a wrapper around ``te.DotProductAttention``. That module is an
        ``autograd.Function`` internally, hence a graph boundary, so the wrapper
        had to recompute the whole forward under ``enable_grad`` just to get a
        graph to differentiate -- fwd, fwd, bwd per training step -- plus a
        CPU/CUDA RNG snapshot/restore dance so the recompute redrew the
        forward's dropout mask. Calling the kernels directly gives their real
        backward and their own ``rng_state`` instead: no recompute, no RNG
        bookkeeping. Measured on an A100 (b=2, h=8, d=64, bf16), holding the
        kernel fixed: 1.2x-2.0x faster fwd+bwd across s=512..8192, the wider
        margins at short sequences where ``DotProductAttention``'s per-call
        Python (its ``AttentionParams`` backend-selection cache) also disappears.

        The softmax LSE the cores return is dropped here (the ring needs it for
        its online-softmax merge; a single un-chunked call does not). That costs
        nothing: both kernels compute the LSE regardless as the softmax
        normaliser and write it out for their own backward.

        What is given up relative to ``DotProductAttention``: FP8, bias, mask and
        GQA support, and its per-*shape* backend dispatch. The latter costs
        little in practice -- for jigsaw's problem class (non-FP8, no mask, no
        bias, no GQA) TE's selector returns the same verdict for every shape
        probed on an A100, which is what makes the coarse per-architecture rule
        in :func:`.functional.cores.te.select_core` a faithful stand-in. Where a
        kernel cannot serve a shape at all, the core raises rather than falling
        back silently.

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
            q, k, v: torch.Tensor
                Query/key/value tensors in ``bshd`` format
                (``[batch, seq, heads, head_dim]``), fp16 or bf16.
            scale: float
                Softmax scale (``head_dim ** -0.5`` for the callers here --
                ``te.DotProductAttention``'s default).
            dropout_p: float
                Attention dropout probability. Already gated on training mode by
                the caller; this function has no ``self.training`` to consult.
            core: module
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
            out, lse, rng_state = core.forward(
                q, k, v, scale, dropout_p, layout="bshd"
            )
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
            Forward and grad are checked against a per-member Python loop by
            ``tests/modules/attention/test_te_vmap_bayesian.py::TestTransformerEngineAttentionVmap``
            (``te_required`` + ``gpu_required``).
            """
            q_bdim, k_bdim, v_bdim = in_dims[0], in_dims[1], in_dims[2]

            # Move the vmapped dimension to the front for each tensor.
            q = q.movedim(q_bdim, 0) if q_bdim is not None else q.expand(info.batch_size, *q.shape)
            k = k.movedim(k_bdim, 0) if k_bdim is not None else k.expand(info.batch_size, *k.shape)
            v = v.movedim(v_bdim, 0) if v_bdim is not None else v.expand(info.batch_size, *v.shape)

            samples = q.shape[0]
            # q/k/v now have shape [samples, batch, seq, heads, head_dim].
            # Fold samples into the batch dimension so TE sees a plain bshd tensor.
            q = q.flatten(0, 1)
            k = k.flatten(0, 1)
            v = v.flatten(0, 1)

            out, lse, rng_state = TransformerEngineAttention.apply(
                q, k, v, scale, dropout_p, core
            )

            # Unfold the samples dimension back out to the front.
            out = out.unflatten(0, (samples, -1))
            lse = lse.unflatten(0, (samples, -1))
            return (out, lse, rng_state), (0, 0, None)


def attn_core_dtype(x: torch.Tensor) -> torch.dtype:
    """Dtype to run the attention core (q/k/v / SDPA / TE) in.

    * Under autocast: the autocast dtype, so an ``autocast(float16)`` region
      stays fp16 instead of being silently forced to bf16.
    * Otherwise on CUDA: bf16, so the SDPA / TE flash kernel stays eligible --
      it rejects fp32 -- instead of dropping to the slower tf32/fp32 math path.
    * On CPU (no flash kernel): keep the incoming dtype; the SDPA math backend
      handles fp32 directly.

    Shared by :class:`DistributedAttention` and the serial :class:`Attention` so
    ``convert_to_jigsaw`` parity holds.
    """
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    if x.is_cuda:
        return torch.bfloat16
    return x.dtype


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
        ring: bool = False,
        ulysses: bool = False,
        attn_backend: AttnBackend | str = AttnBackend.SDPA,
        dtype: torch.dtype = torch.float32,
        config: Optional[attention_config.RingAttentionConfig] = None,
    ):
        """

        Torch Module to perform distributed multi-headed attention.

        x_features: int
            global number of x features
        embed_dim: int
            global embedding dimension
        output_dim: int
            global
        ring: bool
            Run ring (sequence-parallel) attention over ``spatial_group``. A
            separate switch from ``attn_backend``: the ring's inner per-chunk
            attention still uses the chosen backend.
        ulysses: bool
            Run Ulysses (all-to-all, head-parallel) sequence parallelism over
            ``spatial_group`` instead of the ring: one all-to-all trades the
            local sequence shard for a head shard, a single dense attention runs
            over the full sequence on ``heads // (channel * spatial)`` heads, and
            an all-to-all brings the result back. Mutually exclusive with
            ``ring``; needs ``heads // channel_group.size()`` divisible by
            ``spatial_group.size()``. See :mod:`.functional.ulysses`.
        attn_backend: AttnBackend | str
            Attention core implementation -- ``"te"``, ``"sdpa"`` (default) or
            ``"math"``. See :class:`.config.AttnBackend`.
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

        if self.ring:
            assert spatial_group.size() > 1, (
                "Cannot perform ring attention if spatial size is 1."
            )
            self.attn_core = self._ring_attn_core
        elif self.ulysses:
            assert spatial_group.size() > 1, (
                "Cannot perform ulysses attention if spatial size is 1."
            )
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
                dtype=self.dtype,
            )

        if channel_group.size() == 1:
            # No channel/tensor parallelism: the output projection is a plain
            # local matmul, same as ``qkv`` above. ``DistributedXWT`` only has
            # kernels for (row, col) in {(1,1),(1,2),(2,2),(4,2)}, so this also
            # unblocks a pure ring (spatial=N, channel=1).
            self.head_concat = torch.nn.Linear(
                embed_dim, out_dim, bias=True, device=device, dtype=self.dtype
            )
        else:
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
                dtype=self.dtype,
            )

        # sync up up spatial parameters
        for p in self.parameters():
            dist.broadcast(
                p.data, group_src=0, group=self.spatial_group, async_op=False
            )
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

        self.config = self._resolve_ring_config(ring, config, device)

    def _resolve_ring_config(
        self,
        ring: bool,
        config: Optional[attention_config.RingAttentionConfig],
        device: torch.device,
    ) -> Optional[attention_config.RingAttentionConfig]:
        """Pick the ``RingAttentionConfig`` for this module.

        With channel/tensor parallelism the fused ``qkv`` projection is split
        across the channel group (a distributed ``DistributedXWT`` matmul), so
        the ring cannot rebuild K/V from a circulated ``x`` chunk with the plain
        local matmul in :func:`..functional.functional._compute_kv`. The ring
        must instead circulate the already-projected K/V directly -- i.e.
        ``send_in_forward=KV`` *and* ``save_for_backward=KV`` (the only pairing
        that never calls ``_compute_kv``, forward or backward). A default config
        is built that way; an explicit one is required to already be.
        """
        if not ring:
            return config

        channel_parallel = self.channel_group.size() > 1
        if config is None:
            kwargs: dict = dict(device=device, attn_backend=self.attn_backend)
            if channel_parallel:
                kwargs.update(
                    send_in_forward=attention_config.SendForward.KV,
                    save_for_backward=attention_config.SaveBackward.KV,
                )
            return attention_config.RingAttentionConfig(**kwargs)

        if channel_parallel and (
            config.send_in_forward != attention_config.SendForward.KV
            or config.save_for_backward != attention_config.SaveBackward.KV
        ):
            raise ValueError(
                "ring attention with channel/tensor parallelism "
                f"(channel_group.size()={self.channel_group.size()}) needs "
                "send_in_forward=KV and save_for_backward=KV -- the qkv "
                "projection is sharded across the channel group, so the ring "
                "cannot recompute K/V from a circulated x chunk. Got "
                f"send_in_forward={config.send_in_forward.name}, "
                f"save_for_backward={config.save_for_backward.name}."
            )
        return config

    @staticmethod
    def _resolve_backend(attn_backend: AttnBackend) -> AttnBackend:
        """Downgrade ``TE`` to ``SDPA`` when transformer_engine is not installed.

        Deliberately asymmetric with :class:`~.config.RingAttentionConfig`, which
        raises instead: a plain (non-ring) ``DistributedAttention`` stays usable
        outside the container.
        """
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
        :class:`TransformerEngineAttention`, parameterised per call by
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
                q, k, v, is_causal=False,
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
        """Unfused ``softmax(QK^T)V`` reference core (``attn_backend='math'``).

        Forces the MATH SDPA backend, so it runs for any shape / dtype / device.
        """
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False,
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
        """
        Attention computation on the TransformerEngine kernels.

        Runs ``self.te_core`` -- cuDNN on Hopper+, ``flash_attn`` on Ampere --
        shared with the ring's inner core; see
        :class:`TransformerEngineAttention` for why the kernels are called
        directly rather than through ``te.DotProductAttention``.

        Parameters
        ----------
        q: torch.Tensor
            Query tensor, ``bshd`` (``[batch, seq, heads, head_dim]``).
        k: torch.Tensor
            Keys tensor, ``bshd``.
        v: torch.Tensor
            Values tensor, ``bshd``.
        kwargs:
            Additional optional parameters.

        Returns
        -------
        out: torch.Tensor
            Output of attention computation, ``[batch, seq, heads * head_dim]``.

        """
        # There is no dropout nn.Module here to apply its own training-mode
        # gate, so gate it the way the sdpa/math cores do.
        out, _lse, _rng_state = TransformerEngineAttention.apply(
            q, k, v, self.scale, self.dropout_p if self.training else 0.0, self.te_core
        )
        return cast(torch.Tensor, out)

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
        out = RingAttentionFunction.apply(
            q,
            k,
            v,
            x,
            self.qkv.weight,
            self.scale,
            self.dropout_p if self.training else 0.0,
            self.spatial_group,
            self.local_rank,
            self.config,
        )[0]
        out = cast(torch.Tensor, out)
        return out

    def _ulysses_attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Ulysses (all-to-all, head-parallel) sequence-parallel attention.

        ``q``/``k``/``v`` are ``bshd`` ``[b, seq // spatial, local_heads, d]``.
        :func:`.functional.ulysses.ulysses_attention` all-to-alls them to
        ``[b, seq, local_heads // spatial, d]``, runs one dense attention over
        the full sequence, and all-to-alls the result back. The inner kernel is
        the same backend the module was built with -- the resolved TE kernel
        (``bshd``-native) or SDPA (transposed to ``bhsd`` around the call).
        """
        from .functional.ulysses import ulysses_attention

        dropout_p = self.dropout_p if self.training else 0.0

        if self.te:

            def core(
                qc: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor
            ) -> torch.Tensor:
                out, _lse, _rng = TransformerEngineAttention.apply(
                    qc, kc, vc, self.scale, dropout_p, self.te_core
                )  # [b, s, h * d]
                return out.unflatten(-1, (qc.shape[-2], qc.shape[-1]))
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

        return ulysses_attention(q, k, v, self.spatial_group, core)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for distributed multi-head attention.

        x: (n_batch, vars, n_lat, n_lon)
        """
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
        if self.qk_norm:
            q, k, v = qkv.unbind(0)
            q = self.q_norm(q).to(core_dtype)
            k = self.k_norm(k).to(core_dtype)
            v = v.to(core_dtype)
        else:
            q, k, v = qkv.to(core_dtype).unbind(0)

        out = self.attn_core(q, k, v, x=x)
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
                torch.nn.RMSNorm(self.head_dim),
                torch.nn.RMSNorm(self.head_dim),
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
                q, k, v, is_causal=False,
                dropout_p=self.dropout_p if self.training else 0.0,
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
        ulysses: bool = False,
        attn_backend: AttnBackend | str = AttnBackend.SDPA,
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
        ulysses: bool
            Run Ulysses (all-to-all, head-parallel) sequence parallelism on the
            "row" dimension instead of the ring. Mutually exclusive with ``ring``.
        attn_backend: AttnBackend | str
            Attention core implementation (``"te"`` / ``"sdpa"`` / ``"math"``);
            forwarded to :class:`DistributedAttention`. Composes with ``ring``.
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
            ring=ring,
            ulysses=ulysses,
            attn_backend=attn_backend,
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
        if ring:
            instance.attn_core = DistributedAttention._ring_attn_core  # type: ignore[assignment]
        elif ulysses:
            instance.attn_core = instance._ulysses_attn_core  # type: ignore[assignment]

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

        # Update head count to reflect local shard
        instance.heads = heads_per_shard
        # embed_dim attribute: keep global value, since forward uses embed_dim // heads
        # which still gives head_dim correctly

        return instance
