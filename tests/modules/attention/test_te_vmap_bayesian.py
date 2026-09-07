"""``torch.vmap`` / Bayesian-ensemble coverage for the TE attention backend.

``torch_blue`` trains a Bayesian ensemble by giving every parameter a leading
"members" axis and running the model under ``torch.vmap``. The attention path
was never exercised under ``vmap`` -- these tests pin it, at increasing levels
of integration:

1. :class:`TestTransformerEngineAttentionVmap` -- the
   :class:`~jigsaw.modules.attention.attention.TransformerEngineAttention`
   ``autograd.Function`` alone. It is the only piece that *has* to route through
   a custom Function under vmap (``te``'s fused kernels register no batching
   rule); its ``vmap`` staticmethod folds the members axis into the batch,
   runs one un-vmapped call, and unfolds. Forward + grad vs a Python loop.
   **Passes.**
2. :class:`TestDistributedAttentionPlain` -- a plain (no vmap, no ensemble)
   ``DistributedAttention`` forward+backward on a channel-parallel mesh, sdpa
   and te. **Passes.** Regression guards for two fixes: ``forward`` now casts
   the bf16 core output back to the model dtype on the ``_te_bshd`` branch too,
   and ``fused_xw_xtw_twoway`` makes ``grad_output`` contiguous before its
   ``all_to_all``.
3. :class:`TestDistributedAttentionEnsembleVmap` -- a full non-ring
   ``DistributedAttention`` under ``vmap(functional_call(...))`` with stacked
   ensemble parameters, vs a per-member loop, sdpa and te. **Forward passes**
   for both backends. **Grad xfails** for both: the te core's
   ``TransformerEngineAttention.backward`` recomputes the fused kernel under
   ``enable_grad`` + ``requires_grad_()`` (forbidden inside a functorch
   transform), and the sdpa path's ensemble ``qkv.weight`` grad drifts because
   functorch's generic batched fallback for the ``_c10d_functional``
   collectives in ``XWT.backward`` issues one un-lockstepped collective per
   ensemble member.
4. :class:`TestRingVmap` -- ``torch.vmap`` over a ``ring=True``
   ``DistributedAttention``. ``RingAttentionFunction.vmap`` folds the members
   axis into the batch, runs one un-vmapped ring pass and unfolds. Forward
   parity vs a per-member loop, sdpa and te; a stacked-parameter (vmapped
   weight) ensemble is explicitly rejected -- the ring backward is all P2P
   collectives with no batching rule.

Dropout is 0 throughout: the members-folded single call and the per-member loop
would otherwise draw different masks and never match.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from torch.func import functional_call, grad, stack_module_state, vmap

from jigsaw.modules.attention import attention
from tests.parity import assert_close

# TE fused attention needs fp16/bf16 and (on CUDA) the attn core runs bf16
# regardless of backend -- so vmap-vs-loop parity is a bf16 comparison.
_FWD_TOL = dict(atol=2e-2, rtol=2e-2)
_GRAD_TOL = dict(atol=3e-2, rtol=3e-2)  # grads flow through the bf16 core


# --------------------------------------------------------------------------- #
# 1. TransformerEngineAttention (the autograd.Function) under torch.vmap
# --------------------------------------------------------------------------- #
@pytest.mark.gpu_required
@pytest.mark.te_required
class TestTransformerEngineAttentionVmap:
    """The wrapper's ``vmap`` staticmethod folds members into batch and unfolds."""

    E, B, H, S, D = 3, 2, 4, 16, 32

    def _attn(self):
        import transformer_engine.pytorch as te

        a = te.DotProductAttention(
            num_attention_heads=self.H,
            kv_channels=self.D,
            num_gqa_groups=self.H,
            qkv_format="bshd",
            attn_mask_type="no_mask",
            attention_dropout=0.0,
        )
        a.eval()
        return a

    def _qkv(self, device, requires_grad=False):
        torch.manual_seed(0)
        return [
            torch.randn(
                self.E, self.B, self.S, self.H, self.D,
                device=device, dtype=torch.bfloat16, requires_grad=requires_grad,
            )
            for _ in range(3)
        ]

    def test_forward_matches_loop(self, device: torch.device) -> None:
        from jigsaw.modules.attention.attention import (
            TransformerEngineAttention,
            _capture_rng_state,
        )

        attn = self._attn()
        qe, ke, ve = self._qkv(device)
        rng_state = _capture_rng_state(device)

        def f(q, k, v):
            return TransformerEngineAttention.apply(q, k, v, attn, rng_state)

        got = vmap(f)(qe, ke, ve)
        assert got.shape == (self.E, self.B, self.S, self.H * self.D)

        # ``.clone()`` per member: TE's qkv-layout detector rejects a bare
        # ``qe[i]`` view (nonzero storage offset / cross-member pointer math).
        ref = torch.stack(
            [f(qe[i].clone(), ke[i].clone(), ve[i].clone()) for i in range(self.E)]
        )
        torch.testing.assert_close(got, ref, **_FWD_TOL)

    def test_grad_matches_loop(self, device: torch.device) -> None:
        from jigsaw.modules.attention.attention import (
            TransformerEngineAttention,
            _capture_rng_state,
        )

        attn = self._attn()
        qe, ke, ve = self._qkv(device, requires_grad=True)
        rng_state = _capture_rng_state(device)

        def f(q, k, v):
            return TransformerEngineAttention.apply(q, k, v, attn, rng_state)

        vmap(f)(qe, ke, ve).float().pow(2).sum().backward()
        got = (qe.grad, ke.grad, ve.grad)

        ref_members = []
        for i in range(self.E):
            qi, ki, vi = (
                t[i].detach().clone().requires_grad_(True) for t in (qe, ke, ve)
            )
            f(qi, ki, vi).float().pow(2).sum().backward()
            ref_members.append((qi.grad, ki.grad, vi.grad))
        ref = tuple(
            torch.stack([m[j] for m in ref_members]) for j in range(3)
        )

        for name, g, r in zip("qkv", got, ref):
            assert g is not None, f"no grad for {name}"
            torch.testing.assert_close(g, r, **_GRAD_TOL)


# --------------------------------------------------------------------------- #
# helpers for the full-module tests
# --------------------------------------------------------------------------- #
_IN = _EMBED = _OUT = 64
_HEADS = 4


def _build_attn(mesh: dict, device: torch.device, backend: str, bayesian: bool,
                ring: bool = False):
    model = attention.DistributedAttention(
        in_dim=_IN,
        embed_dim=_EMBED,
        out_dim=_OUT,
        heads=_HEADS,
        dropout_p=0.0,
        local_rank=mesh["col"].rank(),
        device=device,
        spatial_group=mesh["row"],
        channel_group=mesh["col"],
        bayesian=bayesian,
        qk_norm=False,
        ring=ring,
        attn_backend=backend,
    )
    model.eval()
    return model


def _local_x(mesh: dict, device: torch.device, *, members: int | None = None):
    chan_local = _IN // mesh["col"].size()
    torch.manual_seed(1000 + dist.get_rank())
    shape = (2, 16, chan_local) if members is None else (members, 2, 16, chan_local)
    return torch.randn(*shape, device=device)


# --------------------------------------------------------------------------- #
# 2. plain (no vmap) DistributedAttention on the TE core, channel-parallel mesh
# --------------------------------------------------------------------------- #
@pytest.mark.gpu_required
class TestDistributedAttentionPlain:
    """A non-ring ``DistributedAttention`` on a channel-parallel mesh, no vmap,
    no ensemble -- the simplest thing that must work before any of the
    vmap/Bayesian story does.

    Regression guards for two fixes:

    * core-dtype: the TE core runs in bf16 on CUDA (see ``attn_core_dtype``)
      while ``head_concat`` holds fp32 master weights. ``forward`` used to fold
      the core output back to ``x.dtype`` only on the ``not self._te_bshd``
      branch, so the ``_te_bshd`` (te + non-ring) path fed bf16 into
      ``head_concat`` and ``DistributedXWT`` raised ``BFloat16 != float``.
    * grad-input contiguity: ``fused_xw_xtw_twoway`` sent ``grad_output``
      straight into ``all_to_all``, which rejects the strided gradient autograd
      hands back through the attention rearranges (``Tensors must be
      contiguous``). Hit ``sdpa`` and ``te`` alike.
    """

    @pytest.mark.dist(4)
    @pytest.mark.te_required
    def test_te_forward_channel_parallel(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        model = _build_attn(process_mesh_dict_2x2, device, "te", bayesian=False)
        x = _local_x(process_mesh_dict_2x2, device)
        y = model(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert torch.isfinite(y).all()

    def _check_backward(self, mesh, device, backend):
        model = _build_attn(mesh, device, backend, bayesian=False)
        x = _local_x(mesh, device).requires_grad_(True)
        model(x).sum().backward()
        assert x.grad is not None
        assert model.head_concat.weight.grad is not None

    @pytest.mark.dist(4)
    def test_sdpa_backward_channel_parallel(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        self._check_backward(process_mesh_dict_2x2, device, "sdpa")

    @pytest.mark.dist(4)
    @pytest.mark.te_required
    def test_te_backward_channel_parallel(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        self._check_backward(process_mesh_dict_2x2, device, "te")


def _copy_weights(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    with torch.no_grad():
        for (_, p_dst), (_, p_src) in zip(
            dst.named_parameters(), src.named_parameters(), strict=True
        ):
            p_dst.copy_(p_src)


# --------------------------------------------------------------------------- #
# 3. full DistributedAttention under vmap (Bayesian ensemble)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu_required
class TestDistributedAttentionEnsembleVmap:
    """A non-ring ``DistributedAttention`` under ``vmap(functional_call(...))``
    with an ensemble (leading members axis) of stacked parameters -- the shape
    ``torch_blue`` drives for Bayesian sampling.

    Reference is a per-member Python loop over ``bayesian=False`` modules holding
    the same weights (``bayesian`` only changes a call-time weight reshape, not
    the values), so a mismatch is the vmap members-fold, not the sharding.

    **Forward** works for both ``sdpa`` and ``te`` once the core-dtype and
    grad-input-contiguity fixes are in (see :class:`TestDistributedAttentionPlain`).

    **Grad** needs several things that a distributed backward under
    ``vmap(grad(...))`` does not get for free:

    * ``config.reshape_weights`` / ``grad_sum`` reduce relative to the trailing
      weight axes (``unsqueeze(-3)`` / ``sum(dim=-3)``) so they are correct
      whether or not ``vmap`` has peeled the members axis.
    * ``fused_xw_xtw_twoway`` swaps ``grad_output`` with a vmap-batchable
      ``all_gather`` (``dist.all_to_all`` has no batching rule), and
      ``get_preallocated_buffer`` hands back a fresh tensor rather than mutating
      its module-level cache (forbidden on a captured tensor under ``grad``).
    * ``XWT.backward`` performs the spatial-group weight/bias grad all-reduce
      itself -- the ``post_accumulate_grad_hook`` that normally does it never
      fires under ``functional_call``.

    The reference's ``head_concat.bias`` grad is scaled down by the spatial
    size: ``DistributedAttention`` registers a *second*, redundant spatial
    grad-sync hook on every parameter on top of the one ``DistributedXWT``
    already puts on its bias, so the eager reference reduces that bias grad
    twice.
    """

    E = 2

    def _build(self, mesh, device, backend):
        """Ensemble of ``bayesian=True`` models + stacked state + a
        weight-matched ``bayesian=False`` per-member reference set + inputs."""
        b_models = [
            _build_attn(mesh, device, backend, bayesian=True) for _ in range(self.E)
        ]
        params, buffers = stack_module_state(b_models)
        ref_models = [
            _build_attn(mesh, device, backend, bayesian=False) for _ in range(self.E)
        ]
        for rm, bm in zip(ref_models, b_models, strict=True):
            _copy_weights(rm, bm)
        x_e = _local_x(mesh, device, members=self.E)
        return b_models[0], params, buffers, ref_models, x_e

    def _fwd(self, mesh, device, backend):
        base, params, buffers, ref_models, x_e = self._build(mesh, device, backend)
        got = vmap(
            lambda p, b, x: functional_call(base, (p, b), (x,))
        )(params, buffers, x_e)
        ref = torch.stack(
            [ref_models[i](x_e[i].contiguous()) for i in range(self.E)]
        )
        return got, ref

    def _grad(self, mesh, device, backend):
        base, params, buffers, ref_models, x_e = self._build(mesh, device, backend)

        def member_loss(p, b, x):
            return functional_call(base, (p, b), (x,)).float().pow(2).sum()

        got = vmap(grad(member_loss))(params, buffers, x_e)

        # The flash SDPA kernel has no functorch batching rule, so the vmap path
        # runs the math attention core (see ``DistributedAttention._attn_core``);
        # point the reference at the same core so this stays a test of the vmap
        # members-fold, not of flash-vs-math bf16 drift.
        if backend == "sdpa":
            for rm in ref_models:
                rm.attn_core = rm._attn_core_math

        ref_list = []
        for i in range(self.E):
            rm = ref_models[i]
            rm.zero_grad(set_to_none=True)
            rm(x_e[i].contiguous()).float().pow(2).sum().backward()
            ref_list.append({n: p.grad for n, p in rm.named_parameters()})
        ref = {k: torch.stack([r[k] for r in ref_list]) for k in got}

        # ``DistributedAttention`` blanket-registers a spatial grad-sync hook on
        # every parameter, on top of the one ``DistributedXWT`` already puts on
        # its bias -- so the eager reference all-reduces ``head_concat.bias``'s
        # grad over the spatial group twice. The vmap path (``XWT.backward``)
        # reduces once, which is the correct single reduction; divide the
        # double-counted reference back down to match.
        spatial = mesh["row"].size()
        if spatial > 1 and "head_concat.bias" in ref:
            ref["head_concat.bias"] = ref["head_concat.bias"] / spatial
        return got, ref

    @pytest.mark.dist(4)
    @pytest.mark.te_required
    def test_te_ensemble_vmap_forward(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        got, ref = self._fwd(process_mesh_dict_2x2, device, "te")
        assert_close(
            "te ensemble forward", got, ref, **_FWD_TOL,
            device=device, group=process_mesh_dict_2x2["model"],
        )

    @pytest.mark.dist(4)
    def test_sdpa_ensemble_vmap_forward(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        got, ref = self._fwd(process_mesh_dict_2x2, device, "sdpa")
        assert_close(
            "sdpa ensemble forward", got, ref, **_FWD_TOL,
            device=device, group=process_mesh_dict_2x2["model"],
        )

    @pytest.mark.dist(4)
    @pytest.mark.te_required
    @pytest.mark.xfail(
        strict=True,
        reason="TransformerEngineAttention.backward recomputes the fused kernel "
        "under enable_grad + requires_grad_(), which functorch's grad transform "
        "forbids; the te core has no vmap(grad(...)) path.",
    )
    def test_te_ensemble_vmap_grad(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        got, ref = self._grad(process_mesh_dict_2x2, device, "te")
        for k in got:
            assert_close(
                f"te ensemble grad[{k}]", got[k], ref[k], **_GRAD_TOL,
                device=device, group=process_mesh_dict_2x2["model"],
            )

    @pytest.mark.dist(4)
    @pytest.mark.xfail(
        strict=True,
        reason="``head_concat`` grads match on the spatial ranks that own the "
        "reduction, but the ensemble ``qkv.weight`` grad -- back-propagated "
        "through the attention core under the members-fold -- drifts far past "
        "tolerance (max abs diff ~5-10) and one ensemble member's grad is near "
        "zero. Root cause is functorch's generic batched fallback for the "
        "spatial ``all_reduce`` / peer ``all_gather`` in ``XWT.backward``: it "
        "issues one collective per members slice with no cross-rank lockstep "
        "guarantee. Needs a real functorch batching rule for the "
        "``_c10d_functional`` collectives.",
    )
    def test_sdpa_ensemble_vmap_grad(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        got, ref = self._grad(process_mesh_dict_2x2, device, "sdpa")
        for k in got:
            assert_close(
                f"sdpa ensemble grad[{k}]", got[k], ref[k], **_GRAD_TOL,
                device=device, group=process_mesh_dict_2x2["model"],
            )


# --------------------------------------------------------------------------- #
# 4. ring attention under vmap -- fold-into-batch rule
# --------------------------------------------------------------------------- #
@pytest.mark.gpu_required
class TestRingVmap:
    """``torch.vmap`` over a ``ring=True`` ``DistributedAttention``.

    ``RingAttentionFunction.vmap`` folds the vmapped (ensemble) axis into the
    batch, runs one un-vmapped ring pass and unfolds -- the ring's per-chunk
    cores and its P2P ring exchange register no ``functorch`` batching rule, so
    the same fold-into-batch trick as :class:`TransformerEngineAttentionVmap`
    is the only way through. Forward parity vs a per-member loop, ``sdpa`` and
    ``te``.

    Grad is *not* covered: the ring backward is entirely P2P collectives with
    no batching rule, so a stacked-parameter ensemble (a vmapped projection
    weight) is explicitly rejected by the staticmethod.
    """

    E = 2

    def _fwd(self, mesh: dict, device: torch.device, backend: str):
        model = _build_attn(mesh, device, backend, bayesian=False, ring=True)
        x_e = _local_x(mesh, device, members=self.E)
        got = vmap(model)(x_e)
        ref = torch.stack([model(x_e[i].contiguous()) for i in range(self.E)])
        return got, ref

    @pytest.mark.dist(4)
    def test_ring_vmap_forward_matches_loop_sdpa(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        got, ref = self._fwd(process_mesh_dict_4x1, device, "sdpa")
        assert_close(
            "ring vmap forward sdpa", got, ref, **_FWD_TOL,
            device=device, group=process_mesh_dict_4x1["model"],
        )

    @pytest.mark.dist(4)
    @pytest.mark.te_required
    def test_ring_vmap_forward_matches_loop_te(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        got, ref = self._fwd(process_mesh_dict_4x1, device, "te")
        assert_close(
            "ring vmap forward te", got, ref, **_FWD_TOL,
            device=device, group=process_mesh_dict_4x1["model"],
        )

    @pytest.mark.dist(4)
    def test_ring_vmap_batched_weight_rejected(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        """A stacked-parameter ensemble (vmapped weight) is a clear error, not
        a silently wrong result -- the ring backward has no batching rule."""
        b_models = [
            _build_attn(process_mesh_dict_4x1, device, "sdpa", bayesian=True, ring=True)
            for _ in range(self.E)
        ]
        params, buffers = stack_module_state(b_models)
        x_e = _local_x(process_mesh_dict_4x1, device, members=self.E)
        with pytest.raises((RuntimeError, NotImplementedError)):
            vmap(
                lambda p, b, x: functional_call(b_models[0], (p, b), (x,))
            )(params, buffers, x_e)
