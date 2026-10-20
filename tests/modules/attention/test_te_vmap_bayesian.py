"""``torch.vmap`` / Bayesian-ensemble coverage for the TE attention backend.

``torch_blue`` trains a Bayesian ensemble by giving every parameter a leading
"members" axis and running the model under ``torch.vmap``. The attention path
was never exercised under ``vmap`` -- these tests pin it, at increasing levels
of integration:

1. :class:`TestTransformerEngineAttentionVmap` -- the
   :class:`~jigsaw.modules.attention._attention._TransformerEngineAttention`
   ``autograd.Function`` alone. It is the only piece that *has* to route through
   a custom Function under vmap (``te``'s fused kernels register no batching
   rule); its ``vmap`` staticmethod folds the members axis into the batch,
   runs one un-vmapped call, and unfolds. Forward + grad vs a Python loop.
   **Passes** -- including after the wrapper moved off
   ``te.DotProductAttention`` onto the fused kernels directly: the fold makes
   the folded ``q/k/v`` produce a matching folded LSE / ``rng_state``, so the
   re-entrant ``apply`` inside the rule sends gradients through the wrapper's
   own ``fused_attn_bwd`` backward.
2. :class:`TestDistributedAttentionPlain` -- a plain (no vmap, no ensemble)
   ``DistributedAttention`` forward+backward on a channel-parallel mesh, sdpa
   and te. **Passes.** Regression guards for two fixes: ``forward`` now casts
   the half-precision core output back to the model dtype on the ``_te_bshd`` branch too,
   and ``fused_xw_xtw_twoway`` makes ``grad_output`` contiguous before its
   ``all_to_all``.
3. :class:`TestDistributedAttentionEnsembleVmap` -- a full non-ring
   ``DistributedAttention`` under ``vmap(functional_call(...))`` with stacked
   ensemble parameters, vs a per-member loop, sdpa and te. **Forward passes;
   grad xfails** -- under ``vmap(grad(...))`` the backward collectives run on
   BatchedTensors, and ``c10d`` ops register no batching rule.
4. :class:`TestRingVmap` -- ``torch.vmap`` over a ``ring=True``
   ``DistributedAttention`` raises: ``_RingAttention`` has no ``vmap``
   staticmethod. **Passes** (asserts the raise); flip to a parity test when a
   fold-into-batch rule lands.

Dropout is 0 throughout: the members-folded single call and the per-member loop
would otherwise draw different masks and never match.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from torch.func import functional_call, grad, stack_module_state, vmap

from jigsaw.modules.attention import _attention as attention
from jigsaw.modules.attention._functional.cores import te as te_cores
from tests.parity import assert_close

# TE fused attention needs fp16/bf16 and (on CUDA) the attn core runs in half
# precision regardless of backend -- so vmap-vs-loop parity is a half-precision
# comparison (bf16 for the direct-core tests, which feed bf16 q/k/v).
_FWD_TOL = dict(atol=2e-2, rtol=2e-2)
_GRAD_TOL = dict(atol=3e-2, rtol=3e-2)  # grads flow through the half-precision core


# --------------------------------------------------------------------------- #
# 1. te_attention (_TransformerEngineAttention's batching rule) under torch.vmap
# --------------------------------------------------------------------------- #
@pytest.mark.gpu_required
@pytest.mark.te_required
class TestTransformerEngineAttentionVmap:
    """The wrapper's ``vmap`` staticmethod folds members into batch and unfolds."""

    E, B, H, S, D = 3, 2, 4, 16, 32
    SCALE = D**-0.5

    def _qkv(self, device, requires_grad=False):
        torch.manual_seed(0)
        return [
            torch.randn(
                self.E,
                self.B,
                self.S,
                self.H,
                self.D,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=requires_grad,
            )
            for _ in range(3)
        ]

    @staticmethod
    def _f(q, k, v, scale, core):
        from jigsaw.modules.attention._attention import _TransformerEngineAttention

        # apply() returns (out, logsumexp, rng_state); only the output is used
        # outside the ring.
        return _TransformerEngineAttention.apply(q, k, v, scale, 0.0, core)[0]

    def test_forward_matches_loop(self, device: torch.device) -> None:
        qe, ke, ve = self._qkv(device)

        core = te_cores.select_core(device)

        def f(q, k, v):
            return self._f(q, k, v, self.SCALE, core)

        got = vmap(f)(qe, ke, ve)
        assert got.shape == (self.E, self.B, self.S, self.H * self.D)

        # ``.clone()`` per member: TE's qkv-layout detector rejects a bare
        # ``qe[i]`` view (nonzero storage offset / cross-member pointer math).
        ref = torch.stack(
            [f(qe[i].clone(), ke[i].clone(), ve[i].clone()) for i in range(self.E)]
        )
        torch.testing.assert_close(got, ref, **_FWD_TOL)

    def test_grad_matches_loop(self, device: torch.device) -> None:
        qe, ke, ve = self._qkv(device, requires_grad=True)

        core = te_cores.select_core(device)

        def f(q, k, v):
            return self._f(q, k, v, self.SCALE, core)

        vmap(f)(qe, ke, ve).float().pow(2).sum().backward()
        got = (qe.grad, ke.grad, ve.grad)

        ref_members = []
        for i in range(self.E):
            qi, ki, vi = (
                t[i].detach().clone().requires_grad_(True) for t in (qe, ke, ve)
            )
            f(qi, ki, vi).float().pow(2).sum().backward()
            ref_members.append((qi.grad, ki.grad, vi.grad))
        ref = tuple(torch.stack([m[j] for m in ref_members]) for j in range(3))

        for name, g, r in zip("qkv", got, ref):
            assert g is not None, f"no grad for {name}"
            torch.testing.assert_close(g, r, **_GRAD_TOL)


# --------------------------------------------------------------------------- #
# helpers for the full-module tests
# --------------------------------------------------------------------------- #
_IN = _EMBED = _OUT = 64
_HEADS = 4


def _build_attn(mesh: dict, device: torch.device, backend: str, ring: bool = False):
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
      ``head_concat`` and ``DistributedLinear`` raised ``BFloat16 != float``.
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
        model = _build_attn(process_mesh_dict_2x2, device, "te")
        x = _local_x(process_mesh_dict_2x2, device)
        y = model(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert torch.isfinite(y).all()

    def _check_backward(self, mesh, device, backend):
        model = _build_attn(mesh, device, backend)
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


# --------------------------------------------------------------------------- #
# 3. full DistributedAttention under vmap (Bayesian ensemble)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu_required
class TestDistributedAttentionEnsembleVmap:
    """A non-ring ``DistributedAttention`` under ``vmap(functional_call(...))``
    with an ensemble (leading members axis) of stacked parameters -- the shape
    ``torch_blue`` drives for Bayesian sampling.

    Reference is a per-member Python loop over the same modules, so a mismatch
    is the vmap members axis, not the sharding.

    **Forward** works for both ``sdpa`` and ``te`` once the core-dtype and
    grad-input-contiguity fixes are in (see :class:`TestDistributedAttentionPlain`).
    **Grad** still xfails: under ``vmap(grad(...))`` the Functions' backward
    runs on BatchedTensors, and the ``c10d`` collectives it calls register no
    batching rule.
    """

    E = 2

    def _build(self, mesh, device, backend):
        """Ensemble of models (the per-member references) + stacked state + inputs."""
        ref_models = [_build_attn(mesh, device, backend) for _ in range(self.E)]
        params, buffers = stack_module_state(ref_models)
        base = _build_attn(mesh, device, backend)
        x_e = _local_x(mesh, device, members=self.E)
        return base, params, buffers, ref_models, x_e

    def _fwd(self, mesh, device, backend):
        base, params, buffers, ref_models, x_e = self._build(mesh, device, backend)
        got = vmap(lambda p, b, x: functional_call(base, (p, b), (x,)))(
            params, buffers, x_e
        )
        ref = torch.stack([ref_models[i](x_e[i].contiguous()) for i in range(self.E)])
        return got, ref

    def _grad(self, mesh, device, backend):
        base, params, buffers, ref_models, x_e = self._build(mesh, device, backend)

        def member_loss(p, b, x):
            return functional_call(base, (p, b), (x,)).float().pow(2).sum()

        got = vmap(grad(member_loss))(params, buffers, x_e)

        ref_list = []
        for i in range(self.E):
            rm = ref_models[i]
            rm.zero_grad(set_to_none=True)
            rm(x_e[i].contiguous()).float().pow(2).sum().backward()
            ref_list.append({n: p.grad for n, p in rm.named_parameters()})
        ref = {k: torch.stack([r[k] for r in ref_list]) for k in got}
        return got, ref

    @pytest.mark.dist(4)
    @pytest.mark.te_required
    def test_te_ensemble_vmap_forward(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        got, ref = self._fwd(process_mesh_dict_2x2, device, "te")
        assert_close(
            "te ensemble forward",
            got,
            ref,
            **_FWD_TOL,
            device=device,
            group=process_mesh_dict_2x2["model"],
        )

    @pytest.mark.dist(4)
    def test_sdpa_ensemble_vmap_forward(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        got, ref = self._fwd(process_mesh_dict_2x2, device, "sdpa")
        assert_close(
            "sdpa ensemble forward",
            got,
            ref,
            **_FWD_TOL,
            device=device,
            group=process_mesh_dict_2x2["model"],
        )

    _XWT_VMAP_GRAD = pytest.mark.xfail(
        strict=True,
        reason="c10d collectives in the Functions' backward have no batching rule under vmap(grad)",
    )

    @pytest.mark.dist(4)
    @pytest.mark.te_required
    @_XWT_VMAP_GRAD
    def test_te_ensemble_vmap_grad(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        got, ref = self._grad(process_mesh_dict_2x2, device, "te")
        for k in got:
            assert_close(
                f"te ensemble grad[{k}]",
                got[k],
                ref[k],
                **_GRAD_TOL,
                device=device,
                group=process_mesh_dict_2x2["model"],
            )

    @pytest.mark.dist(4)
    @_XWT_VMAP_GRAD
    def test_sdpa_ensemble_vmap_grad(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        got, ref = self._grad(process_mesh_dict_2x2, device, "sdpa")
        for k in got:
            assert_close(
                f"sdpa ensemble grad[{k}]",
                got[k],
                ref[k],
                **_GRAD_TOL,
                device=device,
                group=process_mesh_dict_2x2["model"],
            )


# --------------------------------------------------------------------------- #
# 4. ring attention under vmap -- not supported yet
# --------------------------------------------------------------------------- #
@pytest.mark.gpu_required
class TestRingVmap:
    """``torch.vmap`` over a ``ring=True`` ``DistributedAttention`` raises today:
    ``_RingAttention`` defines no ``vmap`` staticmethod and does not set
    ``generate_vmap_rule`` (unlike every other custom Function in jigsaw), so the
    ring core has no batching rule. When a fold-into-batch rule is added this
    should become a per-member parity check mirroring
    :class:`TestDistributedAttentionEnsembleVmap`.
    """

    @pytest.mark.dist(4)
    def test_ring_vmap_currently_unsupported(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        # ``4x1``: with its default config the ring needs an unsharded channel axis.
        model = _build_attn(process_mesh_dict_4x1, device, "sdpa", ring=True)
        x_e = _local_x(process_mesh_dict_4x1, device, members=2)

        with pytest.raises((RuntimeError, NotImplementedError)):
            vmap(model)(x_e)
