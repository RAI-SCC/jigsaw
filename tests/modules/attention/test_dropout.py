"""Dropout correctness tests.

The GPU-free checks:

* the ring's ``math`` core (the CPU fallback for every backend) reproduces,
  via seed replay, the exact backward a plain autograd graph gives for the
  same forward dropout mask: the mechanism ``ring_forward`` /
  ``ring_backward`` rely on to keep forward and backward masks in sync.
* ``_combine_dropout_seed`` is a deterministic function of its inputs.
* dropout is gated by ``self.training`` for the non-ring ``Attention`` /
  ``DistributedAttention`` cores (eval mode must be deterministic and
  bit-identical to a ``dropout_p=0`` module with the same weights).

The GPU-only checks (``gpu_required``, and ``te_required`` for the TE ones;
skipped where the hardware or ``transformer_engine`` is missing):

* ``TestSdpaTeCoreDropoutReplay``: the ring's SDPA and TE (cuDNN and flash)
  inner cores draw an opaque mask in a CUDA kernel and must replay it from the
  saved RNG state in their backward.
* ``TestTeAttnDropoutReplay``: the non-ring ``_TransformerEngineAttention``
  wrapper, which drives those same cores in ``bshd`` and replays their mask
  through the same kernel-supplied ``rng_state``.

The distributed checks (4 ranks, any device; the TE ones need a GPU):

* ``TestDistributedDropoutReplay``: ring attention over every send/save/backward
  policy, and flat and hierarchical Ulysses, match an autograd reference through
  the mask recovered over the *global* sequence. A mismatch here means a chunk's
  backward replayed another chunk's ``seed`` / ``rng_state``.
"""

import itertools

import pytest
import torch
import torch.distributed as dist

from jigsaw.gather import gather_1d_results
from jigsaw.modules.attention import Attention, DistributedAttention
from jigsaw.modules.attention import _config as config
from jigsaw.modules.attention._functional.cores import math as math_core
from jigsaw.modules.attention._functional.cores import sdpa as sdpa_core
from jigsaw.modules.attention._functional.cores.te import cudnn as te_cudnn_core
from jigsaw.modules.attention._functional.cores.te import flash as te_flash_core
from jigsaw.modules.attention._functional.ring import (
    _combine_dropout_seed,
    _RingAttention,
)
from jigsaw.modules.attention._functional.ulysses import (
    hierarchical_ulysses_attention,
    ulysses_attention,
)
from tests.parity import all_ranks_ok, assert_close, no_tf32


class TestMathCoreDropout:
    """The ring's CPU-fallback attention core, with dropout enabled."""

    def test_forward_matches_manual_dropout(self) -> None:
        """``forward``'s masked-softmax output matches a manual reference using the same mask."""
        torch.manual_seed(0)
        b, h, s, d = 2, 3, 5, 4
        q, k, v = (torch.randn(b, h, s, d, dtype=torch.float64) for _ in range(3))
        scale = d**-0.5
        dropout_p = 0.3
        seed = 12345

        out, lse, rng_state = math_core.forward(q, k, v, scale, dropout_p, seed=seed)
        assert rng_state is None  # this core replays from `seed`, not saved state

        scores = (q @ k.mT) * scale
        p = torch.softmax(scores, dim=-1)
        keep = math_core._dropout_mask(p.shape, p.device, seed) >= dropout_p
        out_expected = (p * keep / (1 - dropout_p)) @ v

        torch.testing.assert_close(out, out_expected)
        torch.testing.assert_close(lse, scores.logsumexp(dim=-1))

    def test_backward_matches_autograd(self) -> None:
        """``backward``'s analytic dq/dk/dv match autograd through the same dropout mask.

        ``backward`` doesn't differentiate through a live graph; it
        recomputes dq/dk/dv analytically from the saved *global* ``(out,
        lse)``, which for ``dropout_p > 0`` must also replay the identical
        mask (see ``math.py``'s module docstring for the derivation).
        Cross-checking against autograd through an explicit, differentiable
        reimplementation using the *same* mask is the strongest test
        available without a GPU; the SDPA/TE cores' CUDA dropout paths are
        checked the same way in ``TestSdpaTeCoreDropoutReplay`` below.
        """
        torch.manual_seed(1)
        b, h, s, d = 2, 3, 7, 4
        q0, k0, v0 = (torch.randn(b, h, s, d, dtype=torch.float64) for _ in range(3))
        scale = d**-0.5
        dropout_p = 0.4
        seed = 777

        out, lse, _ = math_core.forward(q0, k0, v0, scale, dropout_p, seed=seed)
        grad_out = torch.randn_like(out)

        dq, dk, dv = math_core.backward(
            grad_out, q0, k0, v0, out, lse, scale, dropout_p=dropout_p, seed=seed
        )

        # Differentiable reference: rebuild the identical computation (same
        # mask, via the same seed) with autograd tracking on, and let autograd
        # supply the ground-truth gradient.
        q = q0.clone().requires_grad_(True)
        k = k0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        scores = (q @ k.mT) * scale
        p = torch.softmax(scores, dim=-1)
        keep = math_core._dropout_mask(p.shape, p.device, seed) >= dropout_p
        out_ref = (p * keep / (1 - dropout_p)) @ v
        out_ref.backward(grad_out)

        torch.testing.assert_close(dq, q.grad)
        torch.testing.assert_close(dk, k.grad)
        torch.testing.assert_close(dv, v.grad)

    def test_zero_dropout_ignores_seed(self) -> None:
        """``dropout_p == 0`` must behave exactly as before: no mask, seed unused."""
        torch.manual_seed(2)
        b, h, s, d = 2, 2, 4, 4
        q, k, v = (torch.randn(b, h, s, d) for _ in range(3))
        scale = d**-0.5

        out_a, lse_a, rng_a = math_core.forward(q, k, v, scale, 0.0, seed=None)
        out_b, lse_b, rng_b = math_core.forward(q, k, v, scale, 0.0, seed=999)
        torch.testing.assert_close(out_a, out_b)
        torch.testing.assert_close(lse_a, lse_b)
        assert rng_a is None
        assert rng_b is None


class TestDropoutSeedDerivation:
    """``_combine_dropout_seed`` is what keeps a ring's forward and backward masks in sync."""

    def test_deterministic(self) -> None:
        assert _combine_dropout_seed(42, 0, 3) == _combine_dropout_seed(42, 0, 3)

    def test_varies_with_each_input(self) -> None:
        base = _combine_dropout_seed(42, 0, 3)
        assert _combine_dropout_seed(42, 1, 3) != base
        assert _combine_dropout_seed(42, 0, 4) != base
        assert _combine_dropout_seed(43, 0, 3) != base


class TestEvalModeDisablesDropout:
    """``model.eval()`` must fully disable dropout, not just leave it at ``dropout_p``."""

    def test_serial_attention_eval_matches_zero_dropout(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cpu")
        in_dim = embed_dim = out_dim = 16
        heads = 2

        model = Attention(in_dim, embed_dim, out_dim, heads, device, dropout_p=0.6)
        model_no_dropout = Attention(
            in_dim, embed_dim, out_dim, heads, device, dropout_p=0.0
        )
        model_no_dropout.load_state_dict(model.state_dict())
        model.eval()
        model_no_dropout.eval()

        x = torch.randn(2, 5, in_dim)
        with torch.no_grad():
            out_dropout_eval = model(x)
            out_never_dropout = model_no_dropout(x)

        torch.testing.assert_close(out_dropout_eval, out_never_dropout)

    def test_serial_attention_train_mode_is_stochastic(self) -> None:
        """Sanity check that the eval-mode match above isn't vacuous (dropout_p=0.9 => almost surely differs)."""
        torch.manual_seed(0)
        device = torch.device("cpu")
        model = Attention(16, 16, 16, 2, device, dropout_p=0.9)
        model.train()

        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            out_a = model(x)
            out_b = model(x)

        assert not torch.allclose(out_a, out_b)

    @pytest.mark.dist_required
    @pytest.mark.dist(4)
    def test_distributed_attention_eval_matches_zero_dropout(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Same check as above, for the non-ring SDPA core `DistributedAttention` shares."""
        torch.manual_seed(0)
        mesh = process_mesh_dict_2x2
        kwargs = dict(
            in_dim=32,
            embed_dim=32,
            out_dim=32,
            heads=2,
            local_rank=mesh["col"].rank(),
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            qk_norm=False,
            ring=False,
        )
        model = DistributedAttention(dropout_p=0.6, **kwargs)
        model_no_dropout = DistributedAttention(dropout_p=0.0, **kwargs)
        model_no_dropout.load_state_dict(model.state_dict())
        model.eval()
        model_no_dropout.eval()

        # The channel dim is sharded over the column group.
        x = torch.randn(2, 8, 32 // mesh["col"].size(), device=device)
        with torch.no_grad():
            out_dropout_eval = model(x)
            out_never_dropout = model_no_dropout(x)

        torch.testing.assert_close(out_dropout_eval, out_never_dropout)


# --------------------------------------------------------------------------- #
# Ring inner cores whose dropout path needs CUDA (see their module docstrings)
# --------------------------------------------------------------------------- #
_CORES = [
    pytest.param(sdpa_core, torch.float16, (32, 32), 3e-3, id="sdpa"),
    pytest.param(
        te_cudnn_core,
        torch.bfloat16,
        (),
        2e-2,
        id="te-cudnn",
        marks=pytest.mark.te_required,
    ),
    # te.flash needs flash_attn rather than transformer_engine, but the marker
    # gates on the container that ships both.
    pytest.param(
        te_flash_core,
        torch.bfloat16,
        (),
        2e-2,
        id="te-flash",
        marks=pytest.mark.te_required,
    ),
]


@pytest.mark.gpu_required
@pytest.mark.parametrize("core, dtype, extra_bwd, tol", _CORES)
class TestSdpaTeCoreDropoutReplay:
    """The SDPA-flash / TE cores must replay their forward dropout mask in backward.

    The kernels draw an *opaque* mask, so the test makes it observable: with ``S == d``
    and ``V`` the identity (per ``(b, h)``), the attention output *is* the
    post-dropout probability matrix, and a differentiable reference built from
    that same recovered mask gives the ground-truth ``dq/dk/dv`` to compare
    ``<core>.backward(..., rng_state=...)`` against.
    """

    B, H, S = 2, 3, 32
    D = S
    SCALE = D**-0.5
    P = 0.3

    def _qkv(self, dtype: torch.dtype):
        torch.manual_seed(0)
        q = torch.randn(self.B, self.H, self.S, self.D, device="cuda", dtype=dtype)
        k = torch.randn(self.B, self.H, self.S, self.D, device="cuda", dtype=dtype)
        v = torch.eye(self.S, device="cuda", dtype=dtype)
        v = v.expand(self.B, self.H, self.S, self.S).contiguous()
        return q, k, v

    def test_forward_mask_is_stochastic_and_scaled(self, core, dtype, extra_bwd, tol):
        q, k, v = self._qkv(dtype)
        out0, _, rng0 = core.forward(q, k, v, self.SCALE, self.P)
        out1, _, _ = core.forward(q, k, v, self.SCALE, self.P)
        assert rng0 is not None
        assert not torch.allclose(out0, out1)  # fresh mask each call

        p_ref = torch.softmax((q.float() @ k.float().mT) * self.SCALE, dim=-1)
        kept = out0.float() > 0
        assert abs((1.0 - kept.float().mean().item()) - self.P) < 0.05
        torch.testing.assert_close(
            out0.float(), p_ref * kept / (1 - self.P), atol=tol, rtol=tol
        )

    def test_backward_replays_forward_mask(self, core, dtype, extra_bwd, tol):
        q, k, v = self._qkv(dtype)
        out, lse, rng = core.forward(q, k, v, self.SCALE, self.P)
        grad_out = torch.randn(
            self.B, self.H, self.S, self.D, device="cuda", dtype=dtype
        )

        dq, dk, dv = core.backward(
            grad_out,
            q,
            k,
            v,
            out,
            lse,
            self.SCALE,
            *extra_bwd,
            dropout_p=self.P,
            rng_state=rng,
        )
        # deterministic given rng_state
        dq2, _, _ = core.backward(
            grad_out,
            q,
            k,
            v,
            out,
            lse,
            self.SCALE,
            *extra_bwd,
            dropout_p=self.P,
            rng_state=rng,
        )
        assert torch.equal(dq, dq2)

        # ground truth: autograd through the identical (recovered) mask
        kept = (out.float() > 0).float()
        qg = q.float().clone().requires_grad_(True)
        kg = k.float().clone().requires_grad_(True)
        vg = v.float().clone().requires_grad_(True)
        p_soft = torch.softmax((qg @ kg.mT) * self.SCALE, dim=-1)
        ((p_soft * kept / (1 - self.P)) @ vg).backward(grad_out.float())
        for got, ref in ((dq, qg.grad), (dk, kg.grad), (dv, vg.grad)):
            torch.testing.assert_close(got.float(), ref, atol=tol, rtol=tol)


# --------------------------------------------------------------------------- #
# Non-ring _TransformerEngineAttention: cuDNN rng_state replay through fused_attn_bwd
# --------------------------------------------------------------------------- #
@pytest.mark.gpu_required
@pytest.mark.te_required
@pytest.mark.parametrize(
    "core",
    [
        pytest.param(te_cudnn_core, id="te-cudnn"),
        pytest.param(te_flash_core, id="te-flash"),
    ],
)
class TestTeAttnDropoutReplay:
    """``_TransformerEngineAttention`` (the non-ring, ``bshd`` wrapper around the
    same cores as :mod:`~jigsaw.modules.attention._functional.cores.te`) must
    replay its forward dropout mask in backward on either arm, so both are
    covered here regardless of which one this GPU's architecture selects.

    Same observability trick as :class:`TestSdpaTeCoreDropoutReplay`, one level
    up: with ``S == d`` and ``V`` the identity per ``(b, h)``, the attention
    output *is* the post-dropout probability matrix, so the mask cuDNN drew
    (opaque inside the kernel) can be recovered from the output and used to
    build a differentiable ground-truth reference.

    The wrapper carries the kernel's own ``rng_state`` from ``forward`` into its
    backward; there is no forward recompute and no ambient CPU/CUDA RNG snapshot
    involved (the mask is not drawn from the ambient stream).
    """

    B, H, S = 2, 3, 32
    D = S
    SCALE = D**-0.5
    P = 0.3
    TOL = 2e-2

    def _qkv(self, requires_grad: bool = False):
        """``bshd`` q/k, and a ``V`` that is the identity for every ``(b, h)``."""
        torch.manual_seed(0)
        q, k = (
            torch.randn(
                self.B,
                self.S,
                self.H,
                self.D,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=requires_grad,
            )
            for _ in range(2)
        )
        v = torch.eye(self.S, device="cuda", dtype=torch.bfloat16)
        v = v[None, :, None, :].expand(self.B, self.S, self.H, self.S).contiguous()
        v.requires_grad_(requires_grad)
        return q, k, v

    def _apply(self, q, k, v, dropout_p, core):
        from jigsaw.modules.attention._attention import _TransformerEngineAttention

        return _TransformerEngineAttention.apply(q, k, v, self.SCALE, dropout_p, core)[
            0
        ]

    def _bhsd(self, t):
        """``[b, s, h*d]`` (the wrapper's output layout) -> ``[b, h, s, d]``."""
        return t.reshape(self.B, self.S, self.H, self.D).transpose(1, 2)

    def test_forward_mask_is_stochastic_and_scaled(self, core) -> None:
        q, k, v = self._qkv()
        out0 = self._bhsd(self._apply(q, k, v, self.P, core))
        out1 = self._bhsd(self._apply(q, k, v, self.P, core))
        assert not torch.allclose(out0, out1)  # fresh mask each call

        qb, kb = (t.transpose(1, 2).float() for t in (q, k))
        p_ref = torch.softmax((qb @ kb.mT) * self.SCALE, dim=-1)
        kept = out0.float() > 0
        assert abs((1.0 - kept.float().mean().item()) - self.P) < 0.05
        torch.testing.assert_close(
            out0.float(), p_ref * kept / (1 - self.P), atol=self.TOL, rtol=self.TOL
        )

    def test_dropout_p_zero_is_deterministic(self, core) -> None:
        """Sanity: the stochasticity above is the dropout, not the kernel."""
        q, k, v = self._qkv()
        torch.testing.assert_close(
            self._apply(q, k, v, 0.0, core), self._apply(q, k, v, 0.0, core)
        )

    def test_backward_replays_the_forward_mask(self, core) -> None:
        q, k, v = self._qkv(requires_grad=True)
        out = self._apply(q, k, v, self.P, core)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        # ground truth: autograd through the identical (recovered) mask
        kept = (self._bhsd(out.detach()).float() > 0).float()
        qg, kg, vg = (
            t.detach().transpose(1, 2).float().clone().requires_grad_(True)
            for t in (q, k, v)
        )
        p_soft = torch.softmax((qg @ kg.mT) * self.SCALE, dim=-1)
        ref_out = (p_soft * kept / (1 - self.P)) @ vg
        ref_out.backward(self._bhsd(grad_out).float())

        for name, got, ref in zip(
            "qkv", (q.grad, k.grad, v.grad), (qg.grad, kg.grad, vg.grad), strict=True
        ):
            assert got is not None, f"no grad for {name}"
            torch.testing.assert_close(
                got.float().transpose(1, 2), ref, atol=self.TOL, rtol=self.TOL
            )

    def test_backward_does_not_draw_a_fresh_mask(self, core) -> None:
        """Guard against the check above being vacuous: a *different* mask
        almost surely gives a different gradient."""
        q, k, v = self._qkv(requires_grad=True)
        out = self._apply(q, k, v, self.P, core)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        torch.manual_seed(1)
        kept = (
            torch.rand(self.B, self.H, self.S, self.S, device="cuda") > self.P
        ).float()
        qg, kg, vg = (
            t.detach().transpose(1, 2).float().clone().requires_grad_(True)
            for t in (q, k, v)
        )
        p_soft = torch.softmax((qg @ kg.mT) * self.SCALE, dim=-1)
        ((p_soft * kept / (1 - self.P)) @ vg).backward(self._bhsd(grad_out).float())
        assert not torch.allclose(
            q.grad.float().transpose(1, 2), qg.grad, atol=self.TOL, rtol=self.TOL
        )


# --------------------------------------------------------------------------- #
# Distributed paths: ring and Ulysses, checked over the global sequence
# --------------------------------------------------------------------------- #
_P = 0.3
_B, _H, _S_LOCAL = 2, 2, 16


def _pairwise_cover(combos: list[tuple]) -> set[tuple]:
    """Greedy (deterministic) subset of ``combos`` covering every value pair."""

    def pairs(c: tuple) -> set[tuple]:
        return {(i, c[i], j, c[j]) for i, j in itertools.combinations(range(len(c)), 2)}

    uncovered = set().union(*map(pairs, combos))
    chosen = set()
    while uncovered:
        best = max(combos, key=lambda c: len(pairs(c) & uncovered))
        chosen.add(best)
        uncovered -= pairs(best)
    return chosen


def _ring_dropout_cases() -> list:
    """Every supported ring policy; the pairwise cover runs by default, the rest is slow."""
    combos = [
        c
        for c in itertools.product(
            config.SendOp,
            config.SendForward,
            config.SaveBackward,
            config.SaveScope,
            config.SendBackward,
            (False, True),
        )
        if not (c[1] == config.SendForward.KV and c[2] == config.SaveBackward.X)
    ]
    default = _pairwise_cover(combos)
    return [
        pytest.param(
            *c,
            id="-".join([*(e.name for e in c[:5]), "overlap" if c[5] else "sync"]),
            marks=() if c in default else pytest.mark.slow,
        )
        for c in combos
    ]


_RING_BACKENDS = [
    pytest.param("sdpa", id="sdpa"),
    pytest.param(
        "te", id="te", marks=[pytest.mark.gpu_required, pytest.mark.te_required]
    ),
    pytest.param("math", id="math"),
]
_ULYSSES_BACKENDS = _RING_BACKENDS[:2]


def _dtype_tol(device: torch.device, backend: str) -> tuple[torch.dtype, float]:
    """The fused CUDA kernels need 16 bit; everything else runs (TF32-free) fp32."""
    if device.type == "cuda" and backend != "math":
        return torch.bfloat16, 2e-2
    return torch.float32, 1e-4


def _identity_v_inputs(
    mesh: dict,
    layout: str,
    heads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """This rank's ``q, k, v, x, w_qkv, dout`` with ``V`` the identity over ``row``.

    ``head_dim`` equals the global sequence, so each attention output row is that
    query's post-dropout probability row and the opaque kernel mask is ``out > 0``.
    ``x`` holds this rank's rows of the identity and ``w_qkv``'s value block is the
    identity per head, so the ring's recompute of K/V from ``x`` sees the same V.
    Each ``col`` rank holds its own heads, as under channel parallelism.
    """
    rank, s = mesh["row"].rank(), _S_LOCAL * mesh["row"].size()
    g = torch.Generator().manual_seed(1234 + mesh["col"].rank())
    w_q, w_k = (torch.randn(heads * s, s, generator=g) * 0.3 for _ in range(2))
    w_v = torch.eye(s).repeat(heads, 1)
    w_qkv = torch.cat([w_q, w_k, w_v]).to(device, dtype)
    x = torch.eye(s, device=device, dtype=dtype)[
        rank * _S_LOCAL : (rank + 1) * _S_LOCAL
    ]
    x = x.expand(_B, -1, -1).contiguous()
    qkv = (x @ w_qkv.mT).reshape(_B, _S_LOCAL, 3, heads, s)
    perm = (2, 0, 1, 3, 4) if layout == "bshd" else (2, 0, 3, 1, 4)
    q, k, v = (t.contiguous().requires_grad_(True) for t in qkv.permute(*perm))
    seed = 100 + dist.get_rank()
    dout = torch.randn(q.shape, generator=torch.Generator().manual_seed(seed))
    return q, k, v, x, w_qkv, dout.to(device, dtype)


def _check_against_recovered_mask(
    mesh: dict,
    layout: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    tol: float,
    device: torch.device,
) -> torch.Tensor:
    """Compare ``out`` and the q/k/v grads with autograd through the recovered mask.

    Returns the ``(b, heads, s, s)`` keep mask of every head on the mesh. Verdicts
    are reduced over the whole model group, so a mismatch in one ``col`` rank's
    ring fails every rank instead of hanging the others.
    """
    group = mesh["model"]

    def bhsd(t: torch.Tensor) -> torch.Tensor:
        return t.transpose(1, 2) if layout == "bshd" else t

    q_g, k_g, out_g, dout_g = (
        gather_1d_results(bhsd(t.detach()).contiguous(), mesh, dim=2, pm_key="row")
        for t in (q, k, out, dout)
    )
    keep = (out_g.float() > 0).float()
    dropped = 1.0 - keep.mean().item()
    if not all_ranks_ok(abs(dropped - _P) < 0.05, device, group):
        raise AssertionError(f"dropped fraction {dropped:.3f}, expected {_P}")

    b, h, s, _ = q_g.shape
    q_ref, k_ref = (t.float().requires_grad_(True) for t in (q_g, k_g))
    v_ref = torch.eye(s, device=device).expand(b, h, s, s).clone().requires_grad_(True)
    p = torch.softmax((q_ref @ k_ref.mT) * s**-0.5, dim=-1)
    out_ref = (p * keep / (1 - _P)) @ v_ref
    out_ref.backward(dout_g.float())

    rank = mesh["row"].rank()
    local = slice(rank * _S_LOCAL, (rank + 1) * _S_LOCAL)
    for name, got, ref in (
        ("out", out, out_ref.detach()),
        ("dq", q.grad, q_ref.grad),
        ("dk", k.grad, k_ref.grad),
        ("dv", v.grad, v_ref.grad),
    ):
        assert got is not None and ref is not None, f"no {name}"
        ref = ref[:, :, local]
        assert_close(
            name,
            bhsd(got).float(),
            ref,
            atol=tol * ref.abs().max().item(),
            rtol=tol,
            device=device,
            group=group,
        )
    return gather_1d_results(keep, mesh, dim=1, pm_key="col")


def _n_identical(blocks: list[torch.Tensor]) -> int:
    return sum(torch.equal(a, b) for a, b in itertools.combinations(blocks, 2))


_SHARED_SEED_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="masks are drawn from the ambient generator (SDPA/TE kernels, the "
    "dense Ulysses core) or seeded from the ring rank only (math core), so ranks "
    "in the same RNG state draw the same mask for different blocks or heads",
)


@pytest.mark.dist_required
@pytest.mark.dist(4)
@pytest.mark.parametrize(
    "mesh", [dict(row=4, col=1), dict(row=2, col=2)], indirect=True, ids=["4x1", "2x2"]
)
class TestDistributedDropoutReplay:
    """Ring and Ulysses dropout against autograd through the global recovered mask.

    Same observability trick as :class:`TestSdpaTeCoreDropoutReplay`, over the
    global sequence: ``head_dim`` equals the global sequence length and ``V`` is the
    identity across the ``row`` (sequence) group. On the 2x2 mesh each ``col`` rank
    runs its own ring over its own heads.
    """

    def _ring(
        self,
        device: torch.device,
        mesh: dict,
        backend: str,
        cfg_kwargs: dict,
    ) -> torch.Tensor:
        group = mesh["row"]
        # Two intra groups, so the HYBRID* ops have a real outer ring.
        cfg = config.RingAttentionConfig(
            device=device,
            intra_size=group.size() // 2,
            attn_backend=backend,
            **cfg_kwargs,
        )
        dtype, tol = _dtype_tol(device, backend)
        q, k, v, x, w_qkv, dout = _identity_v_inputs(
            mesh, cfg.core_layout, _H, dtype, device
        )
        out = _RingAttention.apply(
            q, k, v, x, w_qkv, q.shape[-1] ** -0.5, _P, group, group.rank(), cfg, True
        )[0]
        out.backward(dout)
        return _check_against_recovered_mask(
            mesh, cfg.core_layout, q, k, v, out, dout, tol, device
        )

    def _ulysses(
        self, device: torch.device, mesh: dict, backend: str, hierarchical: bool
    ) -> torch.Tensor:
        spatial, channel = mesh["row"].size(), mesh["col"].size()
        heads = spatial  # local heads; flat Ulysses splits them over the ring
        dtype, tol = _dtype_tol(device, backend)
        s = _S_LOCAL * spatial
        model = DistributedAttention(
            in_dim=s,
            embed_dim=heads * channel * s,
            out_dim=s,
            heads=heads * channel,
            local_rank=mesh["col"].rank(),
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            dropout_p=_P,
            ulysses=True,
            ulysses_intra_size=spatial // 2 if hierarchical else 0,
            attn_backend=backend,
            dtype=dtype,
        )
        q, k, v, _x, _w, dout = _identity_v_inputs(mesh, "bshd", heads, dtype, device)
        if hierarchical:
            out = hierarchical_ulysses_attention(
                q, k, v, model._uly_intra_group, model._inter_ring_core(_P)
            )
        else:
            out = ulysses_attention(q, k, v, mesh["row"], model._dense_bshd_core(_P))
        out.backward(dout)
        return _check_against_recovered_mask(
            mesh, "bshd", q, k, v, out, dout, tol, device
        )

    @pytest.mark.parametrize("backend", _RING_BACKENDS)
    @pytest.mark.parametrize(
        "send_op, send_in_forward, save_for_backward, save_scope, send_backward, "
        "overlap",
        _ring_dropout_cases(),
    )
    def test_ring_replays_each_chunk_mask(
        self,
        request: pytest.FixtureRequest,
        device: torch.device,
        mesh: dict,
        backend: str,
        send_op: config.SendOp,
        send_in_forward: config.SendForward,
        save_for_backward: config.SaveBackward,
        save_scope: config.SaveScope,
        send_backward: config.SendBackward,
        overlap: bool,
    ) -> None:
        if overlap and device.type != "cuda":
            pytest.skip("overlap_comm is inert off CUDA")
        if (
            device.type == "cpu"
            and send_op == config.SendOp.HYBRID
            and save_scope == config.SaveScope.LOCAL
            and mesh["row"].size() == 2
        ):
            request.applymarker(
                pytest.mark.xfail(
                    strict=True,
                    reason="gloo 'Pair is not connected' in the HYBRID LOCAL "
                    "backward with intra groups of 1; fails with dropout_p=0 too",
                )
            )
        torch.manual_seed(dist.get_rank())
        with no_tf32():
            self._ring(
                device,
                mesh,
                backend,
                dict(
                    send_op=send_op,
                    send_in_forward=send_in_forward,
                    save_for_backward=save_for_backward,
                    save_scope=save_scope,
                    send_backward=send_backward,
                    overlap_comm=overlap,
                ),
            )

    @pytest.mark.parametrize("backend", _ULYSSES_BACKENDS)
    @pytest.mark.parametrize("hierarchical", [False, True], ids=["flat", "hier"])
    def test_ulysses_replays_the_mask(
        self, device: torch.device, mesh: dict, backend: str, hierarchical: bool
    ) -> None:
        torch.manual_seed(dist.get_rank())
        with no_tf32():
            self._ulysses(device, mesh, backend, hierarchical)

    @pytest.mark.parametrize("backend", _RING_BACKENDS)
    def test_ring_masks_independent_under_a_shared_seed(
        self,
        request: pytest.FixtureRequest,
        device: torch.device,
        mesh: dict,
        backend: str,
    ) -> None:
        """With the same RNG state on every rank, no two (head shard, block) share a mask."""
        kernel_core = device.type == "cuda" and backend != "math"
        if kernel_core or mesh["col"].size() > 1:
            request.applymarker(_SHARED_SEED_XFAIL)
        torch.manual_seed(0)
        with no_tf32():
            keep = self._ring(
                device,
                mesh,
                backend,
                dict(
                    send_in_forward=config.SendForward.KV,
                    save_scope=config.SaveScope.LOCAL,
                ),
            )
        n = mesh["row"].size()
        blocks = [
            keep[
                :,
                c * _H : (c + 1) * _H,
                i * _S_LOCAL : (i + 1) * _S_LOCAL,
                j * _S_LOCAL : (j + 1) * _S_LOCAL,
            ]
            for c in range(mesh["col"].size())
            for i in range(n)
            for j in range(n)
        ]
        identical = _n_identical(blocks)
        if not all_ranks_ok(identical == 0, device, mesh["model"]):
            raise AssertionError(f"{identical} pairs of blocks share a dropout mask")

    @pytest.mark.parametrize("backend", _ULYSSES_BACKENDS)
    @_SHARED_SEED_XFAIL
    def test_ulysses_masks_independent_under_a_shared_seed(
        self, device: torch.device, mesh: dict, backend: str
    ) -> None:
        """With the same RNG state on every rank, no two heads share a mask."""
        torch.manual_seed(0)
        with no_tf32():
            keep = self._ulysses(device, mesh, backend, False)
        identical = _n_identical(list(keep.unbind(1)))
        if not all_ranks_ok(identical == 0, device, mesh["model"]):
            raise AssertionError(f"{identical} pairs of heads share a dropout mask")
