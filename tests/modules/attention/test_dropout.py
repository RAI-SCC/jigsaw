"""Dropout correctness tests.

Ring attention's SDPA/TE inner cores need CUDA to exercise their dropout path
at all (see their module docstrings in ``jigsaw.modules.attention.functional.cores``);
this file covers what *can* be checked without a GPU:

* the ring's ``math`` core (the CPU fallback for every backend) reproduces,
  via seed replay, the exact backward a plain autograd graph gives for the
  same forward dropout mask -- the mechanism ``ring_forward`` /
  ``ring_backward`` rely on to keep forward and backward masks in sync.
* ``combine_dropout_seed`` is a deterministic function of its inputs.
* dropout is gated by ``self.training`` for the non-ring ``Attention`` /
  ``DistributedAttention`` cores (eval mode must be deterministic and
  bit-identical to a ``dropout_p=0`` module with the same weights).
"""

import pytest
import torch

from jigsaw.modules.attention import Attention, DistributedAttention
from jigsaw.modules.attention.functional.cores import math as math_core
from jigsaw.modules.attention.functional.cores import sdpa as sdpa_core
from jigsaw.modules.attention.functional.cores import te as te_core
from jigsaw.modules.attention.functional.functional import combine_dropout_seed


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

        ``backward`` doesn't differentiate through a live graph -- it
        recomputes dq/dk/dv analytically from the saved *global* ``(out,
        lse)``, which for ``dropout_p > 0`` must also replay the identical
        mask (see ``math.py``'s module docstring for the derivation).
        Cross-checking against autograd through an explicit, differentiable
        reimplementation using the *same* mask is the strongest test
        available without a GPU (the SDPA/TE cores' dropout paths need CUDA
        and are otherwise untested -- see their module docstrings).
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
        """``dropout_p == 0`` must behave exactly as before -- no mask, seed unused."""
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
    """``combine_dropout_seed`` is what keeps a ring's forward and backward masks in sync."""

    def test_deterministic(self) -> None:
        assert combine_dropout_seed(42, 0, 3) == combine_dropout_seed(42, 0, 3)

    def test_varies_with_each_input(self) -> None:
        base = combine_dropout_seed(42, 0, 3)
        assert combine_dropout_seed(42, 1, 3) != base
        assert combine_dropout_seed(42, 0, 4) != base
        assert combine_dropout_seed(43, 0, 3) != base


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
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        """Same check as above, for the non-ring SDPA core `DistributedAttention` shares."""
        torch.manual_seed(0)
        mesh = process_mesh_dict_4x1
        kwargs = dict(
            in_dim=32,
            embed_dim=32,
            out_dim=32,
            heads=2,
            local_rank=mesh["col"].rank(),
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            bayesian=False,
            qk_norm=False,
            ring=False,
        )
        model = DistributedAttention(dropout_p=0.6, **kwargs)
        model_no_dropout = DistributedAttention(dropout_p=0.0, **kwargs)
        model_no_dropout.load_state_dict(model.state_dict())
        model.eval()
        model_no_dropout.eval()

        x = torch.randn(2, 8, 32, device=device)
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
        te_core, torch.bfloat16, (), 2e-2, id="te", marks=pytest.mark.te_required
    ),
]


@pytest.mark.gpu_required
@pytest.mark.parametrize("core, dtype, extra_bwd, tol", _CORES)
class TestSdpaTeCoreDropoutReplay:
    """The SDPA-flash / TE cores must replay their forward dropout mask in backward.

    The kernels draw an *opaque* mask, so we make it observable: with ``S == d``
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
            grad_out, q, k, v, out, lse, self.SCALE, *extra_bwd,
            dropout_p=self.P, rng_state=rng,
        )
        # deterministic given rng_state
        dq2, _, _ = core.backward(
            grad_out, q, k, v, out, lse, self.SCALE, *extra_bwd,
            dropout_p=self.P, rng_state=rng,
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
