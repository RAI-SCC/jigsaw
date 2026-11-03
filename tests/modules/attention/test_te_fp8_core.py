"""The FP8 cuDNN attention core (``JIGSAW_TE_CORE=cudnn_fp8``).

Selection tests run anywhere. The numerical tests need an sm90+ GPU and a
``transformer_engine`` whose FP8 attention API imports (the container); they
compare the core against the fp32 ``math`` core, so the tolerances are those of
8-bit inputs, not of a bug-free fp32 kernel. Re-run them whenever the TE release
(``cudnn_fp8.TE_FP8_VERSION``) changes.
"""

import pytest
import torch

from jigsaw.modules.attention._functional.cores import math as math_core
from jigsaw.modules.attention._functional.cores import te as te_cores

fp8 = te_cores.cudnn_fp8

needs_fp8 = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
        and fp8.CUDNN_FP8_CORE_AVAILABLE
    ),
    reason="needs an sm90+ GPU and transformer_engine with the FP8 attention API",
)


class TestSelection:
    def test_env_selects_fp8_core(self, monkeypatch):
        monkeypatch.setenv("JIGSAW_TE_CORE", "cudnn_fp8")
        monkeypatch.setattr(fp8, "CUDNN_FP8_CORE_AVAILABLE", True)
        assert te_cores.select_core(torch.device("cpu")) is fp8

    def test_unavailable_te_raises(self, monkeypatch):
        monkeypatch.setenv("JIGSAW_TE_CORE", "cudnn_fp8")
        monkeypatch.setattr(fp8, "CUDNN_FP8_CORE_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="FP8"):
            te_cores.select_core(torch.device("cpu"))

    def test_below_sm90_raises(self, monkeypatch):
        monkeypatch.setenv("JIGSAW_TE_CORE", "cudnn_fp8")
        monkeypatch.setattr(fp8, "CUDNN_FP8_CORE_AVAILABLE", True)
        monkeypatch.setattr(torch.version, "hip", None)
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda device=None: (8, 0)
        )
        with pytest.raises(RuntimeError, match="sm90"):
            te_cores.select_core(torch.device("cuda"))

    def test_default_selection_never_picks_fp8(self, monkeypatch):
        monkeypatch.delenv("JIGSAW_TE_CORE", raising=False)
        assert te_cores.select_core(torch.device("cpu")) is not fp8


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@needs_fp8
class TestNumerics:
    B, H, S, D = 2, 4, 256, 128

    def _inputs(self):
        torch.manual_seed(0)
        mk = lambda: torch.randn(  # noqa: E731
            self.B, self.H, self.S, self.D, device="cuda", dtype=torch.bfloat16
        )
        return mk(), mk(), mk()

    def test_forward_matches_math(self):
        q, k, v = self._inputs()
        scale = self.D**-0.5
        out, lse, rng = fp8.forward(q, k, v, scale, 0.0)
        ref_out, ref_lse, _ = math_core.forward(
            q.float(), k.float(), v.float(), scale, 0.0
        )
        assert rng is None and out.dtype == q.dtype and out.shape == q.shape
        assert lse.shape == ref_lse.shape
        assert _rel(out, ref_out) < 0.1
        # the ring merges chunks with this LSE: it must be the true log-sum-exp
        assert (lse.float() - ref_lse.float()).abs().max().item() < 0.1

    def test_backward_matches_math(self):
        q, k, v = self._inputs()
        scale = self.D**-0.5
        g = torch.randn_like(q)
        out, lse, _ = fp8.forward(q, k, v, scale, 0.0)
        dq, dk, dv = fp8.backward(g, q, k, v, out, lse, scale)
        ref = math_core.backward(
            g.float(), q.float(), k.float(), v.float(), out.float(), lse.float(), scale
        )
        for got, want in zip((dq, dk, dv), ref[:3], strict=True):
            assert got.shape == q.shape and got.dtype == q.dtype
            assert _rel(got, want) < 0.25

    def test_bshd_layout(self):
        q, k, v = (t.transpose(1, 2).contiguous() for t in self._inputs())
        scale = self.D**-0.5
        out, lse, _ = fp8.forward(q, k, v, scale, 0.0, layout="bshd")
        assert out.shape == q.shape and lse.shape == (self.B, self.H, self.S)
