"""``AttnBackend`` selection, coercion and the TE-unavailable fallback.

The container path adds a three-way attention-core switch
(:class:`jigsaw.modules.attention.config.AttnBackend` -- ``te`` / ``sdpa`` /
``math``). These tests pin:

* enum construction and string coercion,
* the ring core selectors (``_select_attn_core_{forward,backward}_fn``) and the
  ``_te_core_or_raise`` guard, with TE availability forced via ``monkeypatch``,
* the deliberate asymmetry for ``attn_backend="te"`` when TE is missing --
  :class:`~jigsaw.modules.attention.attention.DistributedAttention` *warns and
  falls back to sdpa*, while :class:`~...config.RingAttentionConfig` *raises*.

Every test runs on a single rank against a 1-rank sub-group, so they work
unchanged whether the suite is launched with one rank or many, on CPU or GPU.
"""

import warnings

import pytest
import torch
import torch.distributed as dist

from jigsaw.modules.attention import attention
from jigsaw.modules.attention.config import (
    AttnBackend,
    RingAttentionConfig,
    _select_attn_core_backward_fn,
    _select_attn_core_forward_fn,
    _te_core_or_raise,
)
from jigsaw.modules.attention.functional.cores import math as math_core
from jigsaw.modules.attention.functional.cores import sdpa as sdpa_core
from jigsaw.modules.attention.functional.cores import te as te_cores

CPU = torch.device("cpu")


@pytest.fixture
def te_unavailable(monkeypatch):
    """Force every TE-availability flag off, regardless of the environment."""
    monkeypatch.setattr(attention, "JIGSAW_TE_AVAILABLE", False)
    monkeypatch.setattr(
        "jigsaw.modules.attention.functional.cores.te.TE_CORE_AVAILABLE", False
    )


# --------------------------------------------------------------------------- #
# enum
# --------------------------------------------------------------------------- #
class TestAttnBackendEnum:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("te", AttnBackend.TE),
            ("sdpa", AttnBackend.SDPA),
            ("math", AttnBackend.MATH),
        ],
    )
    def test_from_string(self, value, expected):
        assert AttnBackend(value) is expected

    def test_idempotent(self):
        assert AttnBackend(AttnBackend.TE) is AttnBackend.TE

    def test_is_str_enum(self):
        """``class AttnBackend(str, Enum)`` -- members compare equal to their value."""
        assert AttnBackend.SDPA == "sdpa"
        assert AttnBackend.TE.value == "te"

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError):
            AttnBackend("flash")


# --------------------------------------------------------------------------- #
# ring inner-core selectors
# --------------------------------------------------------------------------- #
class TestSelectRingCore:
    @pytest.mark.parametrize(
        "select", [_select_attn_core_forward_fn, _select_attn_core_backward_fn]
    )
    def test_math_backend_is_device_independent(self, select):
        assert select(CPU, "math") is select(torch.device("cuda"), "math")

    def test_sdpa_picks_by_device(self):
        assert _select_attn_core_forward_fn(CPU, "sdpa") is math_core.forward
        assert (
            _select_attn_core_forward_fn(torch.device("cuda"), "sdpa")
            is sdpa_core.forward
        )
        assert _select_attn_core_backward_fn(CPU, "sdpa") is math_core.backward

    def test_string_and_enum_arg_agree(self):
        assert _select_attn_core_forward_fn(
            CPU, "math"
        ) is _select_attn_core_forward_fn(CPU, AttnBackend.MATH)

    def test_sdpa_unknown_device_raises(self):
        with pytest.raises(ValueError, match="no attention forward core"):
            _select_attn_core_forward_fn(torch.device("meta"), "sdpa")

    def test_te_backend_raises_without_te(self, te_unavailable):
        with pytest.raises(RuntimeError, match="needs transformer_engine"):
            _select_attn_core_forward_fn(CPU, "te")
        with pytest.raises(RuntimeError, match="needs transformer_engine"):
            _select_attn_core_backward_fn(CPU, "te")


class TestTeCoreSelection:
    """``select_core``: cuDNN on Hopper and newer, ``flash_attn`` on Ampere."""

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("JIGSAW_TE_CORE", "cudnn")
        assert te_cores.select_core(torch.device("cuda")) is te_cores.cudnn
        monkeypatch.setenv("JIGSAW_TE_CORE", "flash")
        assert te_cores.select_core(torch.device("cuda")) is te_cores.flash

    def test_env_override_rejects_unknown(self, monkeypatch):
        monkeypatch.setenv("JIGSAW_TE_CORE", "cudnnn")
        with pytest.raises(ValueError, match="JIGSAW_TE_CORE"):
            te_cores.select_core(CPU)

    def test_cpu_resolves_to_cudnn(self, monkeypatch):
        """Historical behaviour: the selector succeeds, the kernel call reports
        that there is no GPU."""
        monkeypatch.delenv("JIGSAW_TE_CORE", raising=False)
        assert te_cores.select_core(CPU) is te_cores.cudnn

    @pytest.mark.parametrize(
        "capability,expected",
        [((8, 0), "flash"), ((8, 6), "flash"), ((9, 0), "cudnn"), ((10, 0), "cudnn")],
    )
    def test_architecture_rule(self, monkeypatch, capability, expected):
        # Pure selection logic -- force both arms "available" so this passes
        # without transformer_engine/flash_attn installed (e.g. CPU-only CI).
        monkeypatch.setattr(te_cores.cudnn, "CUDNN_CORE_AVAILABLE", True)
        monkeypatch.setattr(te_cores.flash, "FLASH_CORE_AVAILABLE", True)
        monkeypatch.delenv("JIGSAW_TE_CORE", raising=False)
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda device=None: capability
        )
        chosen = te_cores.select_core(torch.device("cuda"))
        assert chosen is getattr(te_cores, expected)

    def test_falls_back_when_preferred_arm_missing(self, monkeypatch):
        """A container without ``flash_attn`` still runs ``te`` on Ampere."""
        # cuDNN must look available for the fallback to land there, independent
        # of whether transformer_engine is actually installed in this environment.
        monkeypatch.setattr(te_cores.cudnn, "CUDNN_CORE_AVAILABLE", True)
        monkeypatch.delenv("JIGSAW_TE_CORE", raising=False)
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda device=None: (8, 0)
        )
        monkeypatch.setattr(te_cores.flash, "FLASH_CORE_AVAILABLE", False)
        assert te_cores.select_core(torch.device("cuda")) is te_cores.cudnn

    def test_raises_when_neither_arm_available(self, monkeypatch):
        monkeypatch.delenv("JIGSAW_TE_CORE", raising=False)
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda device=None: (9, 0)
        )
        monkeypatch.setattr(te_cores.flash, "FLASH_CORE_AVAILABLE", False)
        monkeypatch.setattr(te_cores.cudnn, "CUDNN_CORE_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="neither is importable"):
            te_cores.select_core(torch.device("cuda"))


class TestTeCoreGuard:
    def test_raises_when_unavailable(self, te_unavailable):
        with pytest.raises(RuntimeError, match="run inside the container"):
            _te_core_or_raise()

    def test_noop_when_available(self, monkeypatch):
        monkeypatch.setattr(
            "jigsaw.modules.attention.functional.cores.te.TE_CORE_AVAILABLE", True
        )
        assert _te_core_or_raise() is None


# --------------------------------------------------------------------------- #
# DistributedAttention: warn + fall back
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def solo_group():
    """A 1-rank process group for the current rank.

    ``DistributedAttention`` always builds a ``DistributedLinear`` head and
    broadcasts its parameters over ``spatial_group``; a size-1 group makes that
    a no-op, so the module constructs at any world size (and on CPU under an
    nccl default group) without cross-rank coupling.
    """
    if not dist.is_initialized():
        pytest.skip("needs an initialized process group")
    me = dist.get_rank()
    mine = None
    for r in range(dist.get_world_size()):
        group = dist.new_group(ranks=[r])
        if r == me:
            mine = group
    return mine


def _build_attn(attn_backend, device, group, **overrides):
    """Non-ring ``DistributedAttention`` wired to a 1-rank ``group``."""
    kwargs = dict(
        in_dim=16,
        embed_dim=16,
        out_dim=16,
        heads=4,
        dropout_p=0.0,
        local_rank=0,
        device=device,
        spatial_group=group,
        channel_group=group,
        ring=False,
        attn_backend=attn_backend,
    )
    kwargs.update(overrides)
    return attention.DistributedAttention(**kwargs)


class TestDistributedAttentionBackend:
    def test_te_without_te_warns_and_falls_back(
        self, te_unavailable, device, solo_group
    ):
        with pytest.warns(UserWarning, match="falling back to 'sdpa'"):
            model = _build_attn("te", device, solo_group)

        assert model.attn_backend is AttnBackend.SDPA
        assert model.te is False
        assert model._te_bshd is False
        assert model.attn_core.__name__ == "_attn_core"

    def test_math_backend_selects_math_core_without_warning(
        self, te_unavailable, device, solo_group
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = _build_attn("math", device, solo_group)

        assert model.attn_backend is AttnBackend.MATH
        assert model.te is False
        assert model.attn_core.__name__ == "_attn_core_math"

    def test_default_is_sdpa(self, te_unavailable, device, solo_group):
        model = _build_attn(AttnBackend.SDPA, device, solo_group)
        assert model.attn_backend is AttnBackend.SDPA
        assert model.attn_core.__name__ == "_attn_core"

    def test_string_and_enum_equivalent(self, te_unavailable, device, solo_group):
        assert _build_attn("math", device, solo_group).attn_core.__name__ == (
            _build_attn(AttnBackend.MATH, device, solo_group).attn_core.__name__
        )

    def test_unknown_backend_raises(self, te_unavailable, device, solo_group):
        with pytest.raises(ValueError):
            _build_attn("flash", device, solo_group)

    def test_math_core_is_plain_softmax_attention(
        self, te_unavailable, device, solo_group
    ):
        """``_attn_core_math`` is ``softmax(QK^T / sqrt(d)) V``, no mask/scale tweak."""
        model = _build_attn("math", device, solo_group)
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, model.heads, 8, 4, device=device) for _ in range(3))

        out = model._attn_core_math(q, k, v)
        weights = torch.softmax(
            (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5), dim=-1
        )
        ref = weights @ v

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- #
# RingAttentionConfig: the asymmetric TE behaviour + repr
# --------------------------------------------------------------------------- #
class TestRingAttentionConfigBackend:
    def test_default_backend_and_repr(self):
        cfg = RingAttentionConfig(device=CPU)
        assert cfg.attn_backend is AttnBackend.SDPA
        assert "attn_backend=SDPA" in repr(cfg)

    def test_math_backend_wires_math_core_and_repr(self):
        cfg = RingAttentionConfig(device=CPU, attn_backend="math")
        assert cfg.attn_backend is AttnBackend.MATH
        assert cfg.attn_core_forward_fn is math_core.forward
        assert cfg.attn_core_backward_fn is math_core.backward
        assert "attn_backend=MATH" in repr(cfg)

    def test_te_backend_raises_without_te(self, te_unavailable):
        """Unlike ``DistributedAttention``, the ring config does not fall back."""
        with pytest.raises(RuntimeError, match="needs transformer_engine"):
            RingAttentionConfig(device=CPU, attn_backend="te")


# --------------------------------------------------------------------------- #
# TE present (container)
# --------------------------------------------------------------------------- #
@pytest.mark.te_required
@pytest.mark.gpu_required
class TestBackendWithTe:
    def test_distributed_attention_te_is_wired(self, device, solo_group):
        from jigsaw.modules.attention.attention import DistributedAttention

        model = _build_attn("te", device, solo_group)

        assert model.attn_backend is AttnBackend.TE
        assert model.te is True
        # The TE core carries no `te.DotProductAttention` submodule: it calls the
        # kernels through the shared wrapper, parameterised per call by
        # `self.scale` / `self.dropout_p` and by the core resolved here.
        assert model.attn_core.__func__ is DistributedAttention._transformer_attn_core
        assert not hasattr(model, "attn")
        assert model.te_core is te_cores.select_core(device)

    def test_availability_flags_are_true(self):
        from jigsaw.modules.attention.attention import JIGSAW_TE_AVAILABLE
        from jigsaw.modules.attention.functional.cores.te import TE_CORE_AVAILABLE

        assert JIGSAW_TE_AVAILABLE is True
        assert TE_CORE_AVAILABLE is True

    def test_te_attn_forward_backward_matches_sdpa(self, device):
        """``te_attention`` runs fwd/bwd and tracks SDPA in bf16.

        Ported from the standalone ``test_transformer_engine.py`` so the numeric
        check survives that file's removal on ``refactor/tests``.
        """
        from jigsaw.modules.attention.attention import _TransformerEngineAttention

        batch, seq, heads, head_dim = 2, 16, 4, 32
        scale = head_dim**-0.5
        core = te_cores.select_core(device)

        torch.manual_seed(0)
        q, k, v = (
            torch.randn(
                batch, seq, heads, head_dim, device=device, dtype=torch.bfloat16
            )
            for _ in range(3)
        )
        for t in (q, k, v):
            t.requires_grad_(True)

        out, lse, rng_state = _TransformerEngineAttention.apply(
            q, k, v, scale, 0.0, core
        )
        assert out.shape == (batch, seq, heads * head_dim)
        # The LSE the ring needs is produced here too and simply dropped by the
        # non-ring core; `rng_state` is None without dropout.
        assert lse.shape == (batch, heads, seq)
        assert rng_state is None

        out.sum().backward()
        assert q.grad is not None and k.grad is not None and v.grad is not None

        ref = (
            torch.nn.functional.scaled_dot_product_attention(
                q.detach().transpose(1, 2),
                k.detach().transpose(1, 2),
                v.detach().transpose(1, 2),
                is_causal=False,
            )
            .transpose(1, 2)
            .reshape(batch, seq, heads * head_dim)
        )
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
