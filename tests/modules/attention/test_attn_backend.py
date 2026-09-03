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

Everything runs single-process on CPU against a 1-rank WORLD group; no CUDA.
"""

import warnings

import pytest
import torch
import torch.distributed as dist

from jigsaw.modules.attention import attention
from jigsaw.modules.attention import config as attn_config
from jigsaw.modules.attention.config import (
    AttnBackend,
    RingAttentionConfig,
    _select_attn_core_backward_fn,
    _select_attn_core_forward_fn,
    _te_core_or_raise,
)

CPU = torch.device("cpu")


@pytest.fixture
def te_unavailable(monkeypatch):
    """Force every TE-availability flag off, regardless of the environment."""
    monkeypatch.setattr(attention, "JIGSAW_TE_AVAILABLE", False)
    monkeypatch.setattr(
        "jigsaw.modules.attention.functional.te_core.TE_CORE_AVAILABLE", False
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
        assert _select_attn_core_forward_fn(CPU, "sdpa") is attn_config._attn_core_math
        assert (
            _select_attn_core_forward_fn(torch.device("cuda"), "sdpa")
            is attn_config._attn_core_flash
        )
        assert (
            _select_attn_core_backward_fn(CPU, "sdpa")
            is attn_config._attn_core_math_backward
        )

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


class TestTeCoreGuard:
    def test_raises_when_unavailable(self, te_unavailable):
        with pytest.raises(RuntimeError, match="run inside the container"):
            _te_core_or_raise()

    def test_noop_when_available(self, monkeypatch):
        monkeypatch.setattr(
            "jigsaw.modules.attention.functional.te_core.TE_CORE_AVAILABLE", True
        )
        assert _te_core_or_raise() is None


# --------------------------------------------------------------------------- #
# DistributedAttention: warn + fall back
# --------------------------------------------------------------------------- #
def _build_attn(attn_backend, **overrides):
    """Build a non-ring, single-rank ``DistributedAttention`` on the WORLD group.

    ``channel_group.size() == 1`` keeps ``qkv`` a plain ``nn.Linear`` (no
    ``DistributedXWT``); the grad-sync hook still needs a real
    ``dist.ProcessGroup``, so the default WORLD group (a 1-rank gloo group under
    the autouse ``dist_env`` fixture) is used.
    """
    world = dist.group.WORLD
    kwargs = dict(
        in_dim=16,
        embed_dim=16,
        out_dim=16,
        heads=4,
        dropout_p=0.0,
        local_rank=0,
        device=CPU,
        spatial_group=world,
        channel_group=world,
        ring=False,
        attn_backend=attn_backend,
    )
    kwargs.update(overrides)
    return attention.DistributedAttention(**kwargs)


class TestDistributedAttentionBackend:
    def test_te_without_te_warns_and_falls_back(self, te_unavailable):
        with pytest.warns(UserWarning, match="falling back to 'sdpa'"):
            model = _build_attn("te")

        assert model.attn_backend is AttnBackend.SDPA
        assert model.te is False
        assert model._te_bshd is False
        assert model.attn_core.__name__ == "_attn_core"

    def test_math_backend_selects_math_core_without_warning(self, te_unavailable):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = _build_attn("math")

        assert model.attn_backend is AttnBackend.MATH
        assert model.te is False
        assert model.attn_core.__name__ == "_attn_core_math"

    def test_default_is_sdpa(self, te_unavailable):
        model = _build_attn(AttnBackend.SDPA)
        assert model.attn_backend is AttnBackend.SDPA
        assert model.attn_core.__name__ == "_attn_core"

    def test_string_and_enum_equivalent(self, te_unavailable):
        assert _build_attn("math").attn_core.__name__ == (
            _build_attn(AttnBackend.MATH).attn_core.__name__
        )

    def test_unknown_backend_raises(self, te_unavailable):
        with pytest.raises(ValueError):
            _build_attn("flash")

    def test_math_core_matches_sdpa_reference(self, te_unavailable):
        """The ``math`` core is plain scaled-dot-product attention."""
        model = _build_attn("math")
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, model.heads, 8, 4) for _ in range(3))

        out = model._attn_core_math(q, k, v)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

        torch.testing.assert_close(out, ref)


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
        assert cfg.attn_core_forward_fn is attn_config._attn_core_math
        assert cfg.attn_core_backward_fn is attn_config._attn_core_math_backward
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
    def test_distributed_attention_te_is_wired(self, device):
        import transformer_engine.pytorch as te

        model = _build_attn("te", device=device)

        assert model.attn_backend is AttnBackend.TE
        assert model.te is True
        assert isinstance(model.attn, te.DotProductAttention)

    def test_availability_flags_are_true(self):
        from jigsaw.modules.attention.attention import JIGSAW_TE_AVAILABLE
        from jigsaw.modules.attention.functional.te_core import TE_CORE_AVAILABLE

        assert JIGSAW_TE_AVAILABLE is True
        assert TE_CORE_AVAILABLE is True

    def test_te_attn_forward_backward_matches_sdpa(self, device):
        """``TransformerEngineAttention`` runs fwd/bwd and tracks SDPA in bf16.

        Ported from the standalone ``test_transformer_engine.py`` so the numeric
        check survives that file's removal on ``refactor/tests``.
        """
        import transformer_engine.pytorch as te

        from jigsaw.modules.attention.attention import TransformerEngineAttention

        batch, seq, heads, head_dim = 2, 16, 4, 32
        attn = te.DotProductAttention(
            num_attention_heads=heads,
            kv_channels=head_dim,
            num_gqa_groups=heads,
            qkv_format="bshd",
            attn_mask_type="no_mask",
        )

        torch.manual_seed(0)
        q, k, v = (
            torch.randn(
                batch, seq, heads, head_dim, device=device, dtype=torch.bfloat16
            )
            for _ in range(3)
        )
        for t in (q, k, v):
            t.requires_grad_(True)

        out = TransformerEngineAttention.apply(q, k, v, attn)
        assert out.shape == (batch, seq, heads * head_dim)

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
