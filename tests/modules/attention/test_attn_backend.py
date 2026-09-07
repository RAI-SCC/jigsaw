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
        assert (
            _select_attn_core_backward_fn(CPU, "sdpa") is math_core.backward
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
            "jigsaw.modules.attention.functional.cores.te.TE_CORE_AVAILABLE", True
        )
        assert _te_core_or_raise() is None


# --------------------------------------------------------------------------- #
# DistributedAttention: warn + fall back
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def solo_group():
    """A 1-rank process group for the current rank.

    ``DistributedAttention`` always builds a ``DistributedXWT`` head and
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

    @pytest.mark.parametrize("backend", ["sdpa", "math"])
    def test_causal_flag_wires_into_attn_core(
        self, te_unavailable, device, solo_group, backend
    ):
        """``causal=True`` reaches ``is_causal`` on both the sdpa and math cores."""
        model = _build_attn(backend, device, solo_group, causal=True)
        core = model._attn_core if backend == "sdpa" else model._attn_core_math
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, model.heads, 6, 4, device=device) for _ in range(3))

        out = core(q, k, v)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_causal_defaults_false(self, te_unavailable, device, solo_group):
        model = _build_attn("sdpa", device, solo_group)
        assert model.causal is False

    @pytest.mark.dist(4)
    def test_causal_wires_ring_config(self, device, process_mesh_dict_4x1):
        """``ring=True`` builds its default ``RingAttentionConfig`` with the same ``causal``."""
        model = attention.DistributedAttention(
            in_dim=16,
            embed_dim=16,
            out_dim=16,
            heads=4,
            dropout_p=0.0,
            local_rank=process_mesh_dict_4x1["row"].rank(),
            device=device,
            spatial_group=process_mesh_dict_4x1["row"],
            channel_group=process_mesh_dict_4x1["col"],
            ring=True,
            causal=True,
        )
        assert model.config is not None
        assert model.config.causal is True


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
        import transformer_engine.pytorch as te

        model = _build_attn("te", device, solo_group)

        assert model.attn_backend is AttnBackend.TE
        assert model.te is True
        assert isinstance(model.attn, te.DotProductAttention)

    def test_availability_flags_are_true(self):
        from jigsaw.modules.attention.attention import JIGSAW_TE_AVAILABLE
        from jigsaw.modules.attention.functional.cores.te import TE_CORE_AVAILABLE

        assert JIGSAW_TE_AVAILABLE is True
        assert TE_CORE_AVAILABLE is True

    def test_te_attn_forward_backward_matches_sdpa(self, device):
        """``TransformerEngineAttention`` runs fwd/bwd and tracks SDPA in bf16.

        Ported from the standalone ``test_transformer_engine.py`` so the numeric
        check survives that file's removal on ``refactor/tests``.
        """
        import transformer_engine.pytorch as te

        from jigsaw.modules.attention.attention import (
            TransformerEngineAttention,
            _capture_rng_state,
        )

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

        rng_state = _capture_rng_state(device)
        out = TransformerEngineAttention.apply(q, k, v, attn, rng_state)
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

    def test_te_core_forward_backward_causal_matches_sdpa(self, device):
        """``causal=True`` reaches the cuDNN ``"causal"`` mask type end-to-end.

        Mirrors ``test_math_core_is_plain_softmax_attention`` /
        ``test_causal_flag_wires_into_attn_core`` for the TE ring core.
        """
        from jigsaw.modules.attention.functional.cores import te as te_core

        torch.manual_seed(0)
        b, h, s, d = 2, 4, 16, 32
        scale = d**-0.5
        q, k, v = (
            torch.randn(b, h, s, d, device=device, dtype=torch.bfloat16)
            for _ in range(3)
        )
        for t in (q, k, v):
            t.requires_grad_(True)

        out, lse = te_core.forward(q, k, v, scale, 0.0, causal=True)
        grad_out = torch.ones_like(out)
        dq, dk, dv = te_core.backward(
            grad_out,
            q.detach(),
            k.detach(),
            v.detach(),
            out.detach(),
            lse.detach(),
            scale,
            causal=True,
        )

        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale
        )
        ref.backward(grad_out)

        torch.testing.assert_close(out, ref.detach(), rtol=2e-2, atol=2e-2)
        assert q.grad is not None and k.grad is not None and v.grad is not None
        torch.testing.assert_close(dq, q.grad, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(dk, k.grad, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(dv, v.grad, rtol=2e-2, atol=2e-2)
