"""``attn_mask`` support in :class:`Attention` / :class:`DistributedAttention`.

Motivated by Swin's shifted-window attention on a global (lat/lon) grid: the cyclic
shift used to move the window grid wraps the longitude axis correctly, but a window
straddling a pole in the latitude direction must *not* let the two sides it stitched
together attend to each other. That needs an arbitrary per-window mask, broadcastable
to ``q``'s ``(..., heads, seq, seq)`` shape -- not a simple per-sample valid-length
("padding") cutoff.

Checked against the installed ``transformer_engine`` (2.14.0) before writing this:
its fused kernels (what :mod:`.functional.cores.te` calls directly) only implement
``attn_mask_type`` in ``{no_mask, causal, padding, ...}`` -- no tensor-valued mask, no
``"arbitrary"``. TE's own ``DotProductAttention`` *does* accept an arbitrary mask, but
its source (``dot_product_attention/utils.py``) explicitly disables both FlashAttention
and FusedAttention for it and routes to its *unfused* backend instead -- the same cost
class as the ``"math"`` core already in this repo, not a performance win. So:

* ``attn_backend`` in ``{"sdpa", "math"}`` (the non-ring, non-Ulysses core): genuinely
  supported. ``torch.nn.functional.scaled_dot_product_attention``'s flash backend
  rejects a non-null ``attn_mask`` outright, but with ``sdpa_kernel([FLASH, MATH])``
  (what :meth:`Attention._attn_core`/:meth:`DistributedAttention._attn_core` already
  use) that just silently routes the masked call to the math backend -- no error,
  correct result, confirmed below.
* ``attn_backend="te"``, ``ring=True``, ``ulysses=True``: all raise ``ValueError``
  rather than silently taking that unfused detour or ignoring the mask -- see each
  core's docstring in ``attention.py``.
"""

import pytest
import torch
import torch.distributed as dist

import jigsaw.modules.attention as attention
from jigsaw.modules.attention import config as cfg

DIM, HEADS = 32, 4


def _block_mask(n0: int, n1: int) -> torch.Tensor:
    """A block-diagonal mask: ``n0`` tokens of one region, ``n1`` of another.

    Mimics a shifted Swin window straddling a border it must not attend across (a
    pole, on a global grid): tokens keep their own region's visibility and nothing
    else. Block-diagonal masking exactly decomposes attention into two independent
    per-region calls -- each query's softmax normalises only over the keys it can
    see -- which is what the tests below check against.
    """
    n = n0 + n1
    mask = torch.zeros(n, n, dtype=torch.bool)
    mask[:n0, :n0] = True
    mask[n0:, n0:] = True
    return mask


@pytest.fixture
def solo_group():
    """A 1-rank process group for the current rank (see ``test_attn_backend.py``)."""
    if not dist.is_initialized():
        pytest.skip("needs an initialized process group")
    me = dist.get_rank()
    mine = None
    for r in range(dist.get_world_size()):
        group = dist.new_group(ranks=[r])
        if r == me:
            mine = group
    return mine


# --------------------------------------------------------------------------- #
# plain (serial) Attention
# --------------------------------------------------------------------------- #
class TestAttentionMask:
    """``Attention`` (the un-sharded, single-process module)."""

    n0, n1 = 3, 5

    def _model(self, device: torch.device) -> attention.Attention:
        torch.manual_seed(0)
        model = attention.Attention(DIM, DIM, DIM, HEADS, device)
        model.eval()  # dropout_p=0.0 already, but be explicit about no RNG noise
        return model

    def test_block_mask_equals_separate_attention(self, device: torch.device) -> None:
        """A block-diagonal mask reproduces running attention on each block alone."""
        model = self._model(device)
        n0, n1 = self.n0, self.n1
        torch.manual_seed(1)
        x = torch.randn(2, n0 + n1, DIM, device=device)
        mask = _block_mask(n0, n1).to(device)

        out_masked = model(x, attn_mask=mask)
        out_ref = torch.cat([model(x[:, :n0]), model(x[:, n0:])], dim=1)

        # On CUDA the core runs in bf16 (`attn_core_dtype`): ~1e-3 roundoff.
        tol = 5e-3 if device.type == "cuda" else 1e-5
        torch.testing.assert_close(out_masked, out_ref, rtol=tol, atol=tol)

    def test_additive_float_mask_matches_boolean(self, device: torch.device) -> None:
        """``-inf``-additive and boolean masks are equivalent conventions."""
        model = self._model(device)
        n0, n1 = self.n0, self.n1
        torch.manual_seed(1)
        x = torch.randn(2, n0 + n1, DIM, device=device)
        bool_mask = _block_mask(n0, n1).to(device)
        float_mask = torch.zeros_like(bool_mask, dtype=x.dtype).masked_fill(
            ~bool_mask, float("-inf")
        )

        out_bool = model(x, attn_mask=bool_mask)
        out_float = model(x, attn_mask=float_mask)

        tol = 5e-3 if device.type == "cuda" else 1e-5
        torch.testing.assert_close(out_bool, out_float, rtol=tol, atol=tol)

    def test_no_mask_is_unaffected(self, device: torch.device) -> None:
        """``attn_mask=None`` (the default) reproduces the pre-existing behaviour."""
        model = self._model(device)
        torch.manual_seed(1)
        x = torch.randn(2, self.n0 + self.n1, DIM, device=device)

        torch.testing.assert_close(model(x), model(x, attn_mask=None))

    def test_masked_region_is_blind_to_the_other(self, device: torch.device) -> None:
        """Changing the keys/values of one region must not move the other's output."""
        model = self._model(device)
        n0, n1 = self.n0, self.n1
        torch.manual_seed(1)
        x = torch.randn(2, n0 + n1, DIM, device=device)
        mask = _block_mask(n0, n1).to(device)

        out1 = model(x, attn_mask=mask)
        x2 = x.clone()
        x2[:, n0:] += 100.0  # perturb only the second region
        out2 = model(x2, attn_mask=mask)

        torch.testing.assert_close(out1[:, :n0], out2[:, :n0], rtol=1e-5, atol=1e-5)
        assert not torch.allclose(out1[:, n0:], out2[:, n0:])

    def test_gradient_respects_the_mask(self, device: torch.device) -> None:
        """Backward through the masked call matches backward through the two halves."""
        model = self._model(device)
        n0, n1 = self.n0, self.n1
        torch.manual_seed(1)
        x_global = torch.randn(2, n0 + n1, DIM, device=device)
        mask = _block_mask(n0, n1).to(device)

        x_masked = x_global.clone().requires_grad_(True)
        out_masked = model(x_masked, attn_mask=mask)
        grad_out = torch.randn_like(out_masked)
        out_masked.backward(grad_out)

        x_a = x_global[:, :n0].clone().requires_grad_(True)
        x_b = x_global[:, n0:].clone().requires_grad_(True)
        out_a, out_b = model(x_a), model(x_b)
        out_a.backward(grad_out[:, :n0])
        out_b.backward(grad_out[:, n0:])
        grad_ref = torch.cat([x_a.grad, x_b.grad], dim=1)

        assert x_masked.grad is not None
        # On CUDA the core runs in bf16 (`attn_core_dtype`): ~1e-3 roundoff.
        tol = 5e-3 if device.type == "cuda" else 1e-4
        torch.testing.assert_close(x_masked.grad, grad_ref, rtol=tol, atol=tol)


# --------------------------------------------------------------------------- #
# DistributedAttention, plain (non-ring, non-Ulysses) core
# --------------------------------------------------------------------------- #
class TestDistributedAttentionMask:
    """``attn_backend`` in ``{"sdpa", "math"}``: the only cores that support a mask."""

    n0, n1 = 3, 5

    def _model(
        self, backend: str, device: torch.device, group: dist.ProcessGroup
    ) -> attention.DistributedAttention:
        torch.manual_seed(0)
        model = attention.DistributedAttention(
            in_dim=DIM,
            embed_dim=DIM,
            out_dim=DIM,
            heads=HEADS,
            dropout_p=0.0,
            local_rank=0,
            device=device,
            spatial_group=group,
            channel_group=group,
            ring=False,
            attn_backend=backend,
        )
        model.eval()
        return model

    @pytest.mark.parametrize("backend", ["sdpa", "math"])
    def test_block_mask_equals_separate_attention(
        self, device: torch.device, solo_group, backend: str
    ) -> None:
        model = self._model(backend, device, solo_group)
        n0, n1 = self.n0, self.n1
        torch.manual_seed(1)
        x = torch.randn(2, n0 + n1, DIM, device=device)
        mask = _block_mask(n0, n1).to(device)

        out_masked = model(x, attn_mask=mask)
        out_ref = torch.cat([model(x[:, :n0]), model(x[:, n0:])], dim=1)

        # On CUDA the core runs in bf16 (`attn_core_dtype`): ~1e-3 roundoff.
        tol = 5e-3 if device.type == "cuda" else 1e-5
        torch.testing.assert_close(out_masked, out_ref, rtol=tol, atol=tol)

    @pytest.mark.parametrize("backend", ["sdpa", "math"])
    def test_matches_serial_attention(
        self, device: torch.device, solo_group, backend: str
    ) -> None:
        """Parity with the serial ``Attention`` module, masked, forward and backward."""
        torch.manual_seed(2)
        serial = attention.Attention(DIM, DIM, DIM, HEADS, device)
        model = attention.Attention.convert_to_jigsaw(
            serial,
            {"row": solo_group, "col": solo_group, "model": solo_group},
            device,
            ring=False,
            attn_backend=backend,
        )

        n0, n1 = self.n0, self.n1
        torch.manual_seed(3)
        x_global = torch.randn(2, n0 + n1, DIM, device=device)
        mask = _block_mask(n0, n1).to(device)

        x_ref = x_global.clone().requires_grad_(True)
        y_ref = serial(x_ref, attn_mask=mask)
        grad_out = torch.randn_like(y_ref)
        y_ref.backward(grad_out)

        x = x_global.clone().requires_grad_(True)
        y = model(x, attn_mask=mask)
        y.backward(grad_out)

        tol = 5e-3 if device.type == "cuda" else 1e-4
        torch.testing.assert_close(y, y_ref.detach(), rtol=tol, atol=tol)
        assert x.grad is not None and x_ref.grad is not None
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=tol, atol=tol)

    def test_no_mask_is_unaffected(self, device: torch.device, solo_group) -> None:
        model = self._model("sdpa", device, solo_group)
        torch.manual_seed(1)
        x = torch.randn(2, self.n0 + self.n1, DIM, device=device)
        torch.testing.assert_close(model(x), model(x, attn_mask=None))


# --------------------------------------------------------------------------- #
# cores that do not support a mask: raise, not a silent fallback or a no-op
# --------------------------------------------------------------------------- #
class TestAttnMaskGuards:
    """Unit-level checks of the ``ValueError`` guards.

    Called directly on the core method rather than through ``forward`` so these run
    without TE, a GPU, or an actual multi-rank ring/Ulysses topology: the guard is the
    first thing each method does, before touching any of that.
    """

    def _model(self, device: torch.device, group: dist.ProcessGroup):
        return attention.DistributedAttention(
            in_dim=DIM,
            embed_dim=DIM,
            out_dim=DIM,
            heads=HEADS,
            dropout_p=0.0,
            local_rank=0,
            device=device,
            spatial_group=group,
            channel_group=group,
            ring=False,
            attn_backend="sdpa",
        )

    def test_te_core_rejects_mask(self, device: torch.device, solo_group) -> None:
        model = self._model(device, solo_group)
        q = k = v = torch.randn(2, HEADS, 4, DIM // HEADS, device=device)
        with pytest.raises(ValueError, match="attn_backend='te'"):
            model._transformer_attn_core(
                q, k, v, attn_mask=torch.ones(4, 4, dtype=torch.bool)
            )

    def test_ring_core_rejects_mask(self, device: torch.device, solo_group) -> None:
        model = self._model(device, solo_group)
        q = k = v = torch.randn(2, HEADS, 4, DIM // HEADS, device=device)
        x = torch.randn(2, 4, DIM, device=device)
        with pytest.raises(ValueError, match="ring attention does not support"):
            model._ring_attn_core(
                q, k, v, x, attn_mask=torch.ones(4, 4, dtype=torch.bool)
            )

    def test_ulysses_core_rejects_mask(self, device: torch.device, solo_group) -> None:
        model = self._model(device, solo_group)
        q = k = v = torch.randn(2, HEADS, 4, DIM // HEADS, device=device)
        with pytest.raises(ValueError, match="ulysses attention does not support"):
            model._ulysses_attn_core(
                q, k, v, attn_mask=torch.ones(4, 4, dtype=torch.bool)
            )


@pytest.mark.dist_required
class TestAttnMaskDistributedRaises:
    """``forward`` end to end on an actual multi-rank ring / Ulysses topology."""

    @pytest.mark.dist(4)
    def test_ring_forward_rejects_mask(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        mesh = process_mesh_dict_4x1
        model = attention.DistributedAttention(
            in_dim=DIM,
            embed_dim=DIM,
            out_dim=DIM,
            heads=HEADS,
            dropout_p=0.0,
            local_rank=mesh["col"].rank(),
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            ring=True,
        )
        x = torch.randn(2, 4, DIM, device=device)
        with pytest.raises(ValueError, match="ring attention does not support"):
            model(x, attn_mask=torch.ones(4, 4, dtype=torch.bool, device=device))

    @pytest.mark.dist(4)
    def test_ulysses_forward_rejects_mask(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        mesh = process_mesh_dict_4x1
        model = attention.DistributedAttention(
            in_dim=DIM,
            embed_dim=DIM,
            out_dim=DIM,
            heads=HEADS,
            dropout_p=0.0,
            local_rank=mesh["col"].rank(),
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            ring=False,
            ulysses=True,
        )
        x = torch.randn(2, 4, DIM, device=device)
        with pytest.raises(ValueError, match="ulysses attention does not support"):
            model(x, attn_mask=torch.ones(4, 4, dtype=torch.bool, device=device))


# --------------------------------------------------------------------------- #
# TE present (container + GPU): confirm the real backend, still masked, still raises
# --------------------------------------------------------------------------- #
@pytest.mark.te_required
@pytest.mark.gpu_required
class TestAttnMaskWithTe:
    def test_te_forward_rejects_mask(self, device: torch.device) -> None:
        if not dist.is_initialized():
            pytest.skip("needs an initialized process group")
        me = dist.get_rank()
        group = None
        for r in range(dist.get_world_size()):
            g = dist.new_group(ranks=[r])
            if r == me:
                group = g

        model = attention.DistributedAttention(
            in_dim=DIM,
            embed_dim=DIM,
            out_dim=DIM,
            heads=HEADS,
            dropout_p=0.0,
            local_rank=0,
            device=device,
            spatial_group=group,
            channel_group=group,
            ring=False,
            attn_backend="te",
        )
        assert model.attn_backend is cfg.AttnBackend.TE  # the real core, not a fallback

        # x stays in the module's own (float32) dtype -- `attn_core_dtype` casts
        # internally to bf16 for the core; it is not a dtype the caller must match.
        x = torch.randn(2, 8, DIM, device=device)
        with pytest.raises(ValueError, match="attn_backend='te'"):
            model(x, attn_mask=torch.ones(8, 8, dtype=torch.bool, device=device))
