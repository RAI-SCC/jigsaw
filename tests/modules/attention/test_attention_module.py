"""Multi-rank checks of the attention layer and of ``_RingAttention`` (CPU).

* ``_RingAttention`` is driven through ``autograd`` (forward, ``setup_context``,
  ``backward``) on a 4-rank ring and compared with plain SDPA on the global sequence;
  ``test_ring_attention.py`` only calls ``ring_forward`` / ``ring_backward`` directly.
* ``DistributedAttention`` non-ring paths that the other files do not reach --
  ``test_backend_matches_serial`` across all three ``attn_backend``s (math,
  sdpa -- default and forced-efficient-attention -- te -- both the cuDNN and
  the ``flash_attn`` cores), ``qk_norm`` -- plus the single-process helpers.

``DistributedAttention(ring=True)`` end to end is covered by ``test_attention_convert.py``.
By default the ring recomputes K/V as ``x @ w_qkv.mT`` on the local shard, which is only
right when the channel axis is not split, so it then requires a channel group of size 1.
"""

from typing import Optional

import pytest
import torch
import torch.distributed as dist

import jigsaw.modules.attention as attention
from jigsaw.gather import gather_1d_results
from jigsaw.modules.attention import _config as cfg
from jigsaw.modules.attention._attention import attn_core_dtype
from jigsaw.modules.attention._functional.ring import _RingAttention
from tests.parity import all_ranks_ok, assert_close, forced_sdpa_backend

DIM, HEADS, BATCH = 64, 4, 2


def _channel_shard(t: torch.Tensor, mesh: dict) -> torch.Tensor:
    col = mesh["col"]
    c = t.shape[-1] // col.size()
    return t[..., col.rank() * c : (col.rank() + 1) * c].contiguous().clone()


@pytest.mark.dist_required
class TestRingAttentionFunction:
    """``_RingAttention.apply`` on a 4-rank ring versus global SDPA."""

    h, d, n_local = 2, 8, 6

    def _run(self, mesh: dict, device: torch.device, **config_kwargs):
        ring = mesh["row"]
        rank, world = ring.rank(), ring.size()
        h, d, n_local = self.h, self.d, self.n_local
        config_kwargs.setdefault("gpus_per_node", world)
        ring_config = cfg.RingAttentionConfig(
            device=device, attn_backend="math", **config_kwargs
        )

        torch.manual_seed(0)  # same weights everywhere
        w_qkv = torch.randn(3 * h * d, d, device=device) / d**0.5
        w_qkv.requires_grad_(True)
        torch.manual_seed(1)
        x_global = torch.randn(BATCH, world * n_local, d, device=device)
        x = (
            x_global[:, rank * n_local : (rank + 1) * n_local]
            .clone()
            .requires_grad_(True)
        )

        def project(x_, w_):
            qkv = (x_ @ w_.mT).reshape(BATCH, x_.shape[1], 3, h, d)
            return [t.contiguous() for t in qkv.permute(2, 0, 3, 1, 4)]

        q, k, v = project(x, w_qkv)
        scale = d**-0.5
        out = _RingAttention.apply(
            q, k, v, x, w_qkv, scale, 0.0, ring, rank, ring_config
        )[0]
        grad_out = torch.randn(BATCH, h, world * n_local, d, device=device)
        out.backward(grad_out[:, :, rank * n_local : (rank + 1) * n_local])

        # Global reference with plain autograd.
        x_ref = x_global.clone().requires_grad_(True)
        w_ref = w_qkv.detach().clone().requires_grad_(True)
        q_r, k_r, v_r = project(x_ref, w_ref)
        out_ref = torch.nn.functional.scaled_dot_product_attention(q_r, k_r, v_r)
        out_ref.backward(grad_out)

        out_full = gather_1d_results(out.detach(), mesh, dim=2, pm_key="row")
        sl = slice(rank * n_local, (rank + 1) * n_local)
        assert_close("output", out_full, out_ref.detach(), 1e-4, 1e-4, device, ring)
        assert x.grad is not None and x_ref.grad is not None
        assert_close("x grad", x.grad, x_ref.grad[:, sl], 1e-4, 1e-4, device, ring)
        # The weight gradient is a per-rank partial sum over the local tokens.
        w_grad = w_qkv.grad.detach().clone()
        dist.all_reduce(w_grad, group=ring)
        assert_close("w grad", w_grad, w_ref.grad, 1e-4, 1e-4, device, ring)

    @pytest.mark.dist(4)
    def test_default_policy(self, device, process_mesh_dict_4x1):
        self._run(process_mesh_dict_4x1, device)

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(send_in_forward=cfg.SendForward.KV),
            dict(send_op=cfg.SendOp.P2P, save_scope=cfg.SaveScope.LOCAL),
            dict(send_backward=cfg.SendBackward.GATHER_DKV),
            dict(
                send_op=cfg.SendOp.BROADCAST,
                send_backward=cfg.SendBackward.GATHER_DKV,
                save_scope=cfg.SaveScope.LOCAL,
            ),
            dict(save_for_backward=cfg.SaveBackward.X),
        ],
        ids=["send_kv", "p2p_local", "gather_dkv", "broadcast_gather", "save_x"],
    )
    def test_policy_variants(self, device, process_mesh_dict_4x1, kwargs):
        self._run(process_mesh_dict_4x1, device, **kwargs)

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(send_op=cfg.SendOp.BROADCAST),
            dict(
                send_op=cfg.SendOp.BROADCAST,
                send_backward=cfg.SendBackward.GATHER_DKV,
                save_scope=cfg.SaveScope.LOCAL,
            ),
            dict(send_op=cfg.SendOp.HYBRID_BROADCAST),
            dict(send_op=cfg.SendOp.HYBRID_BROADCAST, gpus_per_node=1),
        ],
        ids=["broadcast", "broadcast_gather", "hybrid_one_node", "hybrid_two_nodes"],
    )
    def test_broadcast_on_replicated_ring(
        self, device, process_mesh_dict_2x2x1, kwargs
    ):
        """The ring of the second replica does not hold global ranks ``0..n-1``."""
        self._run(process_mesh_dict_2x2x1, device, **kwargs)

    @pytest.mark.dist(4)
    def test_hybrid_topology_two_nodes(self, device, process_mesh_dict_4x1):
        """A 4-ring split into 2 nodes of 2 uses the intra/inter-node hybrid path."""
        self._run(process_mesh_dict_4x1, device, gpus_per_node=2)


#: Bound-method attribute each ``attn_backend`` wires ``model.attn_core`` to,
#: per ``DistributedAttention._select_attn_core``.
_ATTN_CORE_ATTR = {
    "math": "_attn_core_math",
    "sdpa": "_attn_core",
    "te": "_transformer_attn_core",
}

#: (attn_backend, variant) cases for the backend-parity test below. ``variant``
#: is ``None`` for the default dispatch, otherwise it pins a specific kernel
#: that the default priority order would not otherwise reach on every GPU:
#:
#: * ``("te", "cudnn" | "flash")`` -- TE's two arms (cuDNN fused attention vs.
#:   the ``flash_attn`` package), selected via ``JIGSAW_TE_CORE`` rather than
#:   letting ``te_cores.select_core`` pick by GPU architecture. Each is
#:   skipped at runtime if that arm's package is not importable
#:   (``te_required`` only guarantees ``transformer_engine`` itself, not
#:   ``flash_attn``).
#: * ``("sdpa", "efficient")`` -- torch's own memory-efficient attention
#:   kernel (``SDPBackend.EFFICIENT_ATTENTION``), forced via
#:   :func:`~tests.parity.forced_sdpa_backend`. ``_attn_core``'s own
#:   ``sdpa_kernel`` priority list is ``[FLASH, CUDNN, EFFICIENT, MATH]``, so
#:   plain ``attn_backend="sdpa"`` with no forcing exercises flash (or cuDNN)
#:   on every GPU this suite runs on -- efficient-attention has no other
#:   coverage without this.
_BACKEND_CASES = [
    ("math", None),
    ("sdpa", None),
    pytest.param(
        "sdpa", "efficient", id="sdpa-efficient", marks=[pytest.mark.gpu_required]
    ),
    pytest.param(
        "te",
        "cudnn",
        id="te-cudnn",
        marks=[pytest.mark.te_required, pytest.mark.gpu_required],
    ),
    pytest.param(
        "te",
        "flash",
        id="te-flash",
        marks=[pytest.mark.te_required, pytest.mark.gpu_required],
    ),
]


@pytest.mark.dist_required
class TestDistributedAttentionPaths:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("attn_backend,variant", _BACKEND_CASES)
    def test_backend_matches_serial(
        self,
        device: torch.device,
        process_mesh_dict_2x1x2: dict,
        attn_backend: str,
        variant: Optional[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Channel-only split (no ring), every backend/kernel variant, matches a serial reference.

        The one non-ring, non-Ulysses parity check that exercises ``te`` and
        forced-``efficient`` ``sdpa``: the other backend coverage of this path
        (``test_attn_backend.py``) only checks that ``te`` is *wired* on a
        1-rank group, never that a sharded, multi-rank run agrees with
        anything. The serial reference always runs the *default*,
        unforced ``sdpa`` priority (flash, typically) regardless of
        ``variant`` -- that is the independent ground truth every other
        kernel is checked against. ``te``/``efficient`` run their own kernel
        rather than that default one, so they get the same looser tolerance
        ``test_te_attn_forward_backward_matches_sdpa`` uses for TE.
        """
        if attn_backend == "te" and variant is not None:
            from jigsaw.modules.attention._functional.cores.te import cudnn, flash

            available = (
                cudnn.CUDNN_CORE_AVAILABLE
                if variant == "cudnn"
                else flash.FLASH_CORE_AVAILABLE
            )
            if not available:
                pytest.skip(f"te core {variant!r} not importable in this environment")
            monkeypatch.setenv("JIGSAW_TE_CORE", variant)

        mesh = process_mesh_dict_2x1x2
        torch.manual_seed(1)
        serial = attention.Attention(DIM, DIM, DIM, HEADS, device)
        x_global = torch.randn(1, 10, DIM, device=device)
        y_ref = serial(x_global)  # unforced default sdpa priority -- the ground truth
        model = attention.Attention.convert_to_jigsaw(
            serial, mesh, device, ring=False, attn_backend=attn_backend
        )
        assert model.attn_core == getattr(model, _ATTN_CORE_ATTR[attn_backend])

        x_local = _channel_shard(x_global, mesh)
        if attn_backend == "sdpa" and variant == "efficient":
            with forced_sdpa_backend(
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
            ) as forced:
                out = gather_1d_results(model(x_local), mesh, dim=-1, pm_key="col")
            # Self-verifying: the forcing has to have actually been consulted,
            # not silently no-op'd (e.g. if a future refactor stopped looking
            # up ``sdpa_kernel`` fresh on ``torch.nn.attention`` per call, in
            # which case this would quietly go back to testing plain ``sdpa``).
            # Reduced, not a bare assert: this sits between collectives (the
            # gather above, assert_close below), so one rank raising alone
            # would hang the rest.
            if not all_ranks_ok(forced.count > 0, device, mesh["model"]):
                raise AssertionError(
                    "forced_sdpa_backend intercepted no sdpa_kernel dispatch -- "
                    "the efficient kernel was never pinned, so this ran the "
                    "default priority list"
                )
        else:
            out = gather_1d_results(model(x_local), mesh, dim=-1, pm_key="col")

        if variant is not None:
            tol = 2e-2
        else:
            # On CUDA both run in bf16, the serial module on the flash kernel.
            tol = 5e-3 if device.type == "cuda" else 1e-4
        assert_close("output", out, y_ref.detach(), tol, tol, device, mesh["model"])

    @pytest.mark.dist(4)
    def test_qk_norm_convert_matches_serial(
        self, device: torch.device, process_mesh_dict_2x1x2: dict
    ) -> None:
        """The serial module applies its query/key norms and conversion keeps them."""
        mesh = process_mesh_dict_2x1x2
        torch.manual_seed(1)
        serial = attention.Attention(DIM, DIM, DIM, HEADS, device, qk_norm=True)
        with torch.no_grad():
            for norm in (serial.q_norm, serial.k_norm):
                norm.weight.uniform_(0.5, 1.5)
                norm.bias.uniform_(-0.5, 0.5)
        x_global = torch.randn(1, 10, DIM, device=device)
        y_ref = serial(x_global)

        plain = attention.Attention(DIM, DIM, DIM, HEADS, device)
        plain.load_state_dict(serial.state_dict(), strict=False)
        assert not torch.allclose(plain(x_global), y_ref, atol=1e-2)

        model = attention.Attention.convert_to_jigsaw(
            serial, mesh, device, ring=False, attn_backend="math"
        )
        out = gather_1d_results(
            model(_channel_shard(x_global, mesh)), mesh, dim=-1, pm_key="col"
        )
        # On CUDA both run in bf16, the serial module on the flash kernel.
        tol = 5e-3 if device.type == "cuda" else 1e-4
        assert_close("output", out, y_ref.detach(), tol, tol, device, mesh["model"])

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        "mesh_name, ring",
        [
            ("process_mesh_dict_2x2", False),
            ("process_mesh_dict_4x1", False),
            ("process_mesh_dict_4x1", True),
        ],
        ids=["2x2", "4x1", "4x1_ring"],
    )
    def test_qk_norm_gradients_match_serial(
        self,
        request: pytest.FixtureRequest,
        device: torch.device,
        mesh_name: str,
        ring: bool,
    ) -> None:
        """The norms are replicated over the whole mesh: every rank gets the full gradient."""
        mesh = request.getfixturevalue(mesh_name)
        row = mesh["row"]
        n_local = 6
        torch.manual_seed(1)
        serial = attention.Attention(DIM, DIM, DIM, HEADS, device, qk_norm=True)
        with torch.no_grad():
            for norm in (serial.q_norm, serial.k_norm):
                norm.weight.uniform_(0.5, 1.5)
                norm.bias.uniform_(-0.5, 0.5)
        x_global = torch.randn(BATCH, row.size() * n_local, DIM, device=device)
        grad_global = torch.randn_like(x_global)

        # Without the ring each row attends within its own chunk.
        chunks = [x_global] if ring else x_global.chunk(row.size(), dim=1)
        y_ref = torch.cat([serial(c) for c in chunks], dim=1)
        y_ref.backward(grad_global)

        # The ring has to send the normalized keys instead of rebuilding them.
        config = (
            cfg.RingAttentionConfig(
                device=device,
                attn_backend="math",
                send_in_forward=cfg.SendForward.KV,
                save_for_backward=cfg.SaveBackward.KV,
            )
            if ring
            else None
        )
        model = attention.Attention.convert_to_jigsaw(
            serial, mesh, device, ring=ring, attn_backend="math", config=config
        )
        sl = slice(row.rank() * n_local, (row.rank() + 1) * n_local)
        y = model(_channel_shard(x_global[:, sl], mesh))
        y.backward(_channel_shard(grad_global[:, sl], mesh))

        # On CUDA the attention core runs in bf16.
        tol = 5e-2 if device.type == "cuda" else 1e-4
        group = mesh["model"]
        y_ref_local = _channel_shard(y_ref.detach()[:, sl], mesh)
        assert_close("output", y.detach(), y_ref_local, tol, tol, device, group)
        # ``k_norm.bias`` is a structurally dead parameter: it shifts every key
        # by the same vector, which adds the per-query constant ``q_i . b`` to
        # every logit of row ``i`` and so cancels in the softmax. Its gradient
        # is identically zero, so in half precision both sides hold nothing but
        # rounding noise, as large as the tolerance itself. Comparing the two to each other is therefore a parity check
        # with no power *and* one unlucky seed from flaking, and it never
        # tested the replication this method is about (an unsynced gradient
        # would be ~0 too). Assert the invariance it can actually show,
        # measured against a live gradient of the same norm.
        k_bias_grad = model.k_norm.bias.grad
        assert_close(
            "k_norm.bias grad (vanishes: softmax is shift-invariant in K)",
            k_bias_grad,
            torch.zeros_like(k_bias_grad),
            tol * float(serial.k_norm.weight.grad.abs().max()),
            tol,
            device,
            group,
        )
        for norm_name, p_name in (
            ("q_norm", "weight"),
            ("q_norm", "bias"),
            ("k_norm", "weight"),
        ):
            got = getattr(getattr(model, norm_name), p_name).grad
            ref = getattr(getattr(serial, norm_name), p_name).grad
            # Summed over every token, so the bf16 rounding grows with the size
            # of the entries -- the same scaling the qkv gradient needs below.
            # ``tol`` as a flat absolute bound is too tight for large entries.
            atol = tol * max(1.0, float(ref.abs().max()))
            assert_close(
                f"{norm_name}.{p_name} grad", got, ref, atol, tol, device, group
            )
        if mesh["col"].size() == 1:
            # Summed over all tokens, so the bf16 rounding grows with the entries.
            ref = serial.qkv.weight.grad
            atol = tol * max(1.0, float(ref.abs().max()))
            assert_close(
                "qkv grad", model.qkv.weight.grad, ref, atol, tol, device, group
            )

    @pytest.mark.dist(4)
    def test_ring_with_qk_norm_auto_builds_a_kv_config(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        """A ring that rebuilds remote keys from the input would skip the key norm,
        so qk_norm auto-selects the KV-circulating policy, same as a channel split."""
        serial = attention.Attention(DIM, DIM, DIM, HEADS, device, qk_norm=True)
        model = attention.Attention.convert_to_jigsaw(
            serial, process_mesh_dict_4x1, device, ring=True
        )
        assert model.config.send_in_forward is cfg.SendForward.KV
        assert model.config.save_for_backward is cfg.SaveBackward.KV

    @pytest.mark.dist(4)
    def test_qk_norm_forward_backward(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        mesh = process_mesh_dict_2x2
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
            qk_norm=True,
            attn_backend="math",
        )
        x = torch.randn(BATCH, 8, DIM // 2, device=device, requires_grad=True)
        y = model(x)
        # A random (contiguous) upstream gradient: the p2p XWT kernels send
        # ``grad_output`` as is, which ``y.sum().backward()`` would hand them expanded.
        y.backward(torch.randn_like(y))
        assert y.shape == x.shape
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert model.q_norm.weight.grad is not None
        assert model.k_norm.weight.grad is not None


class TestSingleProcessHelpers:
    def test_te_backend_falls_back_to_sdpa_with_warning(self, monkeypatch) -> None:
        from jigsaw.modules.attention import _attention as attn_mod

        monkeypatch.setattr(attn_mod, "JIGSAW_TE_AVAILABLE", False)
        resolve = attn_mod.DistributedAttention._resolve_backend
        with pytest.warns(UserWarning, match="falling back to 'sdpa'"):
            assert resolve(cfg.AttnBackend.TE) is cfg.AttnBackend.SDPA
        assert resolve(cfg.AttnBackend.MATH) is cfg.AttnBackend.MATH

    def test_attn_core_dtype(self) -> None:
        x = torch.zeros(2, dtype=torch.float64)
        assert attn_core_dtype(x) == torch.float64
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert attn_core_dtype(x) == torch.bfloat16

    @pytest.mark.gpu_required
    def test_attn_core_dtype_cuda(self, device: torch.device) -> None:
        """fp32 runs the core in bf16 (flash rejects fp32); half dtypes are kept."""
        for dtype, core in (
            (torch.float32, torch.bfloat16),
            (torch.float16, torch.float16),
            (torch.bfloat16, torch.bfloat16),
        ):
            assert attn_core_dtype(torch.zeros(2, device=device, dtype=dtype)) == core
        x = torch.zeros(2, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert attn_core_dtype(x) == torch.bfloat16

    @pytest.mark.gpu_required
    def test_fp32_core_keeps_small_gradients(self, device: torch.device) -> None:
        """An fp32 model without autocast (so without a loss scaler) keeps the tiny
        gradients of a loss averaged over many outputs: an fp16 core would flush
        everything below 6e-8 to zero, bf16 has fp32's range.
        """
        torch.manual_seed(0)
        model = attention.Attention(32, 32, 32, 4, device)
        out = model(torch.randn(2, 64, 32, device=device))
        out.backward(torch.randn_like(out) * 1e-9)  # dL/dout of a mean over ~1e9
        grad = model.qkv.weight.grad  # reaches qkv only through the core
        assert grad is not None and (grad != 0).float().mean() > 0.9

    @pytest.mark.gpu_required
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sdpa_ring_core_keeps_half_dtype(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """The ring's flash core runs in the dtype it is given, bf16 included."""
        from jigsaw.modules.attention._functional.cores import sdpa

        q, k, v = (torch.randn(1, 2, 16, 8, device=device, dtype=dtype) for _ in "qkv")
        out, lse, _ = sdpa.forward(q, k, v, scale=8**-0.5, dropout_p=0.0)
        assert out.dtype == dtype
        dq, dk, dv = sdpa.backward(
            torch.ones_like(out), q, k, v, out, lse, 8**-0.5, 16, 16
        )
        assert dq.dtype == dk.dtype == dv.dtype == dtype
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.double(), k.double(), v.double()
        )
        torch.testing.assert_close(out.double(), ref, atol=2e-2, rtol=2e-2)
