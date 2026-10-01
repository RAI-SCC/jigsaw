"""Multi-rank checks of the attention layer and of ``RingAttentionFunction`` (CPU).

* ``RingAttentionFunction`` is driven through ``autograd`` (forward, ``setup_context``,
  ``backward``) on a 4-rank ring and compared with plain SDPA on the global sequence;
  ``test_ring_attention.py`` only calls ``ring_forward`` / ``ring_backward`` directly.
* ``DistributedAttention`` non-ring paths that the other files do not reach (the math
  backend, ``qk_norm``), plus the single-process helpers.

``DistributedAttention(ring=True)`` end to end is covered by ``test_attention_convert.py``.
The ring recomputes K/V as ``x @ w_qkv.mT`` on the local shard, which is only right when the
channel axis is not split, so it requires a channel group of size 1.
"""

import pytest
import torch
import torch.distributed as dist

import jigsaw.modules.attention as attention
from jigsaw.modules.attention import config as cfg
from jigsaw.modules.attention.attention import attn_core_dtype
from jigsaw.modules.attention.functional.functional import RingAttentionFunction
from jigsaw.utils_aggregate import gather_1d_results
from tests.parity import assert_close

DIM, HEADS, BATCH = 64, 4, 2


def _channel_shard(t: torch.Tensor, mesh: dict) -> torch.Tensor:
    col = mesh["col"]
    c = t.shape[-1] // col.size()
    return t[..., col.rank() * c : (col.rank() + 1) * c].contiguous().clone()


@pytest.mark.dist_required
class TestRingAttentionFunction:
    """``RingAttentionFunction.apply`` on a 4-rank ring versus global SDPA."""

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
        out = RingAttentionFunction.apply(
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
    def test_hybrid_topology_two_nodes(self, device, process_mesh_dict_4x1):
        """A 4-ring split into 2 nodes of 2 uses the intra/inter-node hybrid path."""
        self._run(process_mesh_dict_4x1, device, gpus_per_node=2)


@pytest.mark.dist_required
class TestDistributedAttentionPaths:
    @pytest.mark.dist(4)
    def test_math_backend_matches_serial(
        self, device: torch.device, process_mesh_dict_2x1x2: dict
    ) -> None:
        """Channel-only split with the unfused math core."""
        mesh = process_mesh_dict_2x1x2
        torch.manual_seed(1)
        serial = attention.Attention(DIM, DIM, DIM, HEADS, device)
        x_global = torch.randn(1, 10, DIM, device=device)
        y_ref = serial(x_global)
        model = attention.Attention.convert_to_jigsaw(
            serial, mesh, device, ring=False, attn_backend="math"
        )
        assert model.attn_core == model._attn_core_math
        out = gather_1d_results(
            model(_channel_shard(x_global, mesh)), mesh, dim=-1, pm_key="col"
        )
        # On CUDA both run in bf16, the serial module on the flash kernel: ~1e-3 apart.
        tol = 5e-3 if device.type == "cuda" else 1e-4
        assert_close("output", out, y_ref.detach(), tol, tol, device, mesh["model"])

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
        from jigsaw.modules.attention import attention as attn_mod

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
