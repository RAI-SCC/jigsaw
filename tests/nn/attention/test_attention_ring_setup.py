"""Ring wiring of ``DistributedAttention``.

``test_attention_module.py`` runs ``_RingAttention`` through autograd and the
non-ring modules, and ``test_attention_convert.py`` converts a serial module with
``ring=True`` and compares it end to end. What is left is the glue around the ring: the
construction checks, the ``_ring_attn_core`` wrapper on a stand-in ``self``, and the error
for a backend the module does not know.
"""

from types import SimpleNamespace

import pytest
import torch

import jigsaw.nn.attention as attention
from jigsaw.gather import gather_1d_results
from jigsaw.nn.attention import _config as cfg
from tests.parity import assert_close

DIM, HEADS, BATCH, N_LOCAL = 16, 4, 2, 6


class TestUnknownBackend:
    def test_select_attn_core_rejects_an_unhandled_backend(self):
        stub = SimpleNamespace(attn_backend="bogus")
        with pytest.raises(ValueError, match="unhandled attn_backend"):
            attention.DistributedAttention._select_attn_core(stub)

    @pytest.mark.parametrize(
        "backend, name",
        [
            (cfg.AttnBackend.MATH, "_attn_core_math"),
            (cfg.AttnBackend.SDPA, "_attn_core"),
        ],
    )
    def test_select_attn_core_returns_the_backends_core(self, backend, name):
        stub = SimpleNamespace(
            attn_backend=backend, _attn_core_math="math", _attn_core="sdpa"
        )
        chosen = attention.DistributedAttention._select_attn_core(stub)
        assert chosen == getattr(stub, name)


class TestSerialAttention:
    @pytest.mark.parametrize("qk_norm", [False, True])
    def test_qk_norm_adds_one_norm_per_head_dim(self, device, qk_norm):
        serial = attention.Attention(DIM, DIM, DIM, HEADS, device, qk_norm=qk_norm)
        assert hasattr(serial, "q_norm") == qk_norm
        assert hasattr(serial, "k_norm") == qk_norm
        if qk_norm:
            assert serial.q_norm.weight.shape == (DIM // HEADS,)
            assert serial.k_norm.weight.shape == (DIM // HEADS,)


@pytest.mark.dist_required
class TestRingConstruction:
    @pytest.mark.dist(4)
    def test_ring_module_selects_the_ring_core(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        mesh = process_mesh_dict_4x1
        model = attention.DistributedAttention(
            in_dim=DIM,
            embed_dim=DIM,
            out_dim=DIM,
            heads=HEADS,
            dropout_p=0.0,
            local_rank=mesh["model"].rank(),
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            ring=True,
            attn_backend="math",
        )
        assert model.ring
        assert model.attn_core == model._ring_attn_core

    @pytest.mark.dist(2)
    def test_ring_needs_more_than_one_spatial_rank(
        self, device: torch.device, process_mesh_dict_1x2: dict
    ) -> None:
        mesh = process_mesh_dict_1x2  # row group of one rank, channel axis split
        with pytest.raises(AssertionError, match="spatial size is 1"):
            attention.DistributedAttention(
                in_dim=DIM,
                embed_dim=DIM,
                out_dim=DIM,
                heads=HEADS,
                dropout_p=0.0,
                local_rank=mesh["model"].rank(),
                device=device,
                spatial_group=mesh["row"],
                channel_group=mesh["col"],
                ring=True,
                attn_backend="math",
            )


@pytest.mark.dist_required
class TestRingAttnCore:
    @pytest.mark.dist(4)
    def test_matches_global_sdpa(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        mesh = process_mesh_dict_4x1
        ring = mesh["row"]
        rank, world = ring.rank(), ring.size()
        d = DIM // HEADS
        ring_config = cfg.RingAttentionConfig(
            device=device, attn_backend="math", gpus_per_node=world
        )

        torch.manual_seed(0)  # same weights and data on every rank
        w_qkv = torch.randn(3 * HEADS * d, DIM, device=device) / DIM**0.5
        x_global = torch.randn(BATCH, world * N_LOCAL, DIM, device=device)
        sl = slice(rank * N_LOCAL, (rank + 1) * N_LOCAL)
        x = x_global[:, sl].contiguous()

        def project(x_):
            qkv = (x_ @ w_qkv.mT).reshape(BATCH, x_.shape[1], 3, HEADS, d)
            return [t.contiguous() for t in qkv.permute(2, 0, 3, 1, 4)]

        stand_in = SimpleNamespace(
            qkv=SimpleNamespace(weight=w_qkv),
            scale=d**-0.5,
            dropout_p=0.5,  # ignored outside training
            training=False,
            spatial_group=ring,
            local_rank=rank,
            config=ring_config,
        )
        q, k, v = project(x)
        out = attention.DistributedAttention._ring_attn_core(stand_in, q, k, v, x)

        q_r, k_r, v_r = project(x_global)
        ref = torch.nn.functional.scaled_dot_product_attention(q_r, k_r, v_r)
        out_full = gather_1d_results(out.detach(), mesh, dim=2, pm_key="row")
        assert_close("output", out_full, ref, 1e-4, 1e-4, device, ring)
