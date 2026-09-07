"""Key-padding attention masks: the ``math`` core in isolation, then the full
ring forward/backward against a whole-sequence masked SDPA reference.

Mirrors ``test_causal_ring_attention.py``. The ring's key-padding support
(:func:`jigsaw.modules.attention.functional.forward.ring_forward`,
:func:`...backward.ring_backward`) all-gathers the local ``[B, n_local]``
mask shards into a ``[B, S_global]`` mask once, then slices out the
``[B, n_local]`` sub-mask for whatever global chunk a ring step currently
holds (the same ``global_chunk_idx`` bookkeeping the causal path uses) and
feeds it into the attention core.
"""

import pytest
import torch
import torch.distributed as dist

import jigsaw.modules.attention.config as config
from jigsaw.modules.attention.functional import backward, forward
from jigsaw.modules.attention.functional.cores import math as math_core
from jigsaw.utils_aggregate import gather_1d_results
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]


# --------------------------------------------------------------------------- #
# math core: key-padding masking in isolation, no distribution needed
# --------------------------------------------------------------------------- #
class TestMathCoreKeyPaddingMask:
    def test_forward_matches_sdpa_attn_mask(self):
        torch.manual_seed(0)
        b, h, s, d = 2, 3, 7, 8
        q, k, v = (torch.randn(b, h, s, d) for _ in range(3))
        scale = d**-0.5

        key_mask = torch.ones(b, s, dtype=torch.bool)
        key_mask[0, -2:] = False  # batch 0: last two keys are padding

        out, lse, _ = math_core.forward(q, k, v, scale=scale, key_mask=key_mask)

        allow = key_mask[:, None, None, :].expand(b, h, s, s)
        add_mask = torch.zeros(b, h, s, s).masked_fill(~allow, float("-inf"))
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=add_mask, scale=scale
        )
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

        scores = (q @ k.mT * scale).masked_fill(~allow, float("-inf"))
        torch.testing.assert_close(lse, scores.logsumexp(dim=-1), atol=1e-5, rtol=1e-5)

    def test_forward_combines_with_causal(self):
        """A causal + padding row with zero valid keys stays finite, not nan.

        Batch 1's first two keys are padding, so under causal masking query
        position 0 (and 1) in that batch has no valid key at all -- the
        "dead row" case ``math.forward``'s ``dead_row`` handling exists for.
        """
        torch.manual_seed(1)
        b, h, s, d = 2, 2, 6, 4
        q, k, v = (torch.randn(b, h, s, d) for _ in range(3))
        scale = d**-0.5

        key_mask = torch.ones(b, s, dtype=torch.bool)
        key_mask[1, :2] = False

        out, lse, _ = math_core.forward(
            q, k, v, scale=scale, causal=True, key_mask=key_mask
        )
        assert not torch.isnan(out).any()

        causal_allow = torch.tril(torch.ones(s, s, dtype=torch.bool))
        allow = causal_allow[None, None] & key_mask[:, None, None, :]
        add_mask = torch.zeros(b, h, s, s).masked_fill(~allow, float("-inf"))
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=add_mask, scale=scale
        )
        assert not torch.isnan(ref).any()

        dead = (~allow).all(dim=-1, keepdim=True).expand_as(out)
        torch.testing.assert_close(
            torch.where(dead, torch.zeros_like(out), out),
            torch.where(dead, torch.zeros_like(ref), ref),
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.isneginf(lse[dead[..., 0]]).all()

    def test_backward_matches_autograd(self):
        torch.manual_seed(2)
        b, h, s, d = 2, 2, 6, 4
        scale = d**-0.5
        key_mask = torch.ones(b, s, dtype=torch.bool)
        key_mask[0, -1] = False

        q, k, v = (torch.randn(b, h, s, d, requires_grad=True) for _ in range(3))
        out, lse, _ = math_core.forward(q, k, v, scale=scale, key_mask=key_mask)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        dq, dk, dv = math_core.backward(
            grad_out,
            q.detach(),
            k.detach(),
            v.detach(),
            out.detach(),
            lse.detach(),
            scale,
            key_mask=key_mask,
        )

        torch.testing.assert_close(dq, q.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dk, k.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dv, v.grad, atol=1e-4, rtol=1e-4)

    def test_backward_dead_row_no_nan(self):
        """A globally-dead query row (causal + padding masks out every key)
        must not poison dK/dV for the rest of the batch via the query-sum
        matmul (``dk = d_attn.mT @ q`` sums over the query dimension)."""
        torch.manual_seed(3)
        b, h, s, d = 2, 2, 5, 4
        scale = d**-0.5
        key_mask = torch.ones(b, s, dtype=torch.bool)
        key_mask[1, 0] = False  # batch 1, query 0's only causal key is itself

        q, k, v = (torch.randn(b, h, s, d, requires_grad=True) for _ in range(3))
        out, lse, _ = math_core.forward(
            q, k, v, scale=scale, causal=True, key_mask=key_mask
        )
        grad_out = torch.randn_like(out)
        dq, dk, dv = math_core.backward(
            grad_out,
            q.detach(),
            k.detach(),
            v.detach(),
            out.detach(),
            lse.detach(),
            scale,
            causal=True,
            key_mask=key_mask,
        )
        for t in (dq, dk, dv):
            assert not torch.isnan(t).any()

        out.backward(grad_out)
        torch.testing.assert_close(dq, q.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dk, k.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dv, v.grad, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# full ring forward/backward vs a whole-sequence masked SDPA reference
# --------------------------------------------------------------------------- #
class TestRingAttentionKeyPaddingMask:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        "send_op", [config.SendOp.P2P, config.SendOp.ALLTOALL, config.SendOp.BROADCAST]
    )
    @pytest.mark.parametrize("causal", [False, True])
    def test_ring_forward_backward_key_padding_mask(
        self,
        device: torch.device,
        process_mesh_dict_4x1: dict,
        send_op: config.SendOp,
        causal: bool,
    ) -> None:
        """Ring output/grads with a key-padding mask match a whole-sequence
        masked SDPA reference, with and without also being causal.

        Mirrors ``TestRingAttentionCausal.test_ring_forward_backward_causal``.
        """
        process_mesh_dict = process_mesh_dict_4x1
        ring_config = config.RingAttentionConfig(
            device=device, send_op=send_op, causal=causal
        )

        local_rank = dist.get_rank(group=process_mesh_dict["row"])
        world_size = dist.get_world_size(group=process_mesh_dict["row"])
        dtype = torch.bfloat16 if dist.get_backend() == "nccl" else torch.float32
        torch.manual_seed(2)
        b, h, n_global, d = 3, 6, world_size * 16, 16
        n_local = n_global // world_size

        w_qkv = torch.randn(
            3 * d * h, d, device=device, requires_grad=True, dtype=dtype
        ) / (d**0.5)
        torch.manual_seed(local_rank)
        x = torch.randn(
            b, n_local, d, device=device, requires_grad=True, dtype=dtype
        ) / (d**0.5)
        x.fill_(local_rank * 0.1)
        x = x.contiguous()

        # Pad out the last global chunk's positions for every batch entry --
        # exercises a step whose *entire* local mask slice is invalid for
        # every rank that ever attends to it.
        torch.manual_seed(100 + local_rank)
        key_padding_mask = torch.ones(b, n_local, dtype=torch.bool, device=device)
        if local_rank == world_size - 1:
            key_padding_mask[:, -2:] = False

        qkv = x @ w_qkv.mT
        q_local, k_local, v_local = (
            t.contiguous()
            for t in qkv.reshape(b, n_local, 3, h, d).permute(2, 0, 3, 1, 4)
        )
        scale = q_local.shape[-1] ** -0.5

        out, lse, chunks_to_save, _, _ = forward.ring_forward(
            q_local,
            k_local,
            v_local,
            x,
            w_qkv,
            scale,
            0.0,
            process_mesh_dict["row"],
            local_rank,
            world_size,
            ring_config,
            key_padding_mask=key_padding_mask,
        )
        out_full = gather_1d_results(out, process_mesh_dict, dim=2)

        q_ref = gather_1d_results(q_local, process_mesh_dict, dim=2).requires_grad_(
            True
        )
        k_ref = gather_1d_results(k_local, process_mesh_dict, dim=2).requires_grad_(
            True
        )
        v_ref = gather_1d_results(v_local, process_mesh_dict, dim=2).requires_grad_(
            True
        )
        global_key_mask = gather_1d_results(
            key_padding_mask, process_mesh_dict, dim=1
        )

        allow = global_key_mask[:, None, None, :].expand(b, h, n_global, n_global)
        if causal:
            causal_allow = torch.tril(
                torch.ones(n_global, n_global, dtype=torch.bool, device=device)
            )
            allow = allow & causal_allow[None, None]
        ref_mask = torch.zeros(b, h, n_global, n_global, device=device, dtype=dtype)
        ref_mask = ref_mask.masked_fill(~allow, float("-inf"))

        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            out_ref = torch.nn.functional.scaled_dot_product_attention(
                q_ref, k_ref, v_ref, attn_mask=ref_mask, dropout_p=0.0
            )

        dout_local = torch.ones_like(out, device=device, dtype=out_ref.dtype)
        dq_local, dk_local, dv_local = backward.ring_backward(
            dout_local,
            q_local,
            out,
            x,
            lse,
            w_qkv,
            scale,
            process_mesh_dict["row"],
            local_rank,
            world_size,
            chunks_to_save,
            ring_config=ring_config,
            key_padding_mask=key_padding_mask,
        )

        out_ref.backward(torch.ones_like(out_ref))
        dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

        atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        ring_group = process_mesh_dict["row"]

        dq_full = gather_1d_results(dq_local, process_mesh_dict, dim=2)
        dk_full = gather_1d_results(dk_local, process_mesh_dict, dim=2)
        dv_full = gather_1d_results(dv_local, process_mesh_dict, dim=2)

        assert_close("output", out_full, out_ref, atol, 1e-5, device, ring_group)
        assert_close(
            "dq", dq_full, dq_ref.to(dq_full.dtype), atol, 1e-5, device, ring_group
        )
        assert_close(
            "dk", dk_full, dk_ref.to(dk_full.dtype), atol, 1e-5, device, ring_group
        )
        assert_close(
            "dv", dv_full, dv_ref.to(dv_full.dtype), atol, 1e-5, device, ring_group
        )
