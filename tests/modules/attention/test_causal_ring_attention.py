"""Causal (autoregressive) attention: the ``math`` core in isolation, then the
full ring forward/backward against a whole-sequence causal SDPA reference.

The ring's causal support (:func:`jigsaw.modules.attention.functional.forward.
ring_forward`, :func:`...backward.ring_backward`) classifies each ring step's
K/V chunk relative to the local rank's own (contiguous, evenly-split) Q chunk:
strictly future chunks are skipped, the one chunk at the same global position
is masked causally, and strictly past chunks are attended in full. This file
checks that classification end-to-end for every non-hybrid ``SendOp`` (the
hybrid variants aren't exercised here -- see ``test_ring_hybrid`` in
``test_ring_attention.py``, whose body is presently unreachable, so hybrid
ring correctness is not currently covered by any test).
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
# math core: causal masking in isolation, no distribution needed
# --------------------------------------------------------------------------- #
class TestMathCoreCausal:
    def test_forward_matches_sdpa_is_causal(self):
        torch.manual_seed(0)
        b, h, s, d = 2, 3, 7, 8
        q, k, v = (torch.randn(b, h, s, d) for _ in range(3))
        scale = d**-0.5

        out, lse, _ = math_core.forward(q, k, v, scale=scale, causal=True)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale
        )

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

        # cross-check the returned logsumexp against an explicit masked softmax
        mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
        scores = (q @ k.mT * scale).masked_fill(mask, float("-inf"))
        torch.testing.assert_close(lse, scores.logsumexp(dim=-1), atol=1e-5, rtol=1e-5)

    def test_forward_causal_false_is_unaffected(self):
        """``causal=False`` (the default) is untouched by the new mask path."""
        torch.manual_seed(1)
        q, k, v = (torch.randn(2, 2, 5, 4) for _ in range(3))
        out_default, lse_default, _ = math_core.forward(q, k, v, scale=0.5)
        out_explicit, lse_explicit, _ = math_core.forward(
            q, k, v, scale=0.5, causal=False
        )
        torch.testing.assert_close(out_default, out_explicit)
        torch.testing.assert_close(lse_default, lse_explicit)

    def test_backward_matches_autograd(self):
        torch.manual_seed(2)
        b, h, s, d = 2, 2, 6, 4
        scale = d**-0.5
        q, k, v = (torch.randn(b, h, s, d, requires_grad=True) for _ in range(3))
        out, lse, _ = math_core.forward(q, k, v, scale=scale, causal=True)
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
            causal=True,
        )

        torch.testing.assert_close(dq, q.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dk, k.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dv, v.grad, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# full ring forward/backward vs a whole-sequence causal SDPA reference
# --------------------------------------------------------------------------- #
class TestRingAttentionCausal:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        "send_op", [config.SendOp.P2P, config.SendOp.ALLTOALL, config.SendOp.BROADCAST]
    )
    @pytest.mark.parametrize(
        "save_scope", [config.SaveScope.LOCAL, config.SaveScope.GLOBAL]
    )
    @pytest.mark.parametrize(
        "send_backward",
        [config.SendBackward.SEND_DKV, config.SendBackward.GATHER_DKV],
    )
    def test_ring_forward_backward_causal(
        self,
        device: torch.device,
        process_mesh_dict_4x1: dict,
        send_op: config.SendOp,
        save_scope: config.SaveScope,
        send_backward: config.SendBackward,
    ) -> None:
        """Causal ring output/grads match a whole-sequence causal SDPA reference.

        Mirrors ``TestRingAttention.test_ring_forward_backward`` in
        ``test_ring_attention.py``, with ``causal=True`` and an ``is_causal=True``
        reference instead of ``is_causal=False``.
        """
        process_mesh_dict = process_mesh_dict_4x1
        ring_config = config.RingAttentionConfig(
            device=device,
            send_op=send_op,
            save_scope=save_scope,
            send_backward=send_backward,
            causal=True,
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

        qkv = x @ w_qkv.mT
        q_local, k_local, v_local = (
            t.contiguous()
            for t in qkv.reshape(b, n_local, 3, h, d).permute(2, 0, 3, 1, 4)
        )
        scale = q_local.shape[-1] ** -0.5

        ring_forward = forward.ring_forward
        out, lse, chunks_to_save, _, _ = ring_forward(
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

        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ):
            out_ref = torch.nn.functional.scaled_dot_product_attention(
                q_ref, k_ref, v_ref, is_causal=True, dropout_p=0.0
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

        local_slice = slice(local_rank * n_local, (local_rank + 1) * n_local)
        assert_close(
            "dq local shard",
            dq_local,
            dq_ref[:, :, local_slice].to(dq_local.dtype),
            atol,
            1e-5,
            device,
            ring_group,
        )
        assert_close(
            "dk local shard",
            dk_local,
            dk_ref[:, :, local_slice].to(dk_local.dtype),
            atol,
            1e-5,
            device,
            ring_group,
        )
        assert_close(
            "dv local shard",
            dv_local,
            dv_ref[:, :, local_slice].to(dv_local.dtype),
            atol,
            1e-5,
            device,
            ring_group,
        )

    @pytest.mark.dist(4)
    def test_causal_skips_future_chunks(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        """Rank 0's causal output only ever depends on its own (chunk-0) K/V.

        A direct check of the "skip" branch: zeroing every *other* rank's local
        x before the ring runs must not change rank 0's output at all, since a
        correct causal ring never attends to a future chunk.
        """
        process_mesh_dict = process_mesh_dict_4x1
        ring_config = config.RingAttentionConfig(
            device=device, send_op=config.SendOp.P2P, causal=True
        )
        local_rank = dist.get_rank(group=process_mesh_dict["row"])
        world_size = dist.get_world_size(group=process_mesh_dict["row"])
        b, h, n_local, d = 2, 2, 8, 8

        torch.manual_seed(local_rank)
        x = torch.randn(b, n_local, d, device=device)
        if local_rank != 0:
            x.zero_()
        w_qkv = torch.randn(3 * d * h, d, device=device) / (d**0.5)
        qkv = x @ w_qkv.mT
        q_local, k_local, v_local = (
            t.contiguous()
            for t in qkv.reshape(b, n_local, 3, h, d).permute(2, 0, 3, 1, 4)
        )
        scale = d**-0.5

        out, _, _, _, _ = forward.ring_forward(
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
        )

        if local_rank == 0:
            ref, _, _ = math_core.forward(q_local, k_local, v_local, scale, causal=True)
            torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
