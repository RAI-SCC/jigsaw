import pytest
import torch
import torch.distributed as dist

import jigsaw.attention_ring as ring_attention
import jigsaw.ring_attention_helpers as ring_attn_fns
from jigsaw.utils_aggregate import (
    gather_1d_results,
)
from tests.process_mesh import *

pytestmark = [pytest.mark.distributed_required]

# def test_backward(device, process_mesh):
#     local_rank = dist.get_rank(group=process_mesh["row"])
#     world_size = dist.get_world_size(group=process_mesh["row"])

#     torch.manual_seed(2)  # same seed for weight initialization
#     B, h, N, d = 1, 3, world_size * 16, 8
#     Nlocal = N // world_size
#     W_qkv = torch.randn(
#         3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     X = torch.randn(
#         B, Nlocal, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     X = X.contiguous()

#     QKV = X @ W_qkv.mT

#     Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 1, 3, 4)
#     # QKV has shape [B, N, h, d]
#     Q_local, K_local, V_local = (
#         Q_local.contiguous(),
#         K_local.contiguous(),
#         V_local.contiguous(),
#     )
#     scale = Q_local.shape[-1] ** -0.5

#     # O ref full
#     O_ref_full, logsumexp, *_ = torch.ops.aten._flash_attention_forward(
#         Q_local,
#         K_local,
#         V_local,
#         cum_seq_q=None,
#         cum_seq_k=None,
#         max_q=Nlocal,
#         max_k=Nlocal,
#         dropout_p=0.0,
#         is_causal=False,
#         scale=scale,
#         return_debug_mask=False,
#     )
#     dQ_ref, dK_ref, dV_ref = (
#         torch.ops.aten._scaled_dot_product_flash_attention_backward(
#             torch.ones(
#                 O_ref_full.shape, device=device, dtype=O_ref_full.dtype
#             ).transpose(1, 2),  # grad_output: (B, N, H, d)
#             Q_local.transpose(1, 2),  # query:       (B, N, H, d)
#             K_local.transpose(1, 2),  # key:         (B, N, H, d)
#             V_local.transpose(1, 2),  # value:       (B, N, H, d)
#             O_ref_full.transpose(1, 2),  # out:         (B, N, H d)
#             logsumexp,  # logsumexp:   (B*H, N) — from forward, reshape(B*H, N) before passing
#             dropout_p=0.0,
#             is_causal=False,
#             scale=scale,
#             cum_seq_q=None,
#             cum_seq_k=None,
#             max_q=Nlocal,
#             max_k=Nlocal,
#             philox_seed=torch.zeros(1, dtype=torch.long, device=Q_local.device),
#             philox_offset=torch.zeros(1, dtype=torch.long, device=Q_local.device),
#         )
#     )

#     # QKV_torchflash
#     flash_Q = Q_local.detach().float().requires_grad_(True).transpose(1, 2)
#     flash_K = K_local.detach().float().requires_grad_(True).transpose(1, 2)
#     flash_V = V_local.detach().float().requires_grad_(True).transpose(1, 2)

#     with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
#         flash_out = torch.nn.functional.scaled_dot_product_attention(
#             flash_Q, flash_K, flash_V, is_causal=False, dropout_p=0, attn_mask=None
#         )
#         # Qg grad
#         flash_Q.retain_grad()
#         flash_K.retain_grad()
#         flash_V.retain_grad()
#         flash_out.sum().backward()

#     Qg = Q_local.detach().float().requires_grad_(True).transpose(1, 2)
#     Kg = K_local.detach().float().requires_grad_(True).transpose(1, 2)
#     Vg = V_local.detach().float().requires_grad_(True).transpose(1, 2)

#     # Autograd ground truth — definitionally correct
#     q, k, v = Qg, Kg, Vg
#     S_au = (q @ k.transpose(-1, -2)) * scale  # (B, H, N, N)
#     LSE_au = torch.logsumexp(S_au, dim=-1)  # (B, H, N)
#     O_au = (torch.softmax(S_au, dim=-1) @ v).transpose(1, 2)  # (B, N, H, d)
#     q.retain_grad()
#     k.retain_grad()
#     v.retain_grad()
#     O_au.sum().backward()

#     dQ_au, dK_au, dV_au = q.grad.transpose(1, 2), Kg.grad, Vg.grad

#     # q has shape [B, h, N, d]
#     # O_ref_full has shape [B, N, h, d]
#     # O_au has shape [B, h, d, N]
#     # dQ_ref: [B, N, h, d]
#     # dQ_au: [B, N, h, d]

#     # check flash attention implementation against local implementation
#     print(f"flash_V {flash_V.shape}, v grad shape {v.grad.shape}")
#     assert torch.allclose(flash_out.transpose(1, 2), O_au, atol=1e-3)
#     assert torch.allclose(flash_Q.grad, q.grad, atol=1e-3)
#     assert torch.allclose(flash_K.grad, k.grad, atol=1e-3)
#     assert torch.allclose(flash_V.grad, v.grad, atol=1e-3)

#     # check flash attention against torch kernel
#     assert torch.allclose(flash_out, O_ref_full.transpose(1, 2).float(), atol=1e-3)
#     assert torch.allclose(LSE_au, logsumexp.float(), atol=1e-3)
#     assert torch.allclose(flash_Q.grad, dQ_ref.float(), atol=1e-2)
#     assert torch.allclose(flash_K.grad, dK_ref.float(), atol=1e-2)
#     assert torch.allclose(flash_V.grad, dV_ref.float(), atol=1e-2)


@pytest.mark.dist(4)
@pytest.mark.gpu_required
@pytest.mark.parametrize(
    "flash_kernel",
    [
        (flash_kernel)
        for flash_kernel in [
            torch.nn.attention.SDPBackend.MATH,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        ]
    ],
)
def test_backward_sdpa(
    device: torch.device,
    process_mesh_dict_4x1: dict,
    flash_kernel: torch.nn.attention.SDPBackend,
) -> None:
    """Test flash kernel with direct call to aten function."""
    world_size = dist.get_world_size(group=process_mesh_dict_4x1["row"])

    torch.manual_seed(2)  # same seed for weight initialization
    b, h, n_global, d = 1, 6, world_size * 32, 256
    n_local = n_global // world_size
    w_qkv = torch.randn(
        3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    x = torch.randn(
        b, n_local, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)

    qkv = x @ w_qkv.mT

    q_local, k_local, v_local = qkv.reshape(b, n_local, 3, h, d).permute(2, 0, 3, 1, 4)
    # QKV has shape [B, N, h, d]
    q_local, k_local, v_local = (
        q_local.contiguous(),
        k_local.contiguous(),
        v_local.contiguous(),
    )
    scale = q_local.shape[-1] ** -0.5

    # O ref full
    out_ref, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        q_local,
        k_local,
        v_local,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        return_debug_mask=False,
    )
    dq_ref, dk_ref, dv_ref = (
        torch.ops.aten._scaled_dot_product_flash_attention_backward(
            torch.ones(
                out_ref.shape, device=device, dtype=out_ref.dtype
            ),  # grad_output: (B, N, H, d)
            q_local,  # query:       (B, N, H, d)
            k_local,  # key:         (B, N, H, d)
            v_local,  # value:       (B, N, H, d)
            out_ref,  # out:         (B, N, H d)
            logsumexp,  # logsumexp:   (B*H, N) — from forward, reshape(B*H, N) before passing
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            cum_seq_q=None,
            cum_seq_k=None,
            max_q=n_local,
            max_k=n_local,
            philox_seed=torch.zeros(1, dtype=torch.long, device=q_local.device),
            philox_offset=torch.zeros(1, dtype=torch.long, device=q_local.device),
        )
    )

    # QKV_torchflash
    torch_ref_q = q_local.detach().requires_grad_(True)
    torch_ref_k = k_local.detach().requires_grad_(True)
    torch_ref_v = v_local.detach().requires_grad_(True)

    with torch.nn.attention.sdpa_kernel(flash_kernel):
        torch_ref_out = torch.nn.functional.scaled_dot_product_attention(
            torch_ref_q,
            torch_ref_k,
            torch_ref_v,
            is_causal=False,
            dropout_p=0,
            attn_mask=None,
        )
        # Qg grad
        torch_ref_q.retain_grad()
        torch_ref_k.retain_grad()
        torch_ref_v.retain_grad()
        torch_ref_out.sum().backward()

    q_naive = q_local.detach().requires_grad_(True)
    k_naive = k_local.detach().requires_grad_(True)
    v_naive = v_local.detach().requires_grad_(True)

    # Autograd ground truth — definitionally correct
    s_naive = (q_naive @ k_naive.transpose(-1, -2)) * scale  # (B, H, N, N)
    lse_naive = torch.logsumexp(s_naive, dim=-1)  # (B, H, N)
    out_naive = torch.softmax(s_naive, dim=-1) @ v_naive  # (B, N, H, d)
    q_naive.retain_grad()
    k_naive.retain_grad()
    v_naive.retain_grad()
    out_naive.sum().backward()

    # check flash attention implementation against local implementation
    assert torch.allclose(torch_ref_out, out_naive, atol=1e-2)
    assert torch.allclose(torch_ref_q.grad, q_naive.grad, atol=1e-2)
    assert torch.allclose(torch_ref_k.grad, k_naive.grad, atol=1e-2)
    assert torch.allclose(torch_ref_v.grad, v_naive.grad, atol=1e-2)

    # check flash attention against torch kernel
    assert torch.allclose(torch_ref_out, out_ref, atol=1e-3)
    assert torch.allclose(lse_naive.float(), logsumexp, atol=1e-1)
    assert torch.allclose(torch_ref_q.grad, dq_ref, atol=1e-2)
    assert torch.allclose(torch_ref_k.grad, dk_ref, atol=1e-2)
    assert torch.allclose(torch_ref_v.grad, dv_ref, atol=1e-2)


@pytest.mark.dist(4)
@pytest.mark.parametrize("send_in_forward", [ring_attn_fns.SendForward.KV])
@pytest.mark.parametrize("save_for_backward", [ring_attn_fns.SaveBackward.KV])
@pytest.mark.parametrize(
    "send_forward_op",
    [ring_attn_fns.SendForwardOp.SENDRECV],
)
@pytest.mark.parametrize("save_scope", [ring_attn_fns.SaveScope.LOCAL])
@pytest.mark.parametrize(
    "send_backward",
    [ring_attn_fns.SendBackward.SEND_DKV],
)
@pytest.mark.parametrize("offloader", [ring_attn_fns.Offload.DISABLED])
def test_ring_forward_backward_tmp(
    device: torch.device,
    process_mesh_dict_4x1: dict,
    send_in_forward: ring_attn_fns.SendForward,
    save_for_backward: ring_attn_fns.SaveBackward,
    send_forward_op: ring_attn_fns.SendForwardOp,
    save_scope: ring_attn_fns.SaveScope,
    send_backward: ring_attn_fns.SendBackward,
    offloader: ring_attn_fns.Offload,
) -> None:
    """Verify weights for XWT_Sync are synchronized after a few iterations."""
    if (
        send_in_forward == ring_attn_fns.SendForward.KV
        and save_for_backward == ring_attn_fns.SaveBackward.X
    ):
        pytest.xfail("not yet supported")

    process_mesh_dict = process_mesh_dict_4x1
    ring_config = ring_attn_fns.RingAttentionConfig(
        device=device,
        send_in_forward=send_in_forward,
        send_forward_op=send_forward_op,
        save_for_backward=save_for_backward,
        save_scope=save_scope,
        send_backward=send_backward,
        offload=offloader,
    )

    ring_forward = ring_attn_fns.build_ring_forward(ring_config)
    local_rank = dist.get_rank(group=process_mesh_dict["row"])
    world_size = dist.get_world_size(group=process_mesh_dict["row"])

    torch.manual_seed(2)  # same seed for weight initialization
    b, h, n_global, d = 3, 6, world_size * 16, 64
    n_local = n_global // world_size
    w_qkv = torch.randn(
        3 * d * h, d, device=device, requires_grad=True, dtype=torch.float32
    ) / (d**0.5)
    torch.manual_seed(local_rank)  # different seed for input data generation
    x = torch.randn(
        b, n_local, d, device=device, requires_grad=True, dtype=torch.float32
    ) / (d**0.5)
    x = x.contiguous()
    empty = torch.empty(0, device=x.device, dtype=x.dtype)

    qkv = x @ w_qkv.mT
    # QKV shape: [B, Nlocal, 3*d*h]
    # We need QKV to have shape: []
    q_local, k_local, v_local = qkv.reshape(b, n_local, 3, h, d).permute(2, 0, 3, 1, 4)

    q_local, k_local, v_local = (
        q_local.contiguous(),
        k_local.contiguous(),
        v_local.contiguous(),
    )
    scale = q_local.shape[-1] ** -0.5

    # just test helper function forward
    ring_forward = ring_attn_fns.build_ring_forward(ring_config)
    out, lse, chunks_to_save = ring_forward(
        q_local,
        k_local,
        v_local,
        x,
        w_qkv,
        scale,
        0.0,
        process_mesh_dict["row"],
        empty,
        local_rank,
        world_size,
        ring_config,
    )

    out_full = gather_1d_results(out, process_mesh_dict, dim=2)
    x_full = gather_1d_results(x, process_mesh_dict, dim=1)
    qkv = x_full @ w_qkv.mT

    q_ref = gather_1d_results(q_local, process_mesh_dict, dim=2).requires_grad_(
        True
    )
    k_ref = gather_1d_results(k_local, process_mesh_dict, dim=2).requires_grad_(
        True
    )
    v_ref = gather_1d_results(v_local, process_mesh_dict, dim=2).requires_grad_(
        True
    )

    # Reference calculation
    # ring_attention.ring_attn.build_ring_forward(ring_attention)
    with torch.nn.attention.sdpa_kernel(
        [
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]
    ):
        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, is_causal=False, dropout_p=0.0
        )

    q_ref2, k_ref2, v_ref2 = (
        q_ref.detach().clone(),
        k_ref.detach().clone(),
        v_ref.detach().clone(),
    )
    q_ref2 = q_ref2.requires_grad_(True)
    k_ref2 = k_ref2.requires_grad_(True)
    v_ref2 = v_ref2.requires_grad_(True)
    out_core = ring_config.attn_core_forward(q_ref2, k_ref2, v_ref2, dropout_p=0.0)[0]
    out_core = out_core.requires_grad_(True)

    dout_local = torch.ones_like(out, device=device, dtype=out_ref.dtype)

    dq_local, dk_local, dv_local = ring_attn_fns.ring_backward(
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
        empty=empty,
    )

    # Backward pass
    out_ref.backward(torch.ones_like(out_ref))
    dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

    out_core.backward(torch.ones_like(out_core))
    assert torch.allclose(dq_ref, q_ref2.grad, atol=1e-4)
    assert torch.allclose(dk_ref, k_ref2.grad, atol=1e-4)
    assert torch.allclose(dv_ref, v_ref2.grad, atol=1e-4)

    # gather
    dq_full = gather_1d_results(dq_local, process_mesh_dict, dim=2)
    dk_full = gather_1d_results(dk_local, process_mesh_dict, dim=2)
    dv_full = gather_1d_results(dv_local, process_mesh_dict, dim=2)

    assert torch.allclose(out_full, out_ref, atol=1e-3)
    assert torch.allclose(dq_full, dq_ref.to(dq_full.dtype), atol=1e-3)
    assert torch.allclose(dk_full, dk_ref.to(dk_full.dtype), atol=1e-3)
    assert torch.allclose(dv_full, dv_ref.to(dv_full.dtype), atol=1e-3)

    assert torch.allclose(
        dq_local,
        dq_ref[:, :, (local_rank * n_local) : (local_rank + 1) * n_local].to(
            dq_local.dtype
        ),
        atol=1e-3,
    )
    assert torch.allclose(
        dk_local,
        dk_ref[:, :, (local_rank * n_local) : (local_rank + 1) * n_local].to(
            dk_local.dtype
        ),
        atol=1e-3,
    )
    assert torch.allclose(
        dv_local,
        dv_ref[:, :, (local_rank * n_local) : (local_rank + 1) * n_local].to(
            dv_local.dtype
        ),
        atol=1e-3,
    )
    print(f'this pased')


@pytest.mark.dist(4)
@pytest.mark.parametrize(
    "send_in_forward", [ring_attn_fns.SendForward.X, ring_attn_fns.SendForward.KV]
)
@pytest.mark.parametrize(
    "save_for_backward", [ring_attn_fns.SaveBackward.X, ring_attn_fns.SaveBackward.KV]
)
@pytest.mark.parametrize(
    "send_forward_op",
    [ring_attn_fns.SendForwardOp.SENDRECV, ring_attn_fns.SendForwardOp.BROADCAST],
)
@pytest.mark.parametrize(
    "save_scope", [ring_attn_fns.SaveScope.LOCAL, ring_attn_fns.SaveScope.GLOBAL]
)
@pytest.mark.parametrize(
    "send_backward",
    [ring_attn_fns.SendBackward.SEND_DKV, ring_attn_fns.SendBackward.GATHER_DKV],
)
@pytest.mark.parametrize(
    "offloader", [ring_attn_fns.Offload.DISABLED, ring_attn_fns.Offload.CPU]
)
def test_ring_forward_backward(
    device: torch.device,
    process_mesh_dict_4x1: dict,
    send_in_forward: ring_attn_fns.SendForward,
    save_for_backward: ring_attn_fns.SaveBackward,
    send_forward_op: ring_attn_fns.SendForwardOp,
    save_scope: ring_attn_fns.SaveScope,
    send_backward: ring_attn_fns.SendBackward,
    offloader: ring_attn_fns.Offload,
) -> None:
    """Verify weights for XWT_Sync are synchronized after a few iterations."""
    if (
        send_in_forward == ring_attn_fns.SendForward.KV
        and save_for_backward == ring_attn_fns.SaveBackward.X
    ):
        pytest.xfail("not yet supported")

    ring_config = ring_attn_fns.RingAttentionConfig(
        device=device,
        send_in_forward=send_in_forward,
        send_forward_op=send_forward_op,
        save_for_backward=save_for_backward,
        save_scope=save_scope,
        send_backward=send_backward,
        offload=offloader,
    )

    ring_forward = ring_attn_fns.build_ring_forward(ring_config)
    local_rank = dist.get_rank(group=None)
    world_size = dist.get_world_size(group=None)

    torch.manual_seed(2)  # same seed for weight initialization
    b, h, n_global, d = 3, 6, world_size * 16, 64
    n_local = n_global // world_size
    w_qkv = torch.randn(
        3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    torch.manual_seed(local_rank)  # different seed for input data generation
    x = torch.randn(
        b, n_local, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    x = x.contiguous()
    empty = torch.empty(0, device=x.device, dtype=x.dtype)

    qkv = x @ w_qkv.mT
    # QKV shape: [B, Nlocal, 3*d*h]
    # We need QKV to have shape: []
    q_local, k_local, v_local = qkv.reshape(b, n_local, 3, h, d).permute(2, 0, 3, 1, 4)

    q_local, k_local, v_local = (
        q_local.contiguous(),
        k_local.contiguous(),
        v_local.contiguous(),
    )
    scale = q_local.shape[-1] ** -0.5

    # just test helper function forward
    ring_forward = ring_attn_fns.build_ring_forward(ring_config)
    out, lse, chunks_to_save = ring_forward(
        q_local,
        k_local,
        v_local,
        x,
        w_qkv,
        scale,
        0.0,
        process_mesh_dict_4x1["row"],
        empty,
        local_rank,
        world_size,
        ring_config,
    )

    out_full = gather_1d_results(out, process_mesh_dict_4x1, dim=2)
    x_full = gather_1d_results(x, process_mesh_dict_4x1, dim=1)
    qkv = x_full @ w_qkv.mT

    q_ref = gather_1d_results(q_local, process_mesh_dict_4x1, dim=2).requires_grad_(
        True
    )
    k_ref = gather_1d_results(k_local, process_mesh_dict_4x1, dim=2).requires_grad_(
        True
    )
    v_ref = gather_1d_results(v_local, process_mesh_dict_4x1, dim=2).requires_grad_(
        True
    )

    # Reference calculation
    # ring_attention.ring_attn.build_ring_forward(ring_attention)
    with torch.nn.attention.sdpa_kernel(
        [
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]
    ):
        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, is_causal=False, dropout_p=0.0
        )

    dout_local = torch.ones_like(out, device=device, dtype=out.dtype)

    dq_local, dk_local, dv_local = ring_attn_fns.ring_backward(
        dout_local,
        q_local,
        out,
        x,
        lse,
        w_qkv,
        scale,
        process_mesh_dict_4x1["row"],
        local_rank,
        world_size,
        chunks_to_save,
        ring_config=ring_config,
        empty=empty,
    )

    # Backward pass
    out_ref.backward(torch.ones_like(out_ref))
    dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad
    # dq_ref, dk_ref, dv_ref =
    # (
    #     torch.ops.aten._scaled_dot_product_flash_attention_backward(
    #         torch.ones(
    #             out_ref.shape, device=device, dtype=out_ref.dtype
    #         ),  # grad_output: (B, N, H, d)
    #         q_ref,  # query:       (B, N, H, d)
    #         k_ref,  # key:         (B, N, H, d)
    #         v_ref,  # value:       (B, N, H, d)
    #         out_ref,  # out:         (B, H, N, d)
    #         logsumexp,  # logsumexp:   (B*H, N) — from forward, reshape(B*H, N) before passing
    #         dropout_p=0.0,
    #         is_causal=False,
    #         scale=scale,
    #         cum_seq_q=None,
    #         cum_seq_k=None,
    #         max_q=n_global,
    #         max_k=n_global,
    #         philox_seed=torch.zeros(1, dtype=torch.long, device=q_ref.device),
    #         philox_offset=torch.zeros(1, dtype=torch.long, device=q_ref.device),
    #     )
    # )

    # gather
    dq_full = gather_1d_results(dq_local, process_mesh_dict_4x1, dim=2)
    dk_full = gather_1d_results(dk_local, process_mesh_dict_4x1, dim=2)
    dv_full = gather_1d_results(dv_local, process_mesh_dict_4x1, dim=2)

    assert torch.allclose(out_full, out_ref, atol=1e-3)
    assert torch.allclose(dq_full, dq_ref.to(dq_full.dtype), atol=1e-3)
    assert torch.allclose(dk_full, dk_ref.to(dk_full.dtype), atol=1e-3)
    assert torch.allclose(dv_full, dv_ref.to(dv_full.dtype), atol=1e-3)

    assert torch.allclose(
        dq_local,
        dq_ref[:, :, (local_rank * n_local) : (local_rank + 1) * n_local].to(
            dq_local.dtype
        ),
        atol=1e-3,
    )
    assert torch.allclose(
        dk_local,
        dk_ref[:, :, (local_rank * n_local) : (local_rank + 1) * n_local].to(
            dk_local.dtype
        ),
        atol=1e-3,
    )
    assert torch.allclose(
        dv_local,
        dv_ref[:, :, (local_rank * n_local) : (local_rank + 1) * n_local].to(
            dv_local.dtype
        ),
        atol=1e-3,
    )


def test_ring(device: torch.device, process_mesh_dict_4x1: dict) -> None:
    """Test forward-backward pass of ring attention wrapped by nn.Module."""
    group = process_mesh_dict_4x1["model"]
    local_rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    x_features = 256
    output_dim = embed_dim = 256
    b, heads, n_global, d = 1, 6, 256 * 4, 256

    n_local = n_global // world_size

    torch.manual_seed(2)

    # pass in the QKV mapping alone
    w_qkv = torch.randn(
        3 * d * heads, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)

    # randomize input data per rank
    torch.manual_seed(local_rank)  # different seed for input data generation
    x = torch.randn(
        b, n_local, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    x = x.contiguous()

    qkv = x @ w_qkv.mT
    # QKV shape: [B, Nlocal, 3*d*h]
    # We need QKV to have shape: []
    q_local, k_local, v_local = qkv.reshape(b, n_local, 3, heads, d).permute(
        2, 0, 3, 1, 4
    )

    q_local, k_local, v_local = (
        q_local.contiguous(),
        k_local.contiguous(),
        v_local.contiguous(),
    )

    # Gather full values
    x_ref = gather_1d_results(x, process_mesh_dict_4x1, dim=1)
    q_ref = gather_1d_results(q_local, process_mesh_dict_4x1, dim=2)
    k_ref = gather_1d_results(k_local, process_mesh_dict_4x1, dim=2)
    v_ref = gather_1d_results(v_local, process_mesh_dict_4x1, dim=2)

    model = ring_attention.Ring(
        x_features=x_features,
        embed_dim=embed_dim,
        output_dim=output_dim,
        heads=heads,
        dropout_p=0.0,
        local_rank=local_rank,
        device=device,
        parallelism=group.size(),
        spatial_group=group,
        bayesian=False,
        dtype=torch.bfloat16,
        ring_config=ring_attn_fns.RingAttentionConfig(
            device=device,
            send_in_forward=ring_attn_fns.SendForward.X,
            save_for_backward=ring_attn_fns.SaveBackward.X,
            save_scope=ring_attn_fns.SaveScope.GLOBAL,
            send_backward=ring_attn_fns.SendBackward.GATHER_DKV,
            offload=ring_attn_fns.Offload.DISABLED,
        ),
    )
    model_baseline = ring_attention.RingBaseline(
        x_features=x_features,
        embed_dim=embed_dim,
        output_dim=output_dim,
        heads=heads,
        dropout=0.0,
        local_rank=local_rank,
        device=device,
        parallelism=group.size(),
        spatial_group=group,
        bayesian=False,
        dtype=torch.bfloat16,
    )

    # sped up call
    q_local = q_local.detach().requires_grad_(True)
    k_local = k_local.detach().requires_grad_(True)
    v_local = v_local.detach().requires_grad_(True)

    output = model(x=x.detach(), q=q_local, k=k_local, v=v_local, w_qkv=w_qkv)
    output.sum().backward()

    # baseline
    # unbind
    q_ref = q_ref.detach().requires_grad_(True)
    k_ref = k_ref.detach().requires_grad_(True)
    v_ref = v_ref.detach().requires_grad_(True)

    output_ref = model_baseline(x_ref.detach(), q_ref, k_ref, v_ref, w_qkv)
    output_ref.sum().backward()

    assert isinstance(q_ref.grad, torch.Tensor)
    assert isinstance(k_ref.grad, torch.Tensor)
    assert isinstance(v_ref.grad, torch.Tensor)

    assert torch.allclose(
        output,
        output_ref[:, :, (local_rank * n_local) : (local_rank + 1) * n_local].to(
            output.dtype
        ),
        atol=1e-3,
    )
    assert torch.allclose(
        q_local.grad,
        q_ref.grad[:, :, (local_rank * n_local) : (local_rank + 1) * n_local].to(
            output.dtype
        ),
        atol=1e-3,
    )
    assert torch.allclose(
        k_local.grad,
        k_ref.grad[:, :, (local_rank * n_local) : (local_rank + 1) * n_local].to(
            output.dtype
        ),
        atol=1e-3,
    )
    assert torch.allclose(
        v_local.grad,
        v_ref.grad[:, :, (local_rank * n_local) : (local_rank + 1) * n_local].to(
            output.dtype
        ),
        atol=1e-3,
    )
