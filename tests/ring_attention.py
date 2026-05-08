import pytest
import torch
import torch.distributed as dist

import jigsaw.attention_ring as ring_attention
import jigsaw.ring_attention_helpers as ring_attn_fns
import jigsaw.offload as offload
from tests import conftest


@pytest.fixture()
def process_mesh():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[4, 1], mesh_dim_names=["row", "col"]
    )


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


# def test_backward_sdpa(device, process_mesh):
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

#     Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 3, 1, 4)
#     # QKV has shape [B, N, h, d]
#     Q_local, K_local, V_local = (
#         Q_local.contiguous(),
#         K_local.contiguous(),
#         V_local.contiguous(),
#     )
#     scale = Q_local.shape[-1] ** -0.5

#     # O ref full
#     O_ref_full, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
#         Q_local,
#         K_local,
#         V_local,
#         dropout_p=0.0,
#         is_causal=False,
#         scale=scale,
#         return_debug_mask=False,
#     )
#     dQ_ref, dK_ref, dV_ref = (
#         torch.ops.aten._scaled_dot_product_flash_attention_backward(
#             torch.ones(
#                 O_ref_full.shape, device=device, dtype=O_ref_full.dtype
#             ),  # grad_output: (B, N, H, d)
#             Q_local,  # query:       (B, N, H, d)
#             K_local,  # key:         (B, N, H, d)
#             V_local,  # value:       (B, N, H, d)
#             O_ref_full,  # out:         (B, N, H d)
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
#     flash_Q = Q_local.detach().float().requires_grad_(True)
#     flash_K = K_local.detach().float().requires_grad_(True)
#     flash_V = V_local.detach().float().requires_grad_(True)

#     with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
#         flash_out = torch.nn.functional.scaled_dot_product_attention(
#             flash_Q, flash_K, flash_V, is_causal=False, dropout_p=0, attn_mask=None
#         )
#         # Qg grad
#         flash_Q.retain_grad()
#         flash_K.retain_grad()
#         flash_V.retain_grad()
#         flash_out.sum().backward()

#     Qg = Q_local.detach().float().requires_grad_(True)
#     Kg = K_local.detach().float().requires_grad_(True)
#     Vg = V_local.detach().float().requires_grad_(True)

#     # Autograd ground truth — definitionally correct
#     q, k, v = Qg, Kg, Vg
#     S_au = (q @ k.transpose(-1, -2)) * scale  # (B, H, N, N)
#     LSE_au = torch.logsumexp(S_au, dim=-1)  # (B, H, N)
#     O_au = torch.softmax(S_au, dim=-1) @ v  # (B, N, H, d)
#     q.retain_grad()
#     k.retain_grad()
#     v.retain_grad()
#     O_au.sum().backward()

#     # q has shape [B, h, N, d]
#     # O_ref_full has shape [B, N, h, d]
#     # O_au has shape [B, h, d, N]
#     # dQ_ref: [B, N, h, d]
#     # dQ_au: [B, N, h, d]

#     # check flash attention implementation against local implementation
#     assert torch.allclose(flash_out, O_au, atol=1e-3)
#     assert torch.allclose(flash_Q.grad, q.grad, atol=1e-3)
#     assert torch.allclose(flash_K.grad, k.grad, atol=1e-3)
#     assert torch.allclose(flash_V.grad, v.grad, atol=1e-3)

#     # check flash attention against torch kernel
#     assert torch.allclose(flash_out, O_ref_full.float(), atol=1e-3)
#     assert torch.allclose(LSE_au, logsumexp.float(), atol=1e-3)
#     assert torch.allclose(flash_Q.grad, dQ_ref.float(), atol=1e-2)
#     assert torch.allclose(flash_K.grad, dK_ref.float(), atol=1e-2)
#     assert torch.allclose(flash_V.grad, dV_ref.float(), atol=1e-2)


# def test_backward_flash(device, process_mesh):
#     local_rank = dist.get_rank(group=process_mesh["row"])
#     world_size = dist.get_world_size(group=process_mesh["row"])

#     torch.manual_seed(2)  # same seed for weight initialization
#     B, h, N, d = 1, 3, world_size * 16, 256
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
#     # flash_attention_forward takes [B, N, H, d]
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

#     dQ_ref, dK_ref, dV_ref = torch.ops.aten._flash_attention_backward(
#         torch.ones(
#             O_ref_full.shape, device=device, dtype=O_ref_full.dtype
#         ),  # grad_output: (B, N, H, d)
#         Q_local,  # query:       (B, N, H, d)
#         K_local,  # key:         (B, N, H, d)
#         V_local,  # value:       (B, N, H, d)
#         O_ref_full,  # out:         (B, N, H d)
#         logsumexp,  # logsumexp:   (B*H, N) — from forward, reshape(B*H, N) before passing
#         dropout_p=0.0,
#         is_causal=False,
#         scale=scale,
#         cum_seq_q=None,
#         cum_seq_k=None,
#         max_q=Nlocal,
#         max_k=Nlocal,
#         rng_state=None,
#         unused=None,
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
#     O_au = torch.softmax(S_au, dim=-1) @ v  # (B, H, N, d)
#     q.retain_grad()
#     k.retain_grad()
#     v.retain_grad()
#     O_au.sum().backward()

#     # check flash attention implementation against local implementation
#     assert torch.allclose(flash_out, O_au, atol=1e-3)
#     assert torch.allclose(flash_Q.grad, q.grad, atol=1e-3)
#     assert torch.allclose(flash_K.grad, k.grad, atol=1e-3)
#     assert torch.allclose(flash_V.grad, v.grad, atol=1e-3)

#     # check flash attention against torch kernel
#     assert torch.allclose(flash_out, O_ref_full.transpose(1, 2).float(), atol=1e-3)
#     assert torch.allclose(LSE_au, logsumexp.float(), atol=1e-3)
#     assert torch.allclose(flash_Q.grad, dQ_ref.float().transpose(1, 2), atol=1e-2)
#     assert torch.allclose(flash_K.grad, dK_ref.float().transpose(1, 2), atol=1e-2)
#     assert torch.allclose(flash_V.grad, dV_ref.float().transpose(1, 2), atol=1e-2)


@pytest.mark.parametrize(
    "send_X_or_KV,save_X_or_KV,save_local_or_global,send_dKV_or_not,offloader",
    [
        (s, sv, sl, sd, offloader)
        for s in [0, 1]
        for sv in [0, 1]
        for sl in [0, 1]
        for sd in [0, 1]
        for offloader in [None]
    ],
)
def test_ring_forward_backward(
    device,
    process_mesh,
    send_X_or_KV,
    save_X_or_KV,
    save_local_or_global,
    send_dKV_or_not,
    offloader,
):
    """Verify weights for XWT_Sync are synchronized after a few iterations."""
    if send_X_or_KV == 1 and save_X_or_KV == 0:
        pytest.xfail("not yet supported")

    local_rank = dist.get_rank(group=None)
    world_size = dist.get_world_size(group=None)

    torch.manual_seed(2)  # same seed for weight initialization
    B, h, N, d = 3, 6, world_size * 256, 256
    Nlocal = N // world_size
    W_qkv = torch.randn(
        3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    torch.manual_seed(local_rank)  # different seed for input data generation
    X = torch.randn(
        B, Nlocal, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    X = X.contiguous()
    empty = torch.empty(0, device=X.device, dtype=X.dtype)
    zeros = torch.zeros_like(X, device=X.device, dtype=X.dtype)

    QKV = X @ W_qkv.mT
    # QKV shape: [B, Nlocal, 3*d*h]
    # We need QKV to have shape: []
    Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 3, 1, 4)

    Q_local, K_local, V_local = (
        Q_local.contiguous(),
        K_local.contiguous(),
        V_local.contiguous(),
    )
    scale = Q_local.shape[-1] ** -0.5

    # just test helper function forward
    O, l, chunks_to_save = ring_attn_fns.ring_forward_sendX(
        Q_local,
        X,
        W_qkv,
        scale,
        group=process_mesh["row"],
        empty=empty,
        rank=local_rank,
        world_size=world_size,
        send_X_or_KV=send_X_or_KV,
        save_X_or_KV=save_X_or_KV,
        save_local_or_global=save_local_or_global,
        offloader=offloader,
    )

    O_full = conftest.gather_1d_results(O, process_mesh, dim=2)
    X_full = conftest.gather_1d_results(X, process_mesh, dim=1)
    QKV = X_full @ W_qkv.mT

    Q_ref1, K_ref1, V_ref1 = QKV.reshape(B, N, 3, h, d).permute(2, 0, 3, 1, 4)

    Q_ref = conftest.gather_1d_results(Q_local, process_mesh, dim=2)
    K_ref = conftest.gather_1d_results(K_local, process_mesh, dim=2)
    V_ref = conftest.gather_1d_results(V_local, process_mesh, dim=2)

    assert torch.allclose(Q_ref1, Q_ref, atol=1e-3)
    assert torch.allclose(K_ref1, K_ref, atol=1e-3)
    assert torch.allclose(V_ref1, V_ref, atol=1e-3)
    # Reference calculation
    O_ref_full, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        Q_ref,
        K_ref,
        V_ref,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        return_debug_mask=False,
    )

    dO_local = torch.ones_like(O, device=device, dtype=O.dtype)

    dQ_local, dK_local, dV_local = ring_attn_fns.ring_backward(
        dO_local,
        Q_local,
        O,
        X,
        l,
        W_qkv,
        scale,
        None,
        local_rank,
        world_size,
        chunks_to_save,
        send_X_or_KV=send_X_or_KV,
        save_X_or_KV=save_X_or_KV,
        save_local_or_global=save_local_or_global,
        send_dKV_or_not=send_dKV_or_not,
        offloader=offloader,
        empty=empty,
    )

    # Backward pass
    dQ_ref, dK_ref, dV_ref = (
        torch.ops.aten._scaled_dot_product_flash_attention_backward(
            torch.ones(
                O_ref_full.shape, device=device, dtype=O_ref_full.dtype
            ),  # grad_output: (B, N, H, d)
            Q_ref,  # query:       (B, N, H, d)
            K_ref,  # key:         (B, N, H, d)
            V_ref,  # value:       (B, N, H, d)
            O_ref_full,  # out:         (B, H, N, d)
            logsumexp,  # logsumexp:   (B*H, N) — from forward, reshape(B*H, N) before passing
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            cum_seq_q=None,
            cum_seq_k=None,
            max_q=N,
            max_k=N,
            philox_seed=torch.zeros(1, dtype=torch.long, device=Q_ref.device),
            philox_offset=torch.zeros(1, dtype=torch.long, device=Q_ref.device),
        )
    )

    # gather
    # print(f'[R{local_rank}/{world_size}] dK local {dK_local.shape} dV {dV_local.shape}')
    dQ_full = conftest.gather_1d_results(dQ_local, process_mesh, dim=2)
    dK_full = conftest.gather_1d_results(dK_local, process_mesh, dim=2)
    dV_full = conftest.gather_1d_results(dV_local, process_mesh, dim=2)

    # print(f'Rank {local_rank} is gathered dQ full {dQ_full.sum()}')
    assert torch.allclose(O_full, O_ref_full, atol=1e-3)
    assert torch.allclose(dQ_full, dQ_ref.to(dQ_full.dtype), atol=1e-3)
    assert torch.allclose(dK_full, dK_ref.to(dK_full.dtype), atol=1e-3)
    assert torch.allclose(dV_full, dV_ref.to(dV_full.dtype), atol=1e-3)

    assert torch.allclose(
        dQ_local,
        dQ_ref[:, :, (local_rank * Nlocal) : (local_rank + 1) * Nlocal].to(
            dQ_local.dtype
        ),
        atol=1e-3,
    )
    assert torch.allclose(
        dK_local,
        dK_ref[:, :, (local_rank * Nlocal) : (local_rank + 1) * Nlocal].to(
            dK_local.dtype
        ),
        atol=1e-3,
    )
    assert torch.allclose(
        dV_local,
        dV_ref[:, :, (local_rank * Nlocal) : (local_rank + 1) * Nlocal].to(
            dV_local.dtype
        ),
        atol=1e-3,
    )


@pytest.fixture
def offloader(device):
    # return None
    return offload.NVLinkActivationOffloader(device)


def test_offload(device, process_mesh, offloader):
    """Verify weights for XWT_Sync are synchronized after a few iterations."""
    send_X_or_KV = 0
    save_X_or_KV = 0
    save_local_or_global = 0
    send_dKV_or_not = 0

    local_rank = dist.get_rank(group=None)
    world_size = dist.get_world_size(group=None)

    torch.manual_seed(2)  # same seed for weight initialization
    B, h, N, d = 1, 1, world_size * 256, 256
    Nlocal = N // world_size
    W_qkv = torch.randn(
        3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    torch.manual_seed(local_rank)  # different seed for input data generation
    X = torch.randn(
        B, Nlocal, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    X = X.contiguous()
    empty = torch.empty(0, device=X.device, dtype=X.dtype)
    zeros = torch.zeros_like(X, device=X.device, dtype=X.dtype)

    QKV = X @ W_qkv.mT
    # QKV shape: [B, Nlocal, 3*d*h]
    # We need QKV to have shape: []
    Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 3, 1, 4)

    Q_local, K_local, V_local = (
        Q_local.contiguous(),
        K_local.contiguous(),
        V_local.contiguous(),
    )
    scale = Q_local.shape[-1] ** -0.5

    # just test helper function forward
    O, l, chunks_to_save = ring_attn_fns.ring_forward_sendX(
        Q_local,
        X,
        W_qkv,
        scale,
        group=process_mesh["row"],
        empty=empty,
        rank=local_rank,
        world_size=world_size,
        send_X_or_KV=send_X_or_KV,
        save_X_or_KV=save_X_or_KV,
        save_local_or_global=save_local_or_global,
        offloader=offloader,
    )

    dO_local = torch.ones_like(O, device=device, dtype=O.dtype)

    dQ_local, dK_local, dV_local = ring_attn_fns.ring_backward(
        dO_local,
        Q_local,
        O,
        X,
        l,
        W_qkv,
        scale,
        None,
        local_rank,
        world_size,
        chunks_to_save,
        send_X_or_KV=send_X_or_KV,
        save_X_or_KV=save_X_or_KV,
        save_local_or_global=save_local_or_global,
        send_dKV_or_not=send_dKV_or_not,
        offloader=offloader,
        empty=empty,
    )

    assert True


def test_ring(device, process_mesh):
    group = process_mesh["model"]
    local_rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    x_features = 256
    output_dim = embed_dim = 256
    B, heads, N_global, d = 1, 6, 256*4, 256
    
    N_local = N_global // world_size

    torch.manual_seed(2)

    # pass in the QKV mapping alone
    W_qkv = torch.randn(
        3 * d * heads, d, device=device, requires_grad=True, dtype=torch.bfloat16
    )/ (d**0.5)

    # randomize input data per rank
    torch.manual_seed(local_rank)  # different seed for input data generation
    X = torch.randn(
        B, N_local, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    X = X.contiguous()

    QKV = X @ W_qkv.mT
    # QKV shape: [B, Nlocal, 3*d*h]
    # We need QKV to have shape: []
    Q_local, K_local, V_local = QKV.reshape(B, N_local, 3, heads, d).permute(
        2, 0, 3, 1, 4
    )

    Q_local, K_local, V_local = (
        Q_local.contiguous(),
        K_local.contiguous(),
        V_local.contiguous(),
    )

    # Gather full values
    X_full = conftest.gather_1d_results(X, process_mesh, dim=1)
    Q_ref = conftest.gather_1d_results(Q_local, process_mesh, dim=2)
    K_ref = conftest.gather_1d_results(K_local, process_mesh, dim=2)
    V_ref = conftest.gather_1d_results(V_local, process_mesh, dim=2)

    model = ring_attention.Ring(
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
    
    Q_local = Q_local.detach().requires_grad_(True)
    K_local = K_local.detach().requires_grad_(True)
    V_local = V_local.detach().requires_grad_(True)
    
    output = model(X.detach(), Q_local, K_local, V_local, W_qkv, None)
    output.sum().backward()

    # baseline
    # unbind
    Q_ref = Q_ref.detach().requires_grad_(True)
    K_ref = K_ref.detach().requires_grad_(True)
    V_ref = V_ref.detach().requires_grad_(True)
    
    output_global = model_baseline(X_full.detach(), Q_ref, K_ref, V_ref, W_qkv, None)
    output_global.sum().backward()

    assert torch.allclose(
        output,
        output_global[:, :, (local_rank * N_local) : (local_rank + 1) * N_local].to(
            output.dtype
        ),
        atol=1e-3,
    )
    assert torch.allclose(
        Q_local.grad,
        Q_ref.grad[:, :, (local_rank * N_local) : (local_rank + 1) * N_local].to(
            output.dtype
        ),
        atol=1e-3,
    )
    assert torch.allclose(
        K_local.grad,
        K_ref.grad[:, :, (local_rank * N_local) : (local_rank + 1) * N_local].to(
            output.dtype
        ),
        atol=1e-3,
    )
    assert torch.allclose(
        V_local.grad,
        V_ref.grad[:, :, (local_rank * N_local) : (local_rank + 1) * N_local].to(
            output.dtype
        ),
        atol=1e-3,
    )