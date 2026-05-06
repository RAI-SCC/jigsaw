import pytest
import torch
import torch.distributed as dist

import jigsaw.attention_ring as ring_attention
import jigsaw.ring_attention_helpers as ring_attn_fns
from tests import conftest


@pytest.fixture()
def process_mesh():
    """Return process mesh."""
    return conftest.make_process_mesh(
        device_type="nccl", mesh_shape=[4, 1], mesh_dim_names=["row", "col"]
    )


# def test_ring_forward_sendKV(device, process_mesh):
#     """Verify weights for XWT_Sync are synchronized after a few iterations."""
#     local_rank = dist.get_rank(group=None)
#     world_size = dist.get_world_size(group=None)

#     torch.manual_seed(2)  # same seed for weight initialization
#     B, h, N, d = 1, 6, world_size * 1024, 256
#     Nlocal = N // world_size
#     W_qkv = torch.randn(
#         3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     torch.manual_seed(local_rank)  # different seed for input data generation
#     X = torch.randn(
#         B, Nlocal, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     X = X.contiguous()
#     empty = torch.empty(0, device=X.device, dtype=X.dtype)
#     zeros = torch.zeros_like(X, device=X.device, dtype=X.dtype)

#     QKV = X @ W_qkv.mT
#     # QKV shape: [B, Nlocal, 3*d*h]
#     # We need QKV to have shape: []
#     Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 1, 3, 4)

#     Q_local, K_local, V_local = (
#         Q_local.contiguous(),
#         K_local.contiguous(),
#         V_local.contiguous(),
#     )
#     scale = Q_local.shape[-1] ** -0.5

#     # just test helper function forward
#     O, *_ = ring_attn_fns.ring_forward_sendX(
#         Q_local,
#         X,
#         W_qkv,
#         scale,
#         group=process_mesh["row"],
#         empty=empty,
#         zeros=zeros,
#         rank=local_rank,
#         world_size=world_size,
#         send_X_or_KV=1,
#         save_X_or_KV=1,
#         save_local_or_global=1,
#         offload=False,
#     )

#     O_full = conftest.gather_1d_results(O, process_mesh, dim=1)
#     X_full = conftest.gather_1d_results(X, process_mesh, dim=1)
#     QKV = X_full @ W_qkv.mT

#     Q_ref1, K_ref1, V_ref1 = QKV.reshape(B, N, 3, h, d).permute(2, 0, 1, 3, 4)

#     Q_ref = conftest.gather_1d_results(Q_local, process_mesh, dim=1)
#     K_ref = conftest.gather_1d_results(K_local, process_mesh, dim=1)
#     V_ref = conftest.gather_1d_results(V_local, process_mesh, dim=1)

#     assert torch.allclose(Q_ref1, Q_ref, atol=1e-4)
#     assert torch.allclose(K_ref1, K_ref, atol=1e-4)
#     assert torch.allclose(V_ref1, V_ref, atol=1e-4)
#     # Reference calculation
#     O_ref_full, *_ = torch.ops.aten._flash_attention_forward(
#         Q_ref,
#         K_ref,
#         V_ref,
#         cum_seq_q=None,
#         cum_seq_k=None,
#         max_q=N,
#         max_k=N,
#         dropout_p=0.0,
#         is_causal=False,
#         scale=scale,
#         return_debug_mask=False,
#     )

#     assert torch.allclose(O_full, O_ref_full, atol=1e-4)


# def test_ring_forward_sendX(device, process_mesh):
#     """Verify weights for XWT_Sync are synchronized after a few iterations."""
#     local_rank = dist.get_rank(group=None)
#     world_size = dist.get_world_size(group=None)

#     torch.manual_seed(2)  # same seed for weight initialization
#     B, h, N, d = 1, 6, world_size * 2048, 256
#     Nlocal = N // world_size
#     W_qkv = torch.randn(
#         3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     torch.manual_seed(local_rank)  # different seed for input data generation
#     X = torch.randn(
#         B, Nlocal, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     X = X.contiguous()
#     empty = torch.empty(0, device=X.device, dtype=X.dtype)
#     zeros = torch.zeros_like(X, device=X.device, dtype=X.dtype)

#     QKV = X @ W_qkv.mT
#     # QKV shape: [B, Nlocal, 3*d*h]
#     # We need QKV to have shape: []
#     Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 1, 3, 4)

#     Q_local, K_local, V_local = (
#         Q_local.contiguous(),
#         K_local.contiguous(),
#         V_local.contiguous(),
#     )
#     scale = Q_local.shape[-1] ** -0.5

#     # just test helper function forward
#     O, *_ = ring_attn_fns.ring_forward_sendX(
#         Q_local,
#         X,
#         W_qkv,
#         scale,
#         group=process_mesh["row"],
#         empty=empty,
#         zeros=zeros,
#         rank=local_rank,
#         world_size=world_size,
#         send_X_or_KV=0,
#         save_X_or_KV=1,
#         save_local_or_global=1,
#         offload=False,
#     )

#     O_full = conftest.gather_1d_results(O, process_mesh, dim=1)
#     X_full = conftest.gather_1d_results(X, process_mesh, dim=1)
#     QKV = X_full @ W_qkv.mT

#     Q_ref1, K_ref1, V_ref1 = QKV.reshape(B, N, 3, h, d).permute(2, 0, 1, 3, 4)

#     Q_ref = conftest.gather_1d_results(Q_local, process_mesh, dim=1)
#     K_ref = conftest.gather_1d_results(K_local, process_mesh, dim=1)
#     V_ref = conftest.gather_1d_results(V_local, process_mesh, dim=1)

#     assert torch.allclose(Q_ref1, Q_ref, atol=1e-4)
#     assert torch.allclose(K_ref1, K_ref, atol=1e-4)
#     assert torch.allclose(V_ref1, V_ref, atol=1e-4)
#     # Reference calculation
#     O_ref_full, *_ = torch.ops.aten._flash_attention_forward(
#         Q_ref,
#         K_ref,
#         V_ref,
#         cum_seq_q=None,
#         cum_seq_k=None,
#         max_q=N,
#         max_k=N,
#         dropout_p=0.0,
#         is_causal=False,
#         scale=scale,
#         return_debug_mask=False,
#     )

#     assert torch.allclose(O_full, O_ref_full, atol=1e-4)


# def test_ring_forward_sendX_saveX(device, process_mesh):
#     """Verify weights for XWT_Sync are synchronized after a few iterations."""
#     local_rank = dist.get_rank(group=None)
#     world_size = dist.get_world_size(group=None)

#     torch.manual_seed(2)  # same seed for weight initialization
#     B, h, N, d = 1, 6, world_size * 2048, 256
#     Nlocal = N // world_size
#     W_qkv = torch.randn(
#         3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     torch.manual_seed(local_rank)  # different seed for input data generation
#     X = torch.randn(
#         B, Nlocal, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     X = X.contiguous()
#     empty = torch.empty(0, device=X.device, dtype=X.dtype)
#     zeros = torch.zeros_like(X, device=X.device, dtype=X.dtype)

#     QKV = X @ W_qkv.mT
#     # QKV shape: [B, Nlocal, 3*d*h]
#     # We need QKV to have shape: []
#     Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 1, 3, 4)

#     Q_local, K_local, V_local = (
#         Q_local.contiguous(),
#         K_local.contiguous(),
#         V_local.contiguous(),
#     )
#     scale = Q_local.shape[-1] ** -0.5

#     # just test helper function forward
#     O, *_ = ring_attn_fns.ring_forward_sendX(
#         Q_local,
#         X,
#         W_qkv,
#         scale,
#         group=process_mesh["row"],
#         empty=empty,
#         zeros=zeros,
#         rank=local_rank,
#         world_size=world_size,
#         send_X_or_KV=0,
#         save_X_or_KV=0,
#         save_local_or_global=1,
#         offload=False,
#     )

#     O_full = conftest.gather_1d_results(O, process_mesh, dim=1)
#     X_full = conftest.gather_1d_results(X, process_mesh, dim=1)
#     QKV = X_full @ W_qkv.mT

#     Q_ref1, K_ref1, V_ref1 = QKV.reshape(B, N, 3, h, d).permute(2, 0, 1, 3, 4)

#     Q_ref = conftest.gather_1d_results(Q_local, process_mesh, dim=1)
#     K_ref = conftest.gather_1d_results(K_local, process_mesh, dim=1)
#     V_ref = conftest.gather_1d_results(V_local, process_mesh, dim=1)

#     assert torch.allclose(Q_ref1, Q_ref, atol=1e-4)
#     assert torch.allclose(K_ref1, K_ref, atol=1e-4)
#     assert torch.allclose(V_ref1, V_ref, atol=1e-4)
#     # Reference calculation
#     O_ref_full, *_ = torch.ops.aten._flash_attention_forward(
#         Q_ref,
#         K_ref,
#         V_ref,
#         cum_seq_q=None,
#         cum_seq_k=None,
#         max_q=N,
#         max_k=N,
#         dropout_p=0.0,
#         is_causal=False,
#         scale=scale,
#         return_debug_mask=False,
#     )

#     assert torch.allclose(O_full, O_ref_full, atol=1e-4)


# def test_ring_forward_backward(device, process_mesh):
#     """Verify weights for XWT_Sync are synchronized after a few iterations."""
#     local_rank = dist.get_rank(group=None)
#     world_size = dist.get_world_size(group=None)

#     torch.manual_seed(2)  # same seed for weight initialization
#     B, h, N, d = 1, 6, world_size * 2048, 256
#     Nlocal = N // world_size
#     W_qkv = torch.randn(
#         3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     torch.manual_seed(local_rank)  # different seed for input data generation
#     X = torch.randn(
#         B, Nlocal, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     X = X.contiguous()
#     empty = torch.empty(0, device=X.device, dtype=X.dtype)
#     zeros = torch.zeros_like(X, device=X.device, dtype=X.dtype)

#     QKV = X @ W_qkv.mT
#     # QKV shape: [B, Nlocal, 3*d*h]
#     # We need QKV to have shape: []
#     Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 1, 3, 4)

#     Q_local, K_local, V_local = (
#         Q_local.contiguous(),
#         K_local.contiguous(),
#         V_local.contiguous(),
#     )
#     scale = Q_local.shape[-1] ** -0.5

#     # just test helper function forward
#     O, l, chunks_to_save = ring_attn_fns.ring_forward_sendX(
#         Q_local,
#         X,
#         W_qkv,
#         scale,
#         group=process_mesh["row"],
#         empty=empty,
#         zeros=zeros,
#         rank=local_rank,
#         world_size=world_size,
#         send_X_or_KV=0,
#         save_X_or_KV=0,
#         save_local_or_global=1,
#         offload=False,
#     )

#     O_full = conftest.gather_1d_results(O, process_mesh, dim=1)
#     X_full = conftest.gather_1d_results(X, process_mesh, dim=1)
#     QKV = X_full @ W_qkv.mT

#     Q_ref1, K_ref1, V_ref1 = QKV.reshape(B, N, 3, h, d).permute(2, 0, 1, 3, 4)

#     Q_ref = conftest.gather_1d_results(Q_local, process_mesh, dim=1)
#     K_ref = conftest.gather_1d_results(K_local, process_mesh, dim=1)
#     V_ref = conftest.gather_1d_results(V_local, process_mesh, dim=1)

#     assert torch.allclose(Q_ref1, Q_ref, atol=1e-4)
#     assert torch.allclose(K_ref1, K_ref, atol=1e-4)
#     assert torch.allclose(V_ref1, V_ref, atol=1e-4)
#     # Reference calculation
#     O_ref_full, logsumexp, *_ = torch.ops.aten._flash_attention_forward(
#         Q_ref,
#         K_ref,
#         V_ref,
#         cum_seq_q=None,
#         cum_seq_k=None,
#         max_q=N,
#         max_k=N,
#         dropout_p=0.0,
#         is_causal=False,
#         scale=scale,
#         return_debug_mask=False,
#     )

#     dO_local = torch.ones_like(O, device=device, dtype=O.dtype)
#     dQ_local, dK_local, dV_local = ring_attn_fns.ring_backward(
#         dO_local,
#         Q_local,
#         O,
#         X,
#         l,
#         W_qkv,
#         scale,
#         None,
#         world_size,
#         chunks_to_save,
#         send_X_or_KV=0,
#         save_X_or_KV=0,
#         save_local_or_global=1,
#         offload=False,
#         empty=empty
#     )

#     # Backward pass
#     dQ_ref, dK_ref, dV_ref = torch.ops.aten._scaled_dot_product_flash_attention_backward(
#         torch.ones(
#             O_ref_full.shape, device=device, dtype=O_ref_full.dtype
#         ),  # grad_output: (B, N, H, d)
#         Q_ref,  # query:       (B, N, H, d)
#         K_ref,  # key:         (B, N, H, d)
#         V_ref,  # value:       (B, N, H, d)
#         O_ref_full,  # out:         (B, H, N, d)
#         logsumexp,  # logsumexp:   (B*H, N) — from forward, reshape(B*H, N) before passing
#         dropout_p=0.0,
#         is_causal=False,
#         scale=scale,
#         cum_seq_q=None,
#         cum_seq_k=None,
#         max_q=Nlocal,
#         max_k=Nlocal,
#         philox_seed=torch.zeros(1, dtype=torch.long, device=Q_ref.device),
#         philox_offset=torch.zeros(1, dtype=torch.long, device=Q_ref.device),
#     )

#     # gather
#     dQ_full = conftest.gather_1d_results(dQ_local, process_mesh, dim=1)
#     dK_full = conftest.gather_1d_results(dK_local, process_mesh, dim=1)
#     dV_full = conftest.gather_1d_results(dV_local, process_mesh, dim=1)

#     assert torch.allclose(O_full, O_ref_full, atol=1e-4)
#     assert torch.allclose(dQ_full, dQ_ref, atol=1e-4)
#     assert torch.allclose(dK_full, dK_ref, atol=1e-4)
#     assert torch.allclose(dV_full, dV_ref, atol=1e-4)


def attn_backward_partial(dO, Q, K, V, O, LSE, scale):
    """
    dO, Q, O: (B, M, H, d)         — local
    K, V:     (B, N, H, d)         — one KV chunk
    LSE:      (B, H, M)            — global LSE for local Q (from ring fwd)
    Returns dQ_partial (B, M, H, d), dK (B, N, H, d), dV (B, N, H, d) in fp32.
    Cast back to bf16 outside if you want.
    """
    # move to (B, H, *, d) for batched matmul; promote to fp32 for stability
    q = Q.transpose(1, 2).float()
    k = K.transpose(1, 2).float()
    v = V.transpose(1, 2).float()
    o = O.transpose(1, 2).float()
    do = dO.transpose(1, 2).float()
    lse = LSE.float().unsqueeze(-1)  # (B, H, M, 1)

    S = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, H, M, N)
    P = torch.exp(S - lse)  # global-normalized softmax
    D = (o * do).sum(dim=-1, keepdim=True)  # (B, H, M, 1) — GLOBAL D
    dP = torch.matmul(do, v.transpose(-1, -2))  # (B, H, M, N)
    dS = P * (dP - D)  # (B, H, M, N)

    dV = torch.matmul(P.transpose(-1, -2), do)  # (B, H, N, d)
    dQ = torch.matmul(dS, k) * scale  # (B, H, M, d)
    dK = torch.matmul(dS.transpose(-1, -2), q) * scale  # (B, H, N, d)

    return dQ.transpose(1, 2), dK.transpose(1, 2), dV.transpose(1, 2)


# def test_ring_forward_backward_saveKV(device, process_mesh):
#     local_rank = dist.get_rank(group=process_mesh["row"])
#     world_size = dist.get_world_size(group=process_mesh["row"])

#     torch.manual_seed(2)  # same seed for weight initialization
#     B, h, N, d = 1, 1, world_size * 1, 8
#     Nlocal = N // world_size
#     W_qkv = torch.randn(
#         3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     torch.manual_seed(local_rank)  # different seed for input data generation
#     X = torch.randn(
#         B, Nlocal, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     X.fill_(local_rank / (10))
#     X = X.contiguous()

#     empty = torch.empty(0, device=X.device, dtype=X.dtype)
#     zeros = torch.zeros_like(X, device=X.device, dtype=X.dtype)

#     QKV = X @ W_qkv.mT
#     # QKV shape: [B, Nlocal, 3*d*h]
#     # We need QKV to have shape: []
#     Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 1, 3, 4)

#     Q_local, K_local, V_local = (
#         Q_local.contiguous(),
#         K_local.contiguous(),
#         V_local.contiguous(),
#     )
#     scale = Q_local.shape[-1] ** -0.5

#     # just test helper function forward
#     O, l, chunks_to_save = ring_attn_fns.ring_forward_sendX(
#         Q_local,
#         X,
#         W_qkv,
#         scale,
#         group=process_mesh["row"],
#         empty=empty,
#         zeros=zeros,
#         rank=local_rank,
#         world_size=world_size,
#         send_X_or_KV=0,
#         save_X_or_KV=1,
#         save_local_or_global=1,
#         offload=False,
#     )

#     O_full = conftest.gather_1d_results(O, process_mesh, dim=1)
#     X_full = conftest.gather_1d_results(X, process_mesh, dim=1)
#     QKV = X_full @ W_qkv.mT

#     Q_ref1, K_ref1, V_ref1 = QKV.reshape(B, N, 3, h, d).permute(2, 0, 1, 3, 4)

#     Q_ref = conftest.gather_1d_results(Q_local, process_mesh, dim=1)
#     K_ref = conftest.gather_1d_results(K_local, process_mesh, dim=1)
#     V_ref = conftest.gather_1d_results(V_local, process_mesh, dim=1)

#     assert torch.allclose(Q_ref1, Q_ref, atol=1e-6)
#     assert torch.allclose(K_ref1, K_ref, atol=1e-6)
#     assert torch.allclose(V_ref1, V_ref, atol=1e-6)
#     # Reference calculation
#     O_ref_full, logsumexp, *_ = torch.ops.aten._flash_attention_forward(
#         Q_ref,
#         K_ref,
#         V_ref,
#         cum_seq_q=None,
#         cum_seq_k=None,
#         max_q=N,
#         max_k=N,
#         dropout_p=0.0,
#         is_causal=False,
#         scale=scale,
#         return_debug_mask=False,
#     )

#     # assert l same
#     print(f"l shape {l.shape}, logsumexp shape {logsumexp.shape}")
#     l_gathered = conftest.gather_1d_results(l.mT, process_mesh, dim=-1)

#     l_gathered = l_gathered.reshape(*logsumexp.shape)

#     assert torch.allclose(logsumexp, l_gathered, atol=1e-4)

#     dO_local = torch.ones_like(O, device=device, dtype=O.dtype)

#     dQ_local, dK_local, dV_local = ring_attn_fns.ring_backward(
#         dO_local,
#         Q_local,
#         O,
#         X,
#         l,
#         W_qkv,
#         scale,
#         None,
#         world_size,
#         chunks_to_save,
#         send_X_or_KV=0,
#         save_X_or_KV=1,
#         save_local_or_global=1,
#         offload=False,
#         empty=empty,
#     )

#     # Backward pass
#     dO_full = torch.ones(O_ref_full.shape, device=device, dtype=O_ref_full.dtype)
#     dQ_ref, dK_ref, dV_ref = (
#         torch.ops.aten._scaled_dot_product_flash_attention_backward(
#             torch.ones(
#                 O_ref_full.shape, device=device, dtype=O_ref_full.dtype
#             ),  # grad_output: (B, N, H, d)
#             Q_ref,  # query:       (B, N, H, d)
#             K_ref,  # key:         (B, N, H, d)
#             V_ref,  # value:       (B, N, H, d)
#             O_ref_full,  # out:         (B, H, N, d)
#             logsumexp.mT,  # logsumexp:   (B*H, N) — from forward, reshape(B*H, N) before passing
#             dropout_p=0.0,
#             is_causal=False,
#             scale=scale,
#             cum_seq_q=None,
#             cum_seq_k=None,
#             max_q=N,
#             max_k=N,
#             philox_seed=torch.zeros(1, dtype=torch.long, device=Q_ref.device),
#             philox_offset=torch.zeros(1, dtype=torch.long, device=Q_ref.device),
#         )
#     )
#     # if local_rank == 0:
#     #     print(f'dKref {dK_ref}')
#     # gather
#     dQ_full = conftest.gather_1d_results(dQ_local, process_mesh, dim=1)
#     dK_full = conftest.gather_1d_results(dK_local, process_mesh, dim=1)
#     dV_full = conftest.gather_1d_results(dV_local, process_mesh, dim=1)

#     total_dQ = torch.zeros_like(Q_local)


#     for c in range(world_size):
#         K_c = K_ref[:, c * Nlocal : (c + 1) * Nlocal].contiguous()
#         V_c = V_ref[:, c * Nlocal : (c + 1) * Nlocal].contiguous()
#         dQ_c, _, _ = torch.ops.aten._scaled_dot_product_flash_attention_backward(
#             dO_local,
#             Q_local,
#             K_c,
#             V_c,
#             O,  # full local O from ring forward
#             l.contiguous(),  # full local LSE from ring forward
#             dropout_p=0.0,
#             is_causal=False,
#             scale=scale,
#             cum_seq_q=None,
#             cum_seq_k=None,
#             max_q=Nlocal,
#             max_k=Nlocal,
#             philox_seed=torch.zeros(1, dtype=torch.long, device=device),
#             philox_offset=torch.zeros(1, dtype=torch.long, device=device),
#         )
#         total_dQ += dQ_c

#     # assert torch.allclose(
#     #     total_dQ,
#     #     dQ_ref[:, local_rank * Nlocal : (local_rank + 1) * Nlocal],
#     #     atol=1e-3,
#     # )
#     dQ_m, dK_m, dV_m = attn_backward_partial(
#         torch.ones_like(O_ref_full), Q_ref, K_ref, V_ref,
#         O_ref_full, logsumexp, scale,
#     )
#     print("math ok:", torch.allclose(dQ_m, dQ_ref.to(torch.float32), atol=1e-2),
#                     torch.allclose(dK_m, dK_ref.to(torch.float32), atol=1e-2),
#                     torch.allclose(dV_m, dV_ref.to(torch.float32), atol=1e-2))
# print(f'abs O full {abs(O_full-O_ref_full).sum()}')
# print(f'abs dQ full {abs(dQ_full-dQ_ref).sum()}')
# print(f'abs O full {O_full - O_ref_full}')
# print(f'dQ full {dQ_full - dQ_ref}')

# if local_rank == 0:
#     print(f"R{local_rank}: Q_ref is {Q_ref}")
# print(f"R{local_rank}/{world_size}: dQ is {dQ_local}")

# # assert torch.allclose(O_full, O_ref_full, atol=1e-3)
# assert torch.allclose(
#     dQ_local, dQ_ref[:, local_rank * Nlocal : (local_rank + 1) * Nlocal], atol=1e-3
# )
# assert torch.allclose(dK_full, dK_ref, atol=1e-3)
# assert torch.allclose(dV_full, dV_ref, atol=1e-3)

# def test_backward(device, process_mesh):
#     local_rank = dist.get_rank(group=process_mesh["row"])
#     world_size = dist.get_world_size(group=process_mesh["row"])

#     torch.manual_seed(2)  # same seed for weight initialization
#     B, h, N, d = 1, 1, world_size * 1, 8
#     Nlocal = N // world_size
#     W_qkv = torch.randn(
#         3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     torch.manual_seed(local_rank)  # different seed for input data generation
#     X = torch.randn(
#         B, Nlocal, d, device=device, requires_grad=True, dtype=torch.bfloat16
#     ) / (d**0.5)
#     X.fill_(local_rank / (10))
#     X = X.contiguous()

#     empty = torch.empty(0, device=X.device, dtype=X.dtype)
#     zeros = torch.zeros_like(X, device=X.device, dtype=X.dtype)

#     QKV = X @ W_qkv.mT
#     # QKV shape: [B, Nlocal, 3*d*h]
#     # We need QKV to have shape: []
#     Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 1, 3, 4)

#     Q_local, K_local, V_local = (
#         Q_local.contiguous(),
#         K_local.contiguous(),
#         V_local.contiguous(),
#     )
#     scale = Q_local.shape[-1] ** -0.5


#     Q_ref = conftest.gather_1d_results(Q_local, process_mesh, dim=1)
#     K_ref = conftest.gather_1d_results(K_local, process_mesh, dim=1)
#     V_ref = conftest.gather_1d_results(V_local, process_mesh, dim=1)

#     # O ref full
#     O_ref_full, logsumexp, *_ = torch.ops.aten._flash_attention_forward(
#         Q_ref,
#         K_ref,
#         V_ref,
#         cum_seq_q=None,
#         cum_seq_k=None,
#         max_q=N,
#         max_k=N,
#         dropout_p=0.0,
#         is_causal=False,
#         scale=scale,
#         return_debug_mask=False
#     )
#     dQ_ref, dK_ref, dV_ref = (
#         torch.ops.aten._scaled_dot_product_flash_attention_backward(
#             torch.ones(
#                 O_ref_full.shape, device=device, dtype=O_ref_full.dtype
#             ),  # grad_output: (B, N, H, d)
#             Q_ref,  # query:       (B, N, H, d)
#             K_ref,  # key:         (B, N, H, d)
#             V_ref,  # value:       (B, N, H, d)
#             O_ref_full,  # out:         (B, H, N, d)
#             logsumexp,  # logsumexp:   (B*H, N) — from forward, reshape(B*H, N) before passing
#             dropout_p=0.0,
#             is_causal=False,
#             scale=scale,
#             cum_seq_q=None,
#             cum_seq_k=None,
#             max_q=N,
#             max_k=N,
#             philox_seed=torch.zeros(1, dtype=torch.long, device=Q_ref.device),
#             philox_offset=torch.zeros(1, dtype=torch.long, device=Q_ref.device),
#         )
#     )

#     # Autograd ground truth — definitionally correct
#     Qg = Q_ref.detach().float().requires_grad_(True)
#     Kg = K_ref.detach().float().requires_grad_(True)
#     Vg = V_ref.detach().float().requires_grad_(True)
#     q, k, v = Qg.transpose(1,2), Kg.transpose(1,2), Vg.transpose(1,2)
#     S_au = (q @ k.transpose(-1,-2)) * scale            # (B, H, N, N)
#     LSE_au = torch.logsumexp(S_au, dim=-1)             # (B, H, N)
#     O_au = (torch.softmax(S_au, dim=-1) @ v).mT  # (B, N, H, d)
#     O_au.sum().backward()
#     dQ_au, dK_au, dV_au = Qg.grad, Kg.grad, Vg.grad

#     # Sanity 1: does the flash forward agree with manual on O and LSE?
#     print("flash O ≈ manual O :", torch.allclose(O_au, O_ref_full.float(), atol=1e-4))
#     print("flash LSE ≈ manual:", torch.allclose(LSE_au, logsumexp.float(), atol=1e-4))

#     # Sanity 2: does flash backward agree with autograd?
#     print("flash dQ ≈ autograd:", torch.allclose(dQ_au, dQ_ref.float(), atol=1e-2))


def test_backward(device, process_mesh):
    local_rank = dist.get_rank(group=process_mesh["row"])
    world_size = dist.get_world_size(group=process_mesh["row"])

    torch.manual_seed(2)  # same seed for weight initialization
    B, h, N, d = 1, 3, world_size * 16, 8
    Nlocal = N // world_size
    W_qkv = torch.randn(
        3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    X = torch.randn(
        B, Nlocal, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    X = X.contiguous()

    QKV = X @ W_qkv.mT

    Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 1, 3, 4)
    # QKV has shape [B, N, h, d]
    Q_local, K_local, V_local = (
        Q_local.contiguous(),
        K_local.contiguous(),
        V_local.contiguous(),
    )
    scale = Q_local.shape[-1] ** -0.5

    # O ref full
    O_ref_full, logsumexp, *_ = torch.ops.aten._flash_attention_forward(
        Q_local,
        K_local,
        V_local,
        cum_seq_q=None,
        cum_seq_k=None,
        max_q=Nlocal,
        max_k=Nlocal,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        return_debug_mask=False,
    )
    dQ_ref, dK_ref, dV_ref = (
        torch.ops.aten._scaled_dot_product_flash_attention_backward(
            torch.ones(
                O_ref_full.shape, device=device, dtype=O_ref_full.dtype
            ).transpose(1, 2),  # grad_output: (B, N, H, d)
            Q_local.transpose(1, 2),  # query:       (B, N, H, d)
            K_local.transpose(1, 2),  # key:         (B, N, H, d)
            V_local.transpose(1, 2),  # value:       (B, N, H, d)
            O_ref_full.transpose(1, 2),  # out:         (B, N, H d)
            logsumexp,  # logsumexp:   (B*H, N) — from forward, reshape(B*H, N) before passing
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            cum_seq_q=None,
            cum_seq_k=None,
            max_q=Nlocal,
            max_k=Nlocal,
            philox_seed=torch.zeros(1, dtype=torch.long, device=Q_local.device),
            philox_offset=torch.zeros(1, dtype=torch.long, device=Q_local.device),
        )
    )

    # QKV_torchflash
    flash_Q = Q_local.detach().float().requires_grad_(True).transpose(1, 2)
    flash_K = K_local.detach().float().requires_grad_(True).transpose(1, 2)
    flash_V = V_local.detach().float().requires_grad_(True).transpose(1, 2)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        flash_out = torch.nn.functional.scaled_dot_product_attention(
            flash_Q, flash_K, flash_V, is_causal=False, dropout_p=0, attn_mask=None
        )
        # Qg grad
        flash_Q.retain_grad()
        flash_K.retain_grad()
        flash_V.retain_grad()
        flash_out.sum().backward()

    Qg = Q_local.detach().float().requires_grad_(True).transpose(1, 2)
    Kg = K_local.detach().float().requires_grad_(True).transpose(1, 2)
    Vg = V_local.detach().float().requires_grad_(True).transpose(1, 2)

    # Autograd ground truth — definitionally correct
    q, k, v = Qg, Kg, Vg
    S_au = (q @ k.transpose(-1, -2)) * scale  # (B, H, N, N)
    LSE_au = torch.logsumexp(S_au, dim=-1)  # (B, H, N)
    O_au = (torch.softmax(S_au, dim=-1) @ v).transpose(1, 2)  # (B, N, H, d)
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    O_au.sum().backward()

    dQ_au, dK_au, dV_au = q.grad.transpose(1, 2), Kg.grad, Vg.grad

    # q has shape [B, h, N, d]
    # O_ref_full has shape [B, N, h, d]
    # O_au has shape [B, h, d, N]
    # dQ_ref: [B, N, h, d]
    # dQ_au: [B, N, h, d]

    # check flash attention implementation against local implementation
    print(f"flash_V {flash_V.shape}, v grad shape {v.grad.shape}")
    assert torch.allclose(flash_out.transpose(1, 2), O_au, atol=1e-3)
    assert torch.allclose(flash_Q.grad, q.grad, atol=1e-3)
    assert torch.allclose(flash_K.grad, k.grad, atol=1e-3)
    assert torch.allclose(flash_V.grad, v.grad, atol=1e-3)

    # check flash attention against torch kernel
    assert torch.allclose(flash_out, O_ref_full.transpose(1, 2).float(), atol=1e-3)
    assert torch.allclose(LSE_au, logsumexp.float(), atol=1e-3)
    assert torch.allclose(flash_Q.grad, dQ_ref.float(), atol=1e-2)
    assert torch.allclose(flash_K.grad, dK_ref.float(), atol=1e-2)
    assert torch.allclose(flash_V.grad, dV_ref.float(), atol=1e-2)


def test_backward_sdpa(device, process_mesh):
    local_rank = dist.get_rank(group=process_mesh["row"])
    world_size = dist.get_world_size(group=process_mesh["row"])

    torch.manual_seed(2)  # same seed for weight initialization
    B, h, N, d = 1, 3, world_size * 16, 8
    Nlocal = N // world_size
    W_qkv = torch.randn(
        3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    X = torch.randn(
        B, Nlocal, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    X = X.contiguous()

    QKV = X @ W_qkv.mT

    Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 3, 1, 4)
    # QKV has shape [B, N, h, d]
    Q_local, K_local, V_local = (
        Q_local.contiguous(),
        K_local.contiguous(),
        V_local.contiguous(),
    )
    scale = Q_local.shape[-1] ** -0.5

    # O ref full
    O_ref_full, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        Q_local,
        K_local,
        V_local,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        return_debug_mask=False,
    )
    dQ_ref, dK_ref, dV_ref = (
        torch.ops.aten._scaled_dot_product_flash_attention_backward(
            torch.ones(
                O_ref_full.shape, device=device, dtype=O_ref_full.dtype
            ),  # grad_output: (B, N, H, d)
            Q_local,  # query:       (B, N, H, d)
            K_local,  # key:         (B, N, H, d)
            V_local,  # value:       (B, N, H, d)
            O_ref_full,  # out:         (B, N, H d)
            logsumexp,  # logsumexp:   (B*H, N) — from forward, reshape(B*H, N) before passing
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            cum_seq_q=None,
            cum_seq_k=None,
            max_q=Nlocal,
            max_k=Nlocal,
            philox_seed=torch.zeros(1, dtype=torch.long, device=Q_local.device),
            philox_offset=torch.zeros(1, dtype=torch.long, device=Q_local.device),
        )
    )

    # QKV_torchflash
    flash_Q = Q_local.detach().float().requires_grad_(True)
    flash_K = K_local.detach().float().requires_grad_(True)
    flash_V = V_local.detach().float().requires_grad_(True)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        flash_out = torch.nn.functional.scaled_dot_product_attention(
            flash_Q, flash_K, flash_V, is_causal=False, dropout_p=0, attn_mask=None
        )
        # Qg grad
        flash_Q.retain_grad()
        flash_K.retain_grad()
        flash_V.retain_grad()
        flash_out.sum().backward()

    Qg = Q_local.detach().float().requires_grad_(True)
    Kg = K_local.detach().float().requires_grad_(True)
    Vg = V_local.detach().float().requires_grad_(True)

    # Autograd ground truth — definitionally correct
    q, k, v = Qg, Kg, Vg
    S_au = (q @ k.transpose(-1, -2)) * scale  # (B, H, N, N)
    LSE_au = torch.logsumexp(S_au, dim=-1)  # (B, H, N)
    O_au = torch.softmax(S_au, dim=-1) @ v  # (B, N, H, d)
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    O_au.sum().backward()

    # q has shape [B, h, N, d]
    # O_ref_full has shape [B, N, h, d]
    # O_au has shape [B, h, d, N]
    # dQ_ref: [B, N, h, d]
    # dQ_au: [B, N, h, d]

    # check flash attention implementation against local implementation
    assert torch.allclose(flash_out, O_au, atol=1e-3)
    assert torch.allclose(flash_Q.grad, q.grad, atol=1e-3)
    assert torch.allclose(flash_K.grad, k.grad, atol=1e-3)
    assert torch.allclose(flash_V.grad, v.grad, atol=1e-3)

    # check flash attention against torch kernel
    assert torch.allclose(flash_out, O_ref_full.float(), atol=1e-3)
    assert torch.allclose(LSE_au, logsumexp.float(), atol=1e-3)
    assert torch.allclose(flash_Q.grad, dQ_ref.float(), atol=1e-2)
    assert torch.allclose(flash_K.grad, dK_ref.float(), atol=1e-2)
    assert torch.allclose(flash_V.grad, dV_ref.float(), atol=1e-2)


def test_backward_flash(device, process_mesh):
    local_rank = dist.get_rank(group=process_mesh["row"])
    world_size = dist.get_world_size(group=process_mesh["row"])

    torch.manual_seed(2)  # same seed for weight initialization
    B, h, N, d = 1, 3, world_size * 16, 256
    Nlocal = N // world_size
    W_qkv = torch.randn(
        3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    X = torch.randn(
        B, Nlocal, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    X = X.contiguous()

    QKV = X @ W_qkv.mT

    Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 1, 3, 4)
    # QKV has shape [B, N, h, d]
    Q_local, K_local, V_local = (
        Q_local.contiguous(),
        K_local.contiguous(),
        V_local.contiguous(),
    )
    scale = Q_local.shape[-1] ** -0.5

    # O ref full
    # flash_attention_forward takes [B, N, H, d]
    O_ref_full, logsumexp, *_ = torch.ops.aten._flash_attention_forward(
        Q_local,
        K_local,
        V_local,
        cum_seq_q=None,
        cum_seq_k=None,
        max_q=Nlocal,
        max_k=Nlocal,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        return_debug_mask=False,
    )

    dQ_ref, dK_ref, dV_ref = torch.ops.aten._flash_attention_backward(
        torch.ones(
            O_ref_full.shape, device=device, dtype=O_ref_full.dtype
        ),  # grad_output: (B, N, H, d)
        Q_local,  # query:       (B, N, H, d)
        K_local,  # key:         (B, N, H, d)
        V_local,  # value:       (B, N, H, d)
        O_ref_full,  # out:         (B, N, H d)
        logsumexp,  # logsumexp:   (B*H, N) — from forward, reshape(B*H, N) before passing
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        cum_seq_q=None,
        cum_seq_k=None,
        max_q=Nlocal,
        max_k=Nlocal,
        rng_state=None,
        unused=None,
    )

    # QKV_torchflash
    flash_Q = Q_local.detach().float().requires_grad_(True).transpose(1, 2)
    flash_K = K_local.detach().float().requires_grad_(True).transpose(1, 2)
    flash_V = V_local.detach().float().requires_grad_(True).transpose(1, 2)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        flash_out = torch.nn.functional.scaled_dot_product_attention(
            flash_Q, flash_K, flash_V, is_causal=False, dropout_p=0, attn_mask=None
        )
        # Qg grad
        flash_Q.retain_grad()
        flash_K.retain_grad()
        flash_V.retain_grad()
        flash_out.sum().backward()

    Qg = Q_local.detach().float().requires_grad_(True).transpose(1, 2)
    Kg = K_local.detach().float().requires_grad_(True).transpose(1, 2)
    Vg = V_local.detach().float().requires_grad_(True).transpose(1, 2)

    # Autograd ground truth — definitionally correct
    q, k, v = Qg, Kg, Vg
    S_au = (q @ k.transpose(-1, -2)) * scale  # (B, H, N, N)
    LSE_au = torch.logsumexp(S_au, dim=-1)  # (B, H, N)
    O_au = torch.softmax(S_au, dim=-1) @ v  # (B, H, N, d)
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    O_au.sum().backward()

    # check flash attention implementation against local implementation
    assert torch.allclose(flash_out, O_au, atol=1e-3)
    assert torch.allclose(flash_Q.grad, q.grad, atol=1e-3)
    assert torch.allclose(flash_K.grad, k.grad, atol=1e-3)
    assert torch.allclose(flash_V.grad, v.grad, atol=1e-3)

    # check flash attention against torch kernel
    assert torch.allclose(flash_out, O_ref_full.transpose(1, 2).float(), atol=1e-3)
    assert torch.allclose(LSE_au, logsumexp.float(), atol=1e-3)
    assert torch.allclose(flash_Q.grad, dQ_ref.float().transpose(1, 2), atol=1e-2)
    assert torch.allclose(flash_K.grad, dK_ref.float().transpose(1, 2), atol=1e-2)
    assert torch.allclose(flash_V.grad, dV_ref.float().transpose(1, 2), atol=1e-2)


def test_ring_forward_backward_saveKV_global(device, process_mesh):
    """Verify weights for XWT_Sync are synchronized after a few iterations."""
    local_rank = dist.get_rank(group=None)
    world_size = dist.get_world_size(group=None)

    torch.manual_seed(2)  # same seed for weight initialization
    B, h, N, d = 1, 3, world_size * 1, 16
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
        zeros=zeros,
        rank=local_rank,
        world_size=world_size,
        send_X_or_KV=0,
        save_X_or_KV=1,
        save_local_or_global=1,
        offload=False,
    )

    O_full = conftest.gather_1d_results(O, process_mesh, dim=2)
    X_full = conftest.gather_1d_results(X, process_mesh, dim=1)
    QKV = X_full @ W_qkv.mT

    Q_ref1, K_ref1, V_ref1 = QKV.reshape(B, N, 3, h, d).permute(2, 0, 3, 1, 4)

    Q_ref = conftest.gather_1d_results(Q_local, process_mesh, dim=2)
    K_ref = conftest.gather_1d_results(K_local, process_mesh, dim=2)
    V_ref = conftest.gather_1d_results(V_local, process_mesh, dim=2)

    print(f"Q_ref1 shape {Q_ref1.shape}, Q_Ref shape {Q_ref.shape}")

    assert torch.allclose(Q_ref1, Q_ref, atol=1e-4)
    assert torch.allclose(K_ref1, K_ref, atol=1e-4)
    assert torch.allclose(V_ref1, V_ref, atol=1e-4)
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
        world_size,
        chunks_to_save,
        send_X_or_KV=0,
        save_X_or_KV=1,
        save_local_or_global=1,
        offload=False,
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
    assert torch.allclose(O_full, O_ref_full, atol=1e-2)
    assert torch.allclose(dQ_full.to(torch.float32), dQ_ref.to(torch.float32), atol=1e-2)
    assert torch.allclose(dK_full, dK_ref, atol=1e-2)
    assert torch.allclose(dV_full, dV_ref, atol=1e-2)
    # assert torch.allclose(dQ_local, dQ_ref[:,:,(local_rank*Nlocal):(local_rank+1)*Nlocal], atol=1e-3)
    # assert torch.allclose(dK_local, dK_ref[:,:,(local_rank*Nlocal):(local_rank+1)*Nlocal], atol=1e-3)
    # assert torch.allclose(dV_local, dV_ref[:,:,(local_rank*Nlocal):(local_rank+1)*Nlocal], atol=1e-3)
