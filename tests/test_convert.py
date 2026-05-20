import torch
import torch.distributed as dist

import jigsaw.linear as linear
from jigsaw.linear import DistributedXWT
from jigsaw.attention import Transformer
import jigsaw.convert as convert
from jigsaw.utils_aggregate import gather_results, gather_1d_results
from process_mesh import process_mesh_2x2, process_mesh_2x1x2


# def test_convert_linear(device, process_mesh_2x2):
#     # need to make sure that all ranks have the same initialization!
#     torch.manual_seed(1)
#     # global feature size
#     in_features, out_features = 4, 2
#     x_seq = 6
#     linear = torch.nn.Linear(
#         in_features=in_features, out_features=out_features, bias=False, device=device
#     )
#     linear_weight = linear.weight.clone()
#     print(f'linear weight shape {linear_weight.shape}')
#     jigsaw_linear = convert.convert_to_jigsaw(linear, False, process_mesh_2x2, device)

#     assert isinstance(jigsaw_linear, DistributedXWT)
#     assert hasattr(jigsaw_linear, "reshard_parameters")

#     x_global = torch.randn(x_seq, in_features, device=device)
#     row_rank, col_rank = process_mesh_2x2["row"].rank(), process_mesh_2x2["col"].rank()
#     row_size, col_size = process_mesh_2x2["row"].size(), process_mesh_2x2["col"].size()

#     x_local = x_global[
#         row_rank * x_seq // row_size : (row_rank + 1) * x_seq // row_size,
#         col_rank * in_features // col_size : (col_rank + 1) * in_features // col_size,
#     ].contiguous()
    
#     ref_forward = linear(x_global)
#     dist_forward = jigsaw_linear(x_local)
#     dist_forward = gather_results(dist_forward, process_mesh_2x2)
#     dist_w = jigsaw_linear.weight
#     dist_w = gather_results(dist_w, process_mesh_2x2)
    
    
#     assert torch.allclose(linear_weight, dist_w), "weights do not match"
#     assert torch.allclose(ref_forward, dist_forward), "forward does not match"


# def test_convert_linear2(device, process_mesh_2x1x2):
#     # need to make sure that all ranks have the same initialization!
#     torch.manual_seed(1)
#     in_features, out_features = 4, 2
#     x_seq = 10
#     linear = torch.nn.Linear(
#         in_features=in_features, out_features=out_features, bias=False, device=device
#     )
#     linear_weight = linear.weight.clone()
#     jigsaw_linear = convert.convert_to_jigsaw(linear, False, process_mesh_2x1x2, device)

#     assert isinstance(jigsaw_linear, DistributedXWT)
#     assert hasattr(jigsaw_linear, "reshard_parameters")

#     x_global = torch.randn(1, x_seq, in_features, device=device)
#     row_rank, col_rank = (
#         process_mesh_2x1x2["row"].rank(),
#         process_mesh_2x1x2["col"].rank(),
#     )
#     row_size, col_size = (
#         process_mesh_2x1x2["row"].size(),
#         process_mesh_2x1x2["col"].size(),
#     )

#     x_local = x_global[
#         :,
#         :,
#         col_rank * in_features // col_size : (col_rank + 1) * in_features // col_size,
#     ].contiguous()
#     ref_forward = linear(x_global)
#     dist_forward = jigsaw_linear(x_local)
#     dist_forward = gather_results(dist_forward, process_mesh_2x1x2)
#     dist_w = jigsaw_linear.weight
#     dist_w = gather_results(dist_w, process_mesh_2x1x2)
#     print(f'previous shape {linear_weight.shape} jigsaw shape {jigsaw_linear.weight.shape}')

#     assert torch.allclose(linear_weight, dist_w), "weights do not match"
#     assert torch.allclose(ref_forward, dist_forward), "forward does not match"


# def test_convert_mlp(device, process_mesh_2x2):
#     # need to make sure that all ranks have the same initialization!
#     torch.manual_seed(1)
#     in_features, hidden_dim, out_features = 4, 4, 4
#     x_seq = 10

#     class MLP(torch.nn.Module):
#         def __init__(self, in_features, hidden_dim, out_features):
#             super().__init__()
#             self.linear1 = torch.nn.Linear(
#                 in_features=in_features,
#                 out_features=hidden_dim,
#                 bias=False,
#                 device=device,
#             )
#             self.activation = torch.nn.ReLU()
#             self.linear2 = torch.nn.Linear(
#                 in_features=hidden_dim,
#                 out_features=out_features,
#                 bias=False,
#                 device=device,
#             )

#         def forward(self, x):
#             x = self.linear1(x)
#             x = self.activation(x)
#             x = self.linear2(x)
#             return x

#     mlp = MLP(in_features, hidden_dim, out_features)

#     # copy weights
#     linear1_weight, linear2_weight = (
#         mlp.linear1.weight.clone(),
#         mlp.linear2.weight.clone(),
#     )
#     x_global = torch.randn(1, x_seq, in_features, device=device)
#     x_global.requires_grad_(True)

#     ref_forward = mlp(x_global)
#     ref_forward.backward(torch.ones_like(x_global, device=device))
#     ref_dw_1, ref_dw_2 = (
#         mlp.linear1.weight.grad.clone(),
#         mlp.linear2.weight.grad.clone(),
#     )
#     ref_dx = x_global.grad.clone()

#     row_rank, col_rank = process_mesh_2x2["row"].rank(), process_mesh_2x2["col"].rank()
#     row_size, col_size = process_mesh_2x2["row"].size(), process_mesh_2x2["col"].size()

#     x_local = (
#         x_global[
#             :,
#             row_rank * x_seq // row_size : (row_rank + 1) * x_seq // row_size,
#             col_rank
#             * in_features
#             // col_size : (col_rank + 1)
#             * in_features
#             // col_size,
#         ]
#         .contiguous()
#         .detach()
#         .requires_grad_(True)
#     )

#     jigsaw_mlp = convert.convert_to_jigsaw(mlp, False, process_mesh_2x2, device)

#     dist_forward = jigsaw_mlp(x_local)
#     dist_forward.backward(torch.ones_like(dist_forward, device=device))
#     dist_w_1, dist_w_2 = jigsaw_mlp.linear1.weight, jigsaw_mlp.linear2.weight
#     dist_dw_1, dist_dw_2 = (
#         jigsaw_mlp.linear1.weight.grad,
#         jigsaw_mlp.linear2.weight.grad,
#     )
#     dist_dx = x_local.grad

#     dist_forward = gather_results(dist_forward, process_mesh_2x2)
#     dist_w_1 = gather_results(dist_w_1, process_mesh_2x2)
#     dist_w_2 = gather_results(dist_w_2, process_mesh_2x2)
#     dist_dw_1 = gather_results(dist_dw_1, process_mesh_2x2)
#     dist_dw_2 = gather_results(dist_dw_2, process_mesh_2x2)
#     dist_dx = gather_results(dist_dx, process_mesh_2x2)

#     assert torch.allclose(linear1_weight, dist_w_1), "weights of linear1 do not match"
#     assert torch.allclose(linear2_weight, dist_w_2), "weights of linear2 not match"
#     assert torch.allclose(ref_forward, dist_forward), "forward does not match"
#     assert torch.allclose(ref_dw_1, dist_dw_1), "grad of linear1 weight does not match"
#     assert torch.allclose(ref_dw_2, dist_dw_2), "grad of linear2 weight does not match"
#     assert torch.allclose(ref_dx, dist_dx), "grad of x does not match"


# def test_transformer(device, process_mesh_2x1x2):
#     # need to make sure that all ranks have the same initialization!
#     torch.manual_seed(1)
#     embed_dim = 256
#     heads = 2
#     x_seq = 10

#     class Transformer(torch.nn.Module):
#         def __init__(self, embed_dim, heads):
#             super().__init__()
#             self.q = torch.nn.Linear(
#                 in_features=embed_dim,
#                 out_features=embed_dim,
#                 bias=False,
#                 device=device,
#             )
#             self.k = torch.nn.Linear(
#                 in_features=embed_dim,
#                 out_features=embed_dim,
#                 bias=False,
#                 device=device,
#             )
#             self.v = torch.nn.Linear(
#                 in_features=embed_dim,
#                 out_features=embed_dim,
#                 bias=False,
#                 device=device,
#             )
#             self.head_concat = torch.nn.Linear(
#                 in_features=embed_dim, out_features=embed_dim, bias=False, device=device
#             )
#             self.num_heads = heads
            
#         def forward(self, x):
#             q, k, v = self.q(x), self.k(x), self.v(x)

#             q = q.reshape(
#                 q.shape[0], q.shape[1], self.num_heads, q.shape[2] // self.num_heads
#             )
#             k = k.reshape(
#                 k.shape[0], k.shape[1], self.num_heads, k.shape[2] // self.num_heads
#             )
#             v = v.reshape(
#                 v.shape[0], v.shape[1], self.num_heads, v.shape[2] // self.num_heads
#             )
            
#             q = q.permute(0, 2, 1, 3).to(torch.bfloat16)
#             k = k.permute(0, 2, 1, 3).to(torch.bfloat16)
#             v = v.permute(0, 2, 1, 3).to(torch.bfloat16)

#             with torch.nn.attention.sdpa_kernel(
#                 torch.nn.attention.SDPBackend.FLASH_ATTENTION
#             ):
#                 out = torch.nn.functional.scaled_dot_product_attention(
#                     q, k, v, is_causal=False, dropout_p=0.0
#                 )

#             # out has shape [B, h, N, d]
#             out = torch.transpose(out, 1, 2).reshape(x.shape).to(x.dtype)
#             values = self.head_concat(out)
#             return values

#     transformer = Transformer(embed_dim, heads)

#     # copy weights
#     q_weight = transformer.q.weight.clone()
#     x_global = torch.randn(1, x_seq, embed_dim, device=device)
#     x_global.requires_grad_(True)

#     ref_forward = transformer(x_global)
#     ref_forward.backward(torch.ones_like(x_global, device=device))
#     ref_q_grad = transformer.q.weight.grad
#     ref_k_grad = transformer.k.weight.grad
#     ref_v_grad = transformer.v.weight.grad
#     ref_x_grad = x_global.grad

#     row_rank, col_rank = (
#         process_mesh_2x1x2["row"].rank(),
#         process_mesh_2x1x2["col"].rank(),
#     )
#     row_size, col_size = (
#         process_mesh_2x1x2["row"].size(),
#         process_mesh_2x1x2["col"].size(),
#     )

#     x_local = (
#         x_global[
#             :,
#             :,
#             col_rank * embed_dim // col_size : (col_rank + 1) * embed_dim // col_size,
#         ]
#         .detach()
#         .requires_grad_(True)
#     )
#     jigsaw_transformer = convert.convert_to_jigsaw(
#         transformer, False, process_mesh_2x1x2, device
#     )
#     jigsaw_transformer.num_heads = jigsaw_transformer.num_heads // 2

    
#     dist_forward = jigsaw_transformer(x_local)
#     dist_forward.backward(torch.ones_like(x_local, device=device))

#     dist_q_grad = jigsaw_transformer.q.weight.grad
#     dist_k_grad = jigsaw_transformer.k.weight.grad
#     dist_v_grad = jigsaw_transformer.v.weight.grad
#     dist_x_grad = x_local.grad

#     dist_q_w = jigsaw_transformer.q.weight

#     dist_forward = gather_1d_results(
#         dist_forward, process_mesh_2x1x2, dim=-1, pm_key="col"
#     )
#     dist_q_w = gather_results(dist_q_w, process_mesh_2x1x2)
#     dist_q_grad = gather_results(dist_q_grad, process_mesh_2x1x2)
#     dist_k_grad = gather_results(dist_k_grad, process_mesh_2x1x2)
#     dist_v_grad = gather_results(dist_v_grad, process_mesh_2x1x2)
#     dist_x_grad = gather_results(dist_x_grad, process_mesh_2x1x2)

#     assert torch.allclose(ref_forward, dist_forward), "forward does not match"
#     assert torch.allclose(q_weight, dist_q_w), "weights of qkv do not match"
#     assert torch.allclose(ref_q_grad, dist_q_grad), "grad of q.weight does not match"
#     assert torch.allclose(ref_k_grad, dist_k_grad), "grad of k.weight does not match"
#     assert torch.allclose(ref_v_grad, dist_v_grad), "grad of v.weight does not match"
#     assert torch.allclose(ref_x_grad, dist_x_grad, atol=1e-4), "grad of x does not match"


def test_jigsaw_transformer(device, process_mesh_2x1x2):
    # need to make sure that all ranks have the same initialization!
    torch.manual_seed(1)
    embed_dim = 256
    heads = 2
    x_seq = 10

    transformer = Transformer(embed_dim, heads, device, dropout_p=0.0)

    # copy weights
    qkv_weight = transformer.qkv.weight.clone()
    ref_head_cat_w = transformer.head_concat.weight.clone()
    x_global = torch.randn(1, x_seq, embed_dim, device=device)
    x_global.requires_grad_(True)

    ref_forward = transformer(x_global)
    ref_forward.backward(torch.ones_like(x_global, device=device))
    ref_qkv_grad = transformer.qkv.weight.grad
    ref_x_grad = x_global.grad

    row_rank, col_rank = (
        process_mesh_2x1x2["row"].rank(),
        process_mesh_2x1x2["col"].rank(),
    )
    row_size, col_size = (
        process_mesh_2x1x2["row"].size(),
        process_mesh_2x1x2["col"].size(),
    )

    x_local = (
        x_global[
            :,
            :,
            col_rank * embed_dim // col_size : (col_rank + 1) * embed_dim // col_size,
        ]
        .detach()
        .contiguous()
        .requires_grad_(True)
    )

    jigsaw_transformer = convert.convert_to_jigsaw(
        transformer, False, process_mesh_2x1x2, device
    )
    dist_forward = jigsaw_transformer(x_local)
    dist_forward.backward(torch.ones_like(x_local, device=device))
    dist_head_cat_w = jigsaw_transformer.head_concat.weight
    
    dist_x_grad = x_local.grad

    # dist_qkv_w = jigsaw_transformer.qkv.weight

    dist_forward = gather_1d_results(
        dist_forward, process_mesh_2x1x2, dim=-1, pm_key="col"
    )
    dist_head_cat_w = gather_1d_results(dist_head_cat_w, process_mesh_2x1x2, pm_key="col", dim=-1)
    dist_x_grad = gather_1d_results(dist_x_grad, process_mesh_2x1x2, pm_key="col", dim=-1)
    # print(f'sums {ref_forward.sum()} dist forward {dist_forward.sum()}')
    
    assert torch.allclose(ref_head_cat_w, dist_head_cat_w), "forward does not match"
    assert torch.allclose(ref_forward, dist_forward, atol=1e-5), "forward does not match"
    assert torch.allclose(ref_x_grad, dist_x_grad, atol=1e-4), "grad of x does not match"
