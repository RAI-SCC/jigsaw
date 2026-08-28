from typing import cast

import pytest
import torch

import jigsaw.convert as convert
from jigsaw.modules.attention import Attention, DistributedAttention
from jigsaw.modules.linear import DistributedXWT
from jigsaw.modules.roll import Roll
from jigsaw.utils_aggregate import gather_1d_results, gather_results
from tests.process_mesh import *

pytestmark = [pytest.mark.dist_required]


@pytest.mark.parametrize(
    "bias,process_mesh_dict",
    [
        (bias, process_mesh_dict)
        for bias in [True, False]
        for process_mesh_dict in ["process_mesh_dict_2x2", "process_mesh_dict_2x1x2"]
    ],
)
def test_convert_linear(
    request: pytest.FixtureRequest,
    device: torch.device,
    process_mesh_dict: dict,
    bias: bool,
) -> None:
    """Test conversion of linear layer."""
    process_mesh_dict = request.getfixturevalue(process_mesh_dict)

    # need to make sure that all ranks have the same initialization!
    torch.manual_seed(1)
    # global feature size
    in_features, out_features = 4, 2
    x_seq = 6
    linear = torch.nn.Linear(
        in_features=in_features, out_features=out_features, bias=bias, device=device
    )

    jigsaw_linear = convert.convert_to_jigsaw(linear, process_mesh_dict, device)

    assert isinstance(jigsaw_linear, DistributedXWT)
    assert hasattr(jigsaw_linear, "reshard_parameters")

    x_global = torch.randn(1, x_seq, in_features, device=device).requires_grad_(True)

    ref_forward = linear(x_global)
    ref_forward.backward(torch.ones_like(ref_forward, device=device))

    # gradients

    ref_w = linear.weight.clone()
    ref_d_w = linear.weight.grad
    ref_d_x = x_global.grad
    if bias:
        ref_b = linear.bias.clone()
        ref_d_b = linear.bias.grad

    row_rank, col_rank = (
        process_mesh_dict["row"].rank(),
        process_mesh_dict["col"].rank(),
    )
    row_size, col_size = (
        process_mesh_dict["row"].size(),
        process_mesh_dict["col"].size(),
    )

    x_local = (
        x_global[
            :,
            row_rank * x_seq // row_size : (row_rank + 1) * x_seq // row_size,
            col_rank * in_features // col_size : (col_rank + 1)
            * in_features
            // col_size,
        ]
        .contiguous()
        .detach()
        .requires_grad_(True)
    )

    # distributed calls
    dist_forward = jigsaw_linear(x_local)
    dist_forward.backward(torch.ones_like(dist_forward, device=device))

    # gather values
    dist_w = jigsaw_linear.weight
    dist_d_w = jigsaw_linear.weight.grad
    dist_d_x = x_local.grad
    assert dist_forward is not None
    assert dist_w is not None
    assert dist_d_w is not None
    assert dist_d_x is not None
    dist_forward = gather_results(dist_forward, process_mesh_dict)
    dist_w = gather_results(dist_w, process_mesh_dict)
    dist_d_w = gather_results(dist_d_w, process_mesh_dict)
    dist_d_x = gather_results(dist_d_x, process_mesh_dict)

    assert ref_w is not None
    assert ref_forward is not None
    assert ref_d_w is not None
    assert ref_d_x is not None
    assert torch.allclose(ref_w, dist_w), "weights do not match"
    assert torch.allclose(ref_forward, dist_forward), "forward does not match"
    assert torch.allclose(ref_d_w, dist_d_w), "weight grad does not match"
    assert torch.allclose(ref_d_x, dist_d_x), "dx does not match"

    # bias
    if bias:
        assert jigsaw_linear.bias is not None
        assert jigsaw_linear.bias.grad is not None
        ref_d_b = linear.bias.grad
        dist_b = jigsaw_linear.bias.data
        dist_d_b = jigsaw_linear.bias.grad
        # gather
        dist_b = gather_1d_results(
            dist_b, process_mesh_dict=process_mesh_dict, dim=0, pm_key="col"
        )
        dist_d_b = gather_1d_results(
            dist_d_b, process_mesh_dict=process_mesh_dict, dim=0, pm_key="col"
        )

        assert ref_b is not None
        assert ref_d_b is not None
        assert torch.allclose(ref_b, dist_b), "bias does not match"
        assert torch.allclose(ref_d_b, dist_d_b), "bias grad does not match"


@pytest.mark.parametrize(
    "bias,process_mesh_dict",
    [
        (bias, process_mesh_dict)
        for bias in [True, False]
        for process_mesh_dict in ["process_mesh_dict_2x2", "process_mesh_dict_2x1x2"]
    ],
)
def test_convert_mlp(
    request: pytest.FixtureRequest,
    device: torch.device,
    process_mesh_dict: dict,
    bias: bool,
) -> None:
    """Test conversion of mlp."""
    process_mesh_dict = request.getfixturevalue(process_mesh_dict)
    # need to make sure that all ranks have the same initialization!
    torch.manual_seed(1)
    in_features, hidden_dim, out_features = 4, 4, 4
    x_seq = 10

    class MLP(torch.nn.Module):
        def __init__(self, in_features: int, hidden_dim: int, out_features: int):
            """Initialize the object.

            Parameters
            ----------
            in_features : int
                Number of input features.
            hidden_dim : int
                Hidden feature dimension.
            out_features : int
                Number of output features.

            """
            super().__init__()
            self.linear1 = torch.nn.Linear(
                in_features=in_features,
                out_features=hidden_dim,
                bias=bias,
                device=device,
            )
            self.activation = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(
                in_features=hidden_dim,
                out_features=out_features,
                bias=bias,
                device=device,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the module forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor.

            Returns
            -------
            torch.Tensor
                Result returned by the operation.
            """
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x

    mlp = MLP(in_features, hidden_dim, out_features)

    # copy weights
    ref_w_1, ref_w_2 = (
        mlp.linear1.weight.clone(),
        mlp.linear2.weight.clone(),
    )

    x_global = torch.randn(1, x_seq, in_features, device=device)
    x_global.requires_grad_(True)

    ref_forward = mlp(x_global)
    ref_forward.backward(torch.ones_like(x_global, device=device))

    assert mlp.linear1.weight.grad is not None
    assert mlp.linear2.weight.grad is not None
    assert x_global.grad is not None
    ref_dw_1, ref_dw_2 = (
        mlp.linear1.weight.grad.clone(),
        mlp.linear2.weight.grad.clone(),
    )
    ref_dx = x_global.grad.clone()
    if bias:
        ref_b_1, ref_b_2 = (mlp.linear1.bias.clone(), mlp.linear2.bias.clone())
        ref_d_b_1, ref_d_b_2 = mlp.linear1.bias.grad, mlp.linear2.bias.grad

    row_rank, col_rank = (
        process_mesh_dict["row"].rank(),
        process_mesh_dict["col"].rank(),
    )
    row_size, col_size = (
        process_mesh_dict["row"].size(),
        process_mesh_dict["col"].size(),
    )

    x_local = (
        x_global[
            :,
            row_rank * x_seq // row_size : (row_rank + 1) * x_seq // row_size,
            col_rank * in_features // col_size : (col_rank + 1)
            * in_features
            // col_size,
        ]
        .contiguous()
        .detach()
        .requires_grad_(True)
    )

    jigsaw_mlp = cast(MLP, convert.convert_to_jigsaw(mlp, process_mesh_dict, device))

    dist_forward = jigsaw_mlp(x_local)
    dist_forward.backward(torch.ones_like(dist_forward, device=device))
    dist_w_1, dist_w_2 = jigsaw_mlp.linear1.weight.data, jigsaw_mlp.linear2.weight.data
    dist_dw_1, dist_dw_2 = (
        jigsaw_mlp.linear1.weight.grad,
        jigsaw_mlp.linear2.weight.grad,
    )
    dist_dx = x_local.grad

    assert isinstance(dist_forward, torch.Tensor)
    assert isinstance(dist_w_1, torch.Tensor)
    assert isinstance(dist_w_2, torch.Tensor)
    assert isinstance(dist_dw_1, torch.Tensor)
    assert isinstance(dist_dw_2, torch.Tensor)
    assert isinstance(dist_dx, torch.Tensor)

    dist_forward = gather_results(dist_forward, process_mesh_dict)
    dist_w_1 = gather_results(dist_w_1, process_mesh_dict)
    dist_w_2 = gather_results(dist_w_2, process_mesh_dict)
    dist_dw_1 = gather_results(dist_dw_1, process_mesh_dict)
    dist_dw_2 = gather_results(dist_dw_2, process_mesh_dict)
    dist_dx = gather_results(dist_dx, process_mesh_dict)

    assert isinstance(ref_w_1, torch.Tensor)
    assert isinstance(ref_w_2, torch.Tensor)
    assert isinstance(ref_forward, torch.Tensor)
    assert isinstance(ref_dw_1, torch.Tensor)
    assert isinstance(ref_dw_2, torch.Tensor)
    assert torch.allclose(ref_w_1, dist_w_1), "weights of linear1 do not match"
    assert torch.allclose(ref_w_2, dist_w_2), "weights of linear2 do not match"
    assert torch.allclose(ref_forward, dist_forward), "forward does not match"
    assert torch.allclose(ref_dw_1, dist_dw_1), "grad of linear1 weight does not match"
    assert torch.allclose(ref_dw_2, dist_dw_2), "grad of linear2 weight does not match"
    assert torch.allclose(ref_dx, dist_dx), "grad of x does not match"

    # bias terms
    if bias:
        assert isinstance(jigsaw_mlp.linear1.bias.grad, torch.Tensor)
        assert isinstance(jigsaw_mlp.linear2.bias.grad, torch.Tensor)

        dist_b_1, dist_b_2 = jigsaw_mlp.linear1.bias.data, jigsaw_mlp.linear2.bias.data
        dist_d_b_1, dist_d_b_2 = (
            jigsaw_mlp.linear1.bias.grad,
            jigsaw_mlp.linear2.bias.grad,
        )

        assert isinstance(dist_b_1, torch.Tensor)
        assert isinstance(dist_b_2, torch.Tensor)
        dist_b_1 = gather_1d_results(dist_b_1, process_mesh_dict, dim=0, pm_key="col")
        dist_b_2 = gather_1d_results(dist_b_2, process_mesh_dict, dim=0, pm_key="col")
        dist_d_b_1 = gather_1d_results(
            dist_d_b_1, process_mesh_dict, dim=0, pm_key="col"
        )
        dist_d_b_2 = gather_1d_results(
            dist_d_b_2, process_mesh_dict, dim=0, pm_key="col"
        )

        assert isinstance(ref_b_1, torch.Tensor)
        assert isinstance(ref_b_2, torch.Tensor)
        assert isinstance(ref_d_b_1, torch.Tensor)
        assert isinstance(ref_d_b_2, torch.Tensor)
        assert torch.allclose(ref_b_1, dist_b_1), "weights of bias1 do not match"
        assert torch.allclose(ref_b_2, dist_b_2), "weights of bias2 do not match"
        assert torch.allclose(ref_d_b_1, dist_d_b_1), (
            "weights of bias1 grad do not match"
        )
        assert torch.allclose(ref_d_b_2, dist_d_b_2), (
            "weights of bias2 grad do not match"
        )


def test_attention(device: torch.device, process_mesh_dict_2x1x2: dict) -> None:
    """Test conversion of multiheaded attention function."""
    # need to make sure that all ranks have the same initialization!
    torch.manual_seed(1)
    embed_dim = 256
    heads = 2
    x_seq = 10

    class Attention(torch.nn.Module):
        def __init__(self, embed_dim: int, heads: int):
            """Initialize the object.

            Parameters
            ----------
            embed_dim : int
                Global embedding dimension.
            heads : int
                Number of attention heads.

            """
            super().__init__()
            self.q = torch.nn.Linear(
                in_features=embed_dim,
                out_features=embed_dim,
                bias=False,
                device=device,
            )
            self.k = torch.nn.Linear(
                in_features=embed_dim,
                out_features=embed_dim,
                bias=False,
                device=device,
            )
            self.v = torch.nn.Linear(
                in_features=embed_dim,
                out_features=embed_dim,
                bias=False,
                device=device,
            )
            self.head_concat = torch.nn.Linear(
                in_features=embed_dim, out_features=embed_dim, bias=False, device=device
            )
            self.num_heads = heads

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the module forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor.

            Returns
            -------
            torch.Tensor
                Result returned by the operation.
            """
            q, k, v = self.q(x), self.k(x), self.v(x)

            q = q.reshape(
                q.shape[0], q.shape[1], self.num_heads, q.shape[2] // self.num_heads
            )
            k = k.reshape(
                k.shape[0], k.shape[1], self.num_heads, k.shape[2] // self.num_heads
            )
            v = v.reshape(
                v.shape[0], v.shape[1], self.num_heads, v.shape[2] // self.num_heads
            )

            q = q.permute(0, 2, 1, 3).to(torch.bfloat16)
            k = k.permute(0, 2, 1, 3).to(torch.bfloat16)
            v = v.permute(0, 2, 1, 3).to(torch.bfloat16)

            with torch.nn.attention.sdpa_kernel(
                [
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.MATH,
                ]
            ):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=False, dropout_p=0.0
                )

            # out has shape [B, h, N, d]
            out = torch.transpose(out, 1, 2).reshape(x.shape).to(x.dtype)
            values: torch.Tensor = self.head_concat(out)
            return values

    transformer = Attention(embed_dim, heads)
    # copy weights
    q_weight = transformer.q.weight.clone()
    x_global = torch.randn(1, x_seq, embed_dim, device=device)
    x_global.requires_grad_(True)

    ref_forward = transformer(x_global)
    ref_forward.backward(torch.ones_like(x_global, device=device))
    ref_q_grad = transformer.q.weight.grad
    ref_k_grad = transformer.k.weight.grad
    ref_v_grad = transformer.v.weight.grad
    ref_x_grad = x_global.grad

    _, col_rank = (
        process_mesh_dict_2x1x2["row"].rank(),
        process_mesh_dict_2x1x2["col"].rank(),
    )
    _, col_size = (
        process_mesh_dict_2x1x2["row"].size(),
        process_mesh_dict_2x1x2["col"].size(),
    )

    x_local = (
        x_global[
            :,
            :,
            col_rank * embed_dim // col_size : (col_rank + 1) * embed_dim // col_size,
        ]
        .detach()
        .requires_grad_(True)
    )
    jigsaw_attention = cast(
        Attention,
        convert.convert_to_jigsaw(
            transformer, process_mesh_dict_2x1x2, device, ring=False
        ),
    )
    jigsaw_attention.num_heads = jigsaw_attention.num_heads // 2

    dist_forward = jigsaw_attention(x_local)
    dist_forward.backward(torch.ones_like(x_local, device=device))

    dist_q_grad = jigsaw_attention.q.weight.grad
    dist_k_grad = jigsaw_attention.k.weight.grad
    dist_v_grad = jigsaw_attention.v.weight.grad
    dist_x_grad = x_local.grad

    dist_q_w = jigsaw_attention.q.weight

    assert isinstance(dist_forward, torch.Tensor)
    assert isinstance(dist_q_w, torch.Tensor)
    assert isinstance(dist_q_grad, torch.Tensor)
    assert isinstance(dist_k_grad, torch.Tensor)
    assert isinstance(dist_v_grad, torch.Tensor)
    assert isinstance(dist_x_grad, torch.Tensor)
    dist_forward = gather_1d_results(
        dist_forward, process_mesh_dict_2x1x2, dim=-1, pm_key="col"
    )
    dist_q_w = gather_results(dist_q_w, process_mesh_dict_2x1x2)
    dist_q_grad = gather_results(dist_q_grad, process_mesh_dict_2x1x2)
    dist_k_grad = gather_results(dist_k_grad, process_mesh_dict_2x1x2)
    dist_v_grad = gather_results(dist_v_grad, process_mesh_dict_2x1x2)
    dist_x_grad = gather_results(dist_x_grad, process_mesh_dict_2x1x2)

    assert isinstance(ref_forward, torch.Tensor)
    assert isinstance(q_weight, torch.Tensor)
    assert isinstance(ref_q_grad, torch.Tensor)
    assert isinstance(ref_k_grad, torch.Tensor)
    assert isinstance(ref_v_grad, torch.Tensor)
    assert isinstance(ref_x_grad, torch.Tensor)
    assert torch.allclose(ref_forward, dist_forward, atol=1e-4), (
        "forward does not match"
    )
    assert torch.allclose(q_weight, dist_q_w, atol=1e-4), "weights of qkv do not match"
    assert torch.allclose(ref_q_grad, dist_q_grad, atol=1e-4), (
        "grad of q.weight does not match"
    )
    assert torch.allclose(ref_k_grad, dist_k_grad, atol=1e-4), (
        "grad of k.weight does not match"
    )
    assert torch.allclose(ref_v_grad, dist_v_grad, atol=1e-4), (
        "grad of v.weight does not match"
    )
    assert torch.allclose(ref_x_grad, dist_x_grad, atol=1e-4), (
        "grad of x does not match"
    )


def test_jigsaw_transformer(
    device: torch.device, process_mesh_dict_2x1x2: dict
) -> None:
    """Test Attention on 2x1x2 mesh."""
    # need to make sure that all ranks have the same initialization!
    torch.manual_seed(1)
    in_features = 256
    embed_dim = 512
    out_features = 256
    heads = 4
    x_seq = 10

    transformer = Attention(
        in_features, embed_dim, out_features, heads, device, dropout_p=0.0
    )

    # copy weights
    ref_qkv_w = transformer.qkv.weight.clone()
    ref_head_cat_w = transformer.head_concat.weight.clone()
    x_global = torch.randn(1, x_seq, in_features, device=device)
    x_global.requires_grad_(True)

    ref_forward = transformer(x_global)
    ref_forward.backward(torch.ones_like(ref_forward, device=device))
    ref_qkv_grad = transformer.qkv.weight.grad
    ref_head_cat_grad = transformer.head_concat.weight.grad
    ref_x_grad = x_global.grad

    col_rank = process_mesh_dict_2x1x2["col"].rank()
    col_size = process_mesh_dict_2x1x2["col"].size()

    x_local = (
        x_global[
            :,
            :,
            col_rank * in_features // col_size : (col_rank + 1)
            * in_features
            // col_size,
        ]
        .detach()
        .contiguous()
        .requires_grad_(True)
    )

    jigsaw_attention = cast(
        DistributedAttention,
        convert.convert_to_jigsaw(
            transformer, process_mesh_dict_2x1x2, device, ring=False
        ),
    )
    dist_forward = jigsaw_attention(x_local)
    dist_forward.backward(torch.ones_like(dist_forward, device=device))

    assert isinstance(jigsaw_attention.head_concat.weight, torch.Tensor)
    assert isinstance(jigsaw_attention.head_concat.weight.grad, torch.Tensor)
    assert isinstance(jigsaw_attention.qkv.weight, torch.Tensor)
    assert isinstance(jigsaw_attention.qkv.weight.grad, torch.Tensor)
    assert isinstance(x_local.grad, torch.Tensor)
    dist_qkv_w = jigsaw_attention.qkv.weight.data
    dist_qkv_grad = jigsaw_attention.qkv.weight.grad
    dist_head_cat_w = jigsaw_attention.head_concat.weight.data
    dist_head_cat_grad = jigsaw_attention.head_concat.weight.grad
    dist_x_grad = x_local.grad

    dist_forward = gather_1d_results(
        dist_forward, process_mesh_dict_2x1x2, dim=-1, pm_key="col"
    )
    dist_qkv_w = gather_1d_results(
        dist_qkv_w, process_mesh_dict_2x1x2, pm_key="col", dim=-1
    )
    dist_head_cat_w = gather_1d_results(
        dist_head_cat_w, process_mesh_dict_2x1x2, pm_key="col", dim=-1
    )
    dist_qkv_grad = gather_1d_results(
        dist_qkv_grad, process_mesh_dict_2x1x2, pm_key="col", dim=-1
    )
    dist_x_grad = gather_1d_results(
        dist_x_grad, process_mesh_dict_2x1x2, pm_key="col", dim=-1
    )
    dist_head_cat_grad = gather_1d_results(
        dist_head_cat_grad, process_mesh_dict_2x1x2, pm_key="col", dim=-1
    )

    assert ref_forward is not None
    assert ref_qkv_w is not None
    assert ref_qkv_grad is not None
    assert ref_head_cat_w is not None
    assert ref_head_cat_grad is not None
    assert ref_x_grad is not None
    assert torch.allclose(ref_forward, dist_forward, atol=1e-5), (
        "forward does not match"
    )
    assert torch.allclose(ref_qkv_w.sum(), dist_qkv_w.sum(), atol=1e-5), (
        "sum of weights do not match"
    )
    assert torch.allclose(ref_qkv_grad.sum(), dist_qkv_grad.sum(), atol=1e-5), (
        "sum of grad weights do not match"
    )
    assert torch.allclose(ref_head_cat_w, dist_head_cat_w), "forward does not match"
    assert torch.allclose(ref_head_cat_grad, dist_head_cat_grad), (
        "grad of head concat does not match"
    )
    assert torch.allclose(ref_x_grad, dist_x_grad, atol=1e-4), (
        "grad of x does not match"
    )


def test_jigsaw_conv2d(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Test conversion of 2d convolution."""
    torch.manual_seed(1)
    in_channels = 32
    out_channels = 64
    kernel_size = 4
    h, w = 32, 32
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    conv = torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=kernel_size,
        bias=False,
        device=device,
    )
    # copy weights
    ref_conv_weights = conv.weight.clone()

    x_global = torch.randn(1, in_channels, h, w, device=device)
    x_global.requires_grad_(True)
    ref_forward = conv(x_global)
    ref_forward.backward(torch.ones_like(ref_forward, device=device))
    ref_d_w = conv.weight.grad
    ref_d_x = x_global.grad
    col_rank, col_size = (
        process_mesh_dict_2x2["col"].rank(),
        process_mesh_dict_2x2["col"].size(),
    )
    row_rank, _ = (
        process_mesh_dict_2x2["row"].rank(),
        process_mesh_dict_2x2["row"].size(),
    )

    local_w = w // process_mesh_dict_2x2["row"].size()
    x_local = (
        x_global[
            :,
            col_rank * in_channels // col_size : (col_rank + 1)
            * in_channels
            // col_size,
            :,
            row_rank * local_w : (row_rank + 1) * local_w,
        ]
        .detach()
        .contiguous()
        .requires_grad_(True)
    )

    jigsaw_conv2d = convert.convert_to_jigsaw(conv, process_mesh_dict_2x2, device)
    dist_forward = jigsaw_conv2d(x_local)

    dist_forward.backward(torch.ones_like(dist_forward, device=device))
    dist_d_x = x_local.grad

    # weights
    dist_w = jigsaw_conv2d.weight
    dist_d_w = jigsaw_conv2d.weight.grad
    # gather the results in two steps due to convolution
    dist_forward = gather_1d_results(
        dist_forward, process_mesh_dict_2x2, dim=1, pm_key="col"
    )
    dist_forward = gather_1d_results(
        dist_forward, process_mesh_dict_2x2, dim=-1, pm_key="row"
    )
    assert isinstance(dist_w, torch.Tensor)
    assert isinstance(dist_d_w, torch.Tensor)
    dist_w = gather_1d_results(dist_w, process_mesh_dict_2x2, dim=0, pm_key="col")
    dist_d_w = gather_1d_results(dist_d_w, process_mesh_dict_2x2, dim=0, pm_key="col")
    # two-step gather for dx
    assert isinstance(dist_d_x, torch.Tensor)
    dist_d_x = gather_1d_results(dist_d_x, process_mesh_dict_2x2, dim=1, pm_key="col")
    dist_d_x = gather_1d_results(dist_d_x, process_mesh_dict_2x2, dim=-1, pm_key="row")
    assert ref_forward is not None
    assert ref_conv_weights is not None
    assert ref_d_w is not None
    assert ref_d_x is not None
    assert torch.allclose(ref_forward, dist_forward, atol=1e-3), (
        "forward does not match"
    )
    assert torch.allclose(ref_conv_weights, dist_w, atol=1e-5), "weights do not match"
    assert torch.allclose(ref_d_w, dist_d_w, atol=1e-4), "weight grad does not match"
    assert torch.allclose(ref_d_x, dist_d_x, atol=1e-4), "weight grad does not match"


def test_jigsaw_conv2dtranspose(
    device: torch.device, process_mesh_dict_2x2: dict
) -> None:
    """Test conversion of Conv2DTranspose on a 2x2 process mesh."""
    torch.manual_seed(1)
    in_channels = 32
    out_channels = 32
    kernel_size = 4
    h, w = 100, 100
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv = torch.nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=kernel_size,
        bias=False,
        device=device,
    )
    # copy weights
    ref_conv_weights = conv.weight.clone()
    _ = conv.bias.clone() if conv.bias is not None else None

    x_global = torch.randn(1, in_channels, h, w, device=device)
    x_global.requires_grad_(True)

    ref_forward = conv(x_global)
    ref_forward.backward(torch.ones_like(ref_forward, device=device) / 100)
    ref_d_w = conv.weight.grad
    ref_d_x = x_global.grad

    col_rank, col_size = (
        process_mesh_dict_2x2["col"].rank(),
        process_mesh_dict_2x2["col"].size(),
    )
    row_rank, _ = (
        process_mesh_dict_2x2["row"].rank(),
        process_mesh_dict_2x2["row"].size(),
    )

    local_w = w // process_mesh_dict_2x2["row"].size()
    x_local = (
        x_global[
            :,
            col_rank * in_channels // col_size : (col_rank + 1)
            * in_channels
            // col_size,
            :,
            row_rank * local_w : (row_rank + 1) * local_w,
        ]
        .detach()
        .contiguous()
        .requires_grad_(True)
    )

    jigsaw_conv2d = convert.convert_to_jigsaw(conv, process_mesh_dict_2x2, device)
    dist_forward = jigsaw_conv2d(x_local)

    dist_forward.backward(torch.ones_like(dist_forward, device=device) / 100)
    dist_d_x = x_local.grad

    # weights
    dist_w = jigsaw_conv2d.weight
    dist_d_w = jigsaw_conv2d.weight.grad
    # gather the results in two steps due to convolution
    dist_forward = gather_1d_results(
        dist_forward, process_mesh_dict_2x2, dim=1, pm_key="col"
    )
    dist_forward = gather_1d_results(
        dist_forward, process_mesh_dict_2x2, dim=-1, pm_key="row"
    )
    assert isinstance(dist_w, torch.Tensor)
    assert isinstance(dist_d_w, torch.Tensor)
    dist_w = gather_1d_results(dist_w, process_mesh_dict_2x2, dim=1, pm_key="col")
    dist_d_w = gather_1d_results(dist_d_w, process_mesh_dict_2x2, dim=1, pm_key="col")
    # two-step gather for dx
    assert isinstance(dist_d_x, torch.Tensor)
    dist_d_x = gather_1d_results(dist_d_x, process_mesh_dict_2x2, dim=1, pm_key="col")
    dist_d_x = gather_1d_results(dist_d_x, process_mesh_dict_2x2, dim=-1, pm_key="row")

    assert ref_forward is not None
    assert ref_conv_weights is not None
    assert ref_d_w is not None
    assert ref_d_x is not None
    assert torch.allclose(ref_forward, dist_forward, atol=1e-3), (
        "forward does not match"
    )
    assert torch.allclose(ref_conv_weights, dist_w, atol=1e-5), "weights do not match"
    assert torch.allclose(ref_d_w, dist_d_w, atol=1e-4), "weight grad does not match"
    assert torch.allclose(ref_d_x, dist_d_x, atol=1e-4), "weight grad does not match"


@pytest.mark.gpu_required
def test_jigsaw_roll(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Test conversion of 2d distributed roll."""
    torch.manual_seed(1)
    in_channels = 32
    h, w = 100, 100

    roll = Roll(shifts=(3, 2), dims=(-3, -2))

    x_global = torch.randn(1, h, w, in_channels, device=device)
    x_global.requires_grad_(True)

    ref_forward = roll(x_global)

    col_rank, col_size = (
        process_mesh_dict_2x2["col"].rank(),
        process_mesh_dict_2x2["col"].size(),
    )
    row_rank, _ = (
        process_mesh_dict_2x2["row"].rank(),
        process_mesh_dict_2x2["row"].size(),
    )

    local_w = w // process_mesh_dict_2x2["row"].size()
    x_local = (
        x_global[
            :,
            :,
            row_rank * local_w : (row_rank + 1) * local_w,
            col_rank * in_channels // col_size : (col_rank + 1)
            * in_channels
            // col_size,
        ]
        .detach()
        .contiguous()
        .requires_grad_(True)
    )

    jigsaw_roll = convert.convert_to_jigsaw(roll, process_mesh_dict_2x2, device)
    dist_forward = jigsaw_roll(x_local)

    # gather the results in two steps due to rolling
    dist_forward = gather_1d_results(
        dist_forward, process_mesh_dict_2x2, dim=-1, pm_key="col"
    )
    dist_forward = gather_1d_results(
        dist_forward, process_mesh_dict_2x2, dim=-2, pm_key="row"
    )

    assert torch.allclose(ref_forward, dist_forward, atol=1e-3), (
        "forward does not match"
    )
