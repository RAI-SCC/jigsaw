import torch

import torch
import torch.distributed as dist

import jigsaw.linear as linear
from jigsaw.linear import DistributedXWT
import jigsaw.convert as convert
from jigsaw.utils_aggregate import gather_results
from process_mesh import process_mesh_2x2, process_mesh_2x1x2


def test_convert_linear(device, process_mesh_2x2):
    # need to make sure that all ranks have the same initialization!
    torch.manual_seed(1)
    in_features, out_features = 4, 2
    x_seq = 10
    linear = torch.nn.Linear(
        in_features=in_features, out_features=out_features, bias=False, device=device
    )
    linear_weight = linear.weight.clone()
    jigsaw_linear = convert.convert_to_jigsaw(linear, False, process_mesh_2x2, device)

    assert isinstance(jigsaw_linear, DistributedXWT)
    assert hasattr(jigsaw_linear, "reshard_parameters")

    x_global = torch.randn(x_seq, in_features, device=device)
    row_rank, col_rank = process_mesh_2x2["row"].rank(), process_mesh_2x2["col"].rank()
    row_size, col_size = process_mesh_2x2["row"].size(), process_mesh_2x2["col"].size()

    x_local = x_global[
        row_rank * x_seq // row_size : (row_rank + 1) * x_seq // row_size,
        col_rank * in_features // col_size : (col_rank + 1) * in_features // col_size,
    ].contiguous()
    ref_forward = linear(x_global)
    dist_forward = jigsaw_linear(x_local)
    dist_forward = gather_results(dist_forward, process_mesh_2x2)
    dist_w = jigsaw_linear.weight
    dist_w = gather_results(dist_w, process_mesh_2x2)

    assert torch.allclose(linear_weight, dist_w), "weights do not match"
    assert torch.allclose(ref_forward, dist_forward), "forward does not match"

def test_convert_mlp(device, process_mesh_2x2):
    # need to make sure that all ranks have the same initialization!
    torch.manual_seed(1)
    in_features, hidden_dim, out_features = 4, 4, 4
    x_seq = 10
    
    class MLP(torch.nn.Module):
        def __init__(self, in_features, hidden_dim, out_features):
            super().__init__()
            self.linear1 = torch.nn.Linear(
                in_features=in_features, out_features=hidden_dim, bias=False, device=device
            )
            self.activation = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(
                in_features=hidden_dim, out_features=out_features, bias=False, device=device
            )
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x
    
    mlp = MLP(in_features, hidden_dim, out_features)
    
    # copy weights
    linear1_weight, linear2_weight = mlp.linear1.weight.clone(), mlp.linear2.weight.clone()
    x_global = torch.randn(x_seq, in_features, device=device)
    
    ref_forward = mlp(x_global)
    
    row_rank, col_rank = process_mesh_2x2["row"].rank(), process_mesh_2x2["col"].rank()
    row_size, col_size = process_mesh_2x2["row"].size(), process_mesh_2x2["col"].size()

    x_local = x_global[
        row_rank * x_seq // row_size : (row_rank + 1) * x_seq // row_size,
        col_rank * in_features // col_size : (col_rank + 1) * in_features // col_size,
    ].contiguous()
    
    jigsaw_mlp = convert.convert_to_jigsaw(mlp, False, process_mesh_2x2, device)

    dist_forward = jigsaw_mlp(x_local)
    dist_forward = gather_results(dist_forward, process_mesh_2x2)
    dist_w_1, dist_w_2 = jigsaw_mlp.linear1.weight, jigsaw_mlp.linear2.weight
    dist_w_1 = gather_results(dist_w_1, process_mesh_2x2)
    dist_w_2 = gather_results(dist_w_2, process_mesh_2x2)

    assert torch.allclose(linear1_weight, dist_w_1), "weights of linear1 do not match"
    assert torch.allclose(linear2_weight, dist_w_2), "weights of linear2 not match"
    assert torch.allclose(ref_forward, dist_forward), "forward does not match"
    
def test_convert_mlp(device, process_mesh_2x2):
    # need to make sure that all ranks have the same initialization!
    torch.manual_seed(1)
    in_features, hidden_dim, out_features = 4, 4, 4
    x_seq = 10
    
    class MLP(torch.nn.Module):
        def __init__(self, in_features, hidden_dim, out_features):
            super().__init__()
            self.linear1 = torch.nn.Linear(
                in_features=in_features, out_features=hidden_dim, bias=False, device=device
            )
            self.activation = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(
                in_features=hidden_dim, out_features=out_features, bias=False, device=device
            )
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x
    
    mlp = MLP(in_features, hidden_dim, out_features)
    
    # copy weights
    linear1_weight, linear2_weight = mlp.linear1.weight.clone(), mlp.linear2.weight.clone()
    x_global = torch.randn(x_seq, in_features, device=device)
    
    ref_forward = mlp(x_global)
    
    row_rank, col_rank = process_mesh_2x2["row"].rank(), process_mesh_2x2["col"].rank()
    row_size, col_size = process_mesh_2x2["row"].size(), process_mesh_2x2["col"].size()

    x_local = x_global[
        row_rank * x_seq // row_size : (row_rank + 1) * x_seq // row_size,
        col_rank * in_features // col_size : (col_rank + 1) * in_features // col_size,
    ].contiguous()
    
    jigsaw_mlp = convert.convert_to_jigsaw(mlp, False, process_mesh_2x2, device)

    dist_forward = jigsaw_mlp(x_local)
    dist_forward = gather_results(dist_forward, process_mesh_2x2)
    dist_w_1, dist_w_2 = jigsaw_mlp.linear1.weight, jigsaw_mlp.linear2.weight
    dist_w_1 = gather_results(dist_w_1, process_mesh_2x2)
    dist_w_2 = gather_results(dist_w_2, process_mesh_2x2)

    assert torch.allclose(linear1_weight, dist_w_1), "weights of linear1 do not match"
    assert torch.allclose(linear2_weight, dist_w_2), "weights of linear2 not match"
    assert torch.allclose(ref_forward, dist_forward), "forward does not match"