import torch
import pytest
import jigsaw.normalization as norm

from jigsaw.utils_aggregate import (
    gather_1d_results,
    gather_1d_results_list,
    gather_results,
)

from process_mesh import process_mesh_2x2, process_mesh_1x4, process_mesh_2x1x2, process_mesh_4x1, process_mesh_2x2x1


@pytest.mark.parametrize(
    "process_mesh",
    [
        (process_mesh)
        for process_mesh in [
            "process_mesh_2x2",
            "process_mesh_2x1x2",
            "process_mesh_1x4",
            "process_mesh_4x1",
            "process_mesh_2x2x1",
        ]
    ],
)
def test_distributed_layer_norm(request, device, process_mesh):
    process_mesh = request.getfixturevalue(process_mesh)
    local_dim = 12
    layer_norm = norm.DistributedLayerNorm(
        local_dim=local_dim,
        local_rank=process_mesh["model"].rank(),
        device=device,
        spatial_group=process_mesh["row"],
        channel_group=process_mesh["col"],
    )

    # synchronize model initialization for weights and biases
    torch.manual_seed(process_mesh["col"].rank())

    layer_norm.weight.data = torch.randn(local_dim, device=device)
    layer_norm.bias.data = torch.randn(local_dim, device=device)

    # all ranks should have different input data
    torch.manual_seed(process_mesh["model"].rank())
    x = torch.randn(6, 4, local_dim, device=device)

    # gather everything for forward pass
    x_gathered = gather_results(x, process_mesh)
    w_gathered = gather_1d_results(
        layer_norm.weight.data, process_mesh, dim=0, pm_key="col"
    )
    b_gathered = gather_1d_results(
        layer_norm.bias.data, process_mesh, dim=0, pm_key="col"
    )
    
    ref_layer_norm = torch.nn.LayerNorm(
        local_dim * process_mesh["col"].size(), device=device
    )
    ref_layer_norm.weight.data = w_gathered
    ref_layer_norm.bias.data = b_gathered


    forward_dist = layer_norm(x)
    forward_ref = ref_layer_norm(x_gathered)
    forward_dist = gather_results(
        forward_dist, process_mesh
    )

    assert torch.allclose(forward_dist, forward_ref, atol=1e-3)

    x_dist = x.detach().clone().requires_grad_(True)
    x_ref = x_gathered.detach().clone().requires_grad_(True)

    # Need fresh ref layer norm with grad-enabled params matching distributed ones
    ref_ln = torch.nn.LayerNorm(local_dim * process_mesh["col"].size(), device=device)
    ref_ln.weight.data = w_gathered.clone()
    ref_ln.bias.data = b_gathered.clone()

    # Distributed forward + backward
    # Use a deterministic, non-trivial upstream gradient
    local_grad = torch.randn(x.shape, device=device)
    grad_gathered = gather_results(local_grad, process_mesh)
    layer_norm.zero_grad()
    forward_dist = layer_norm(x_dist)
    forward_dist.backward(local_grad)

    # Reference forward + backward with full grad
    y_ref = ref_ln(x_ref)
    y_ref.backward(grad_gathered)

    # Gather distributed grads
    grad_x_gathered = gather_results(x_dist.grad, process_mesh)
    grad_w_gathered = gather_1d_results(layer_norm.weight.grad, process_mesh, dim=-1, pm_key="col")
    grad_b_gathered = gather_1d_results(layer_norm.bias.grad, process_mesh, dim=-1, pm_key="col")

    assert torch.allclose(grad_x_gathered, x_ref.grad, atol=1e-3)
    assert torch.allclose(grad_w_gathered, ref_ln.weight.grad, atol=1e-3)
    assert torch.allclose(grad_b_gathered, ref_ln.bias.grad, atol=1e-3)