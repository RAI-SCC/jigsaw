import pytest
import torch

import jigsaw.modules.normalization as norm
from jigsaw.gather import (
    gather_1d_results,
    gather_results,
)
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]


class TestDistributedLayerNorm:
    # Every mesh below spans exactly 4 ranks; without this an N != 4 run would
    # trip the world-size assertion in ``make_process_mesh``.
    pytestmark = pytest.mark.dist(4)

    @pytest.mark.parametrize(
        "process_mesh_dict",
        [
            (process_mesh_dict)
            for process_mesh_dict in [
                "process_mesh_dict_2x2",
                "process_mesh_dict_2x1x2",
                "process_mesh_dict_1x4",
                "process_mesh_dict_4x1",
                "process_mesh_dict_2x2x1",
            ]
        ],
    )
    def test_distributed_layer_norm(
        self,
        request: pytest.FixtureRequest,
        device: torch.device,
        process_mesh_dict: dict,
    ) -> None:
        """Test for forward--backward pass of layer norm."""
        process_mesh_dict = request.getfixturevalue(process_mesh_dict)
        local_dim = 12
        layer_norm = norm.DistributedLayerNorm(
            normalized_dim=local_dim * process_mesh_dict["col"].size(),
            local_rank=process_mesh_dict["model"].rank(),
            device=device,
            row_group=process_mesh_dict["row"],
            col_group=process_mesh_dict["col"],
        )

        # synchronize model initialization for weights and biases
        torch.manual_seed(process_mesh_dict["col"].rank())

        layer_norm.weight.data = torch.randn(local_dim, device=device)
        layer_norm.bias.data = torch.randn(local_dim, device=device)

        # all ranks should have different input data
        torch.manual_seed(process_mesh_dict["model"].rank())
        x = torch.randn(6, 4, local_dim, device=device)

        # gather everything for forward pass
        x_gathered = gather_results(x, process_mesh_dict)
        w_gathered = gather_1d_results(
            layer_norm.weight.data, process_mesh_dict, dim=0, pm_key="col"
        )
        b_gathered = gather_1d_results(
            layer_norm.bias.data, process_mesh_dict, dim=0, pm_key="col"
        )

        ref_layer_norm = torch.nn.LayerNorm(
            local_dim * process_mesh_dict["col"].size(), device=device
        )
        ref_layer_norm.weight.data = w_gathered
        ref_layer_norm.bias.data = b_gathered

        forward_dist = layer_norm(x)
        forward_ref = ref_layer_norm(x_gathered)
        forward_dist = gather_results(forward_dist, process_mesh_dict)

        assert_close(
            "forward",
            forward_dist,
            forward_ref,
            1e-3,
            1e-5,
            device,
            process_mesh_dict["model"],
        )

        x_dist = x.detach().clone().requires_grad_(True)
        x_ref = x_gathered.detach().clone().requires_grad_(True)

        # Need fresh ref layer norm with grad-enabled params matching distributed ones
        ref_ln = torch.nn.LayerNorm(
            local_dim * process_mesh_dict["col"].size(), device=device
        )
        ref_ln.weight.data = w_gathered.clone()
        ref_ln.bias.data = b_gathered.clone()

        # Distributed forward + backward
        # Use a deterministic, non-trivial upstream gradient
        local_grad = torch.randn(x.shape, device=device)
        grad_gathered = gather_results(local_grad, process_mesh_dict)
        layer_norm.zero_grad()
        forward_dist = layer_norm(x_dist)
        forward_dist.backward(local_grad)

        # Reference forward + backward with full grad
        y_ref = ref_ln(x_ref)
        y_ref.backward(grad_gathered)

        # Gather distributed grads
        assert isinstance(x_dist.grad, torch.Tensor)
        assert isinstance(layer_norm.weight.grad, torch.Tensor)
        assert isinstance(layer_norm.bias.grad, torch.Tensor)
        assert isinstance(x_dist.grad, torch.Tensor)
        grad_x_gathered = gather_results(x_dist.grad, process_mesh_dict)
        grad_w_gathered = gather_1d_results(
            layer_norm.weight.grad, process_mesh_dict, dim=-1, pm_key="col"
        )
        grad_b_gathered = gather_1d_results(
            layer_norm.bias.grad, process_mesh_dict, dim=-1, pm_key="col"
        )
        assert isinstance(x_ref.grad, torch.Tensor)
        assert isinstance(ref_ln.weight.grad, torch.Tensor)
        assert isinstance(ref_ln.bias.grad, torch.Tensor)

        model_group = process_mesh_dict["model"]
        assert_close(
            "input grad", grad_x_gathered, x_ref.grad, 1e-3, 1e-5, device, model_group
        )
        assert_close(
            "weight grad",
            grad_w_gathered,
            ref_ln.weight.grad,
            1e-3,
            1e-5,
            device,
            model_group,
        )
        assert_close(
            "bias grad",
            grad_b_gathered,
            ref_ln.bias.grad,
            1e-3,
            1e-5,
            device,
            model_group,
        )
