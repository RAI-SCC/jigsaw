import pytest
import torch

from jigsaw.modules import roll
from jigsaw.utils_aggregate import gather_1d_results
from tests.process_mesh import process_mesh_dict_2x2
pytestmark = [pytest.mark.gpu_required, pytest.mark.dist_required]


@pytest.mark.dist(4)
def test_dist_roll_init(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Test forward pass of 1d distributed roll."""
    b, x, y, c = 1, 6, 4, 24
    x_data = torch.randn(b, x, y, c, device=device)
    roll_jigsaw = roll.DistributedRoll(
        shifts=(x // 2, y // 2),
        dims=(1, 2),
        distributed_dims=(2,),
        spatial_groups=(process_mesh_dict_2x2["row"],),
        device=device,
        dtype=torch.float32,
    )
    assert isinstance(roll_jigsaw, roll.DistributedRoll)

    torch.manual_seed(process_mesh_dict_2x2["model"].rank())
    # let x data be: [B, x, y, C]
    dist_forward = roll_jigsaw(x_data)
    ref_x = gather_1d_results(x_data, process_mesh_dict_2x2, pm_key="row", dim=-2)
    ref_roll = torch.roll(ref_x, shifts=(x // 2, y // 2), dims=(1, 2))
    y_gathered = gather_1d_results(
        dist_forward, process_mesh_dict_2x2, pm_key="row", dim=-2
    )

    assert torch.allclose(ref_roll, y_gathered, atol=1e-4)

    roll_jigsaw = roll.DistributedRoll(
        shifts=(-(x // 2), -(y // 2)),
        dims=(1, 2),
        distributed_dims=(2,),
        spatial_groups=(process_mesh_dict_2x2["row"],),
        device=device,
        dtype=torch.float32,
    )
    roll_around = roll_jigsaw(dist_forward)
    assert torch.allclose(roll_around, x_data)


@pytest.mark.dist(8)
def test_dist_roll_init_8(device: torch.device, process_mesh_dict_4x2: dict) -> None:
    """Test forward pass of 1d distributed roll."""

    b, x, y, c = 1, 2, 2, 1
    x_data = torch.randn(b, x, y, c, device=device)
    roll_jigsaw = roll.DistributedRoll(
        shifts=(x, y),
        dims=(1, 2),
        distributed_dims=(2,),
        spatial_groups=(process_mesh_dict_4x2["row"],),
        device=device,
        dtype=torch.float32,
    )
    assert isinstance(roll_jigsaw, roll.DistributedRoll)

    torch.manual_seed(process_mesh_dict_4x2["model"].rank())
    # let x data be: [B, x, y, C]
    ref_x = gather_1d_results(x_data, process_mesh_dict_4x2, pm_key="row", dim=-2)
    dist_forward = roll_jigsaw(x_data)

    ref_roll = torch.roll(ref_x, shifts=(x, y), dims=(1, 2))
    y_gathered = gather_1d_results(
        dist_forward, process_mesh_dict_4x2, pm_key="row", dim=-2
    )
    assert torch.allclose(ref_roll, y_gathered, atol=1e-4)

    roll_jigsaw = roll.DistributedRoll(
        shifts=(-x, -y),
        dims=(1, 2),
        distributed_dims=(2,),
        spatial_groups=(process_mesh_dict_4x2["row"],),
        device=device,
        dtype=torch.float32,
    )
    roll_around = roll_jigsaw(dist_forward)
    assert torch.allclose(roll_around, x_data)
