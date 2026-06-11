import pytest
import torch

from jigsaw import roll
from jigsaw.utils_aggregate import gather_1d_results

pytestmark = [pytest.mark.gpu_required, pytest.mark.distributed_required]


def test_dist_roll_init(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Test forward pass of 1d distributed roll."""
    roll_jigsaw = roll.DistributedRoll(
        shifts=(3, 3),
        dims=(1, 2),
        distributed_dims=(2,),
        spatial_groups=(process_mesh_dict_2x2["row"],),
        device=device,
        dtype=torch.float32,
    )
    assert isinstance(roll_jigsaw, roll.DistributedRoll)

    torch.manual_seed(process_mesh_dict_2x2["model"].rank())
    # let x data be: [B, x, y, C]
    x_data = torch.randn(1, 6, 4, 24, device=device)
    dist_forward = roll_jigsaw(x_data)
    ref_x = gather_1d_results(x_data, process_mesh_dict_2x2, pm_key="row", dim=-2)
    ref_roll = torch.roll(ref_x, shifts=(3, 3), dims=(1, 2))
    y_gathered = gather_1d_results(
        dist_forward, process_mesh_dict_2x2, pm_key="row", dim=-2
    )

    assert torch.allclose(ref_roll, y_gathered, atol=1e-4)
