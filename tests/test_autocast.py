import pytest
import torch

from jigsaw.modules.attention import DistributedAttention

# @pytest.mark.dist(4)
# def test_autocast_linear(process_mesh_dict_2x2: dict, device) -> None:
#     process_mesh_dict = process_mesh_dict_2x2
#     local_rank = process_mesh_dict["model"].rank()
#     batch, seq, in_dim, out_dim = 1, 20, 10, 20
#     model = DistributedXWT(
#         in_dim,
#         out_dim,
#         bias=False,
#         local_rank=local_rank,
#         device=device,
#         model_group=process_mesh_dict["model"],
#         spatial_group=process_mesh_dict["row"],
#         channel_group=process_mesh_dict["col"],
#     )

#     mse_loss = torch.nn.MSELoss()

#     for _ in range(5):
#         x = torch.randn(batch, seq // 2, in_dim // 2)
#         with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
#             y = model(x)
#         loss = mse_loss(y, torch.ones_like(y))

#         loss.backward()  # backward outside autocast


@pytest.mark.dist(4)
def test_autocast_attention(process_mesh_dict_2x2: dict, device) -> None:
    process_mesh_dict = process_mesh_dict_2x2
    local_rank = process_mesh_dict["model"].rank()
    batch, seq, in_dim, out_dim = 1, 2, 8, 8
    model = DistributedAttention(
        in_dim,
        embed_dim=out_dim,
        out_dim=out_dim,
        heads=4,
        dropout_p=0.0,
        local_rank=local_rank,
        device=device,
        spatial_group=process_mesh_dict["row"],
        channel_group=process_mesh_dict["col"],
        ring_attention=False,
    )

    mse_loss = torch.nn.MSELoss()

    for _ in range(5):
        x = torch.randn(batch, seq // 2, in_dim // 2, device=device).contiguous()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            y = model(x)
        loss = mse_loss(y, torch.ones_like(y))

        loss.backward()  # backward outside autocast
