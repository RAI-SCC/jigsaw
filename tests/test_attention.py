import pytest
import torch

import jigsaw.attention as attention
from jigsaw.utils_aggregate import (
    gather_1d_results_list,
)

from .process_mesh import *

pytestmark = [pytest.mark.distributed_required]


def test_init_attention(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    """Initialize the attention module."""
    in_dim, embed_dim, out_dim = 256, 256, 256
    heads = 4
    model = attention.DistributedAttention(
        in_dim=in_dim,
        embed_dim=embed_dim,
        out_dim=out_dim,
        heads=heads,
        dropout_p=0.0,
        local_rank=process_mesh_dict_2x2["col"].rank(),
        device=device,
        spatial_group=process_mesh_dict_2x2["row"],
        channel_group=process_mesh_dict_2x2["col"],
        bayesian=False,
        qk_norm=False,
        ring_attention=False,
    )

    assert isinstance(model, attention.DistributedAttention)
    weights = gather_1d_results_list(
        model.qkv.weight, process_mesh_dict=process_mesh_dict_2x2, pm_key="model"
    )
    assert torch.allclose(weights[0], weights[2])
    assert not torch.allclose(weights[0], weights[1])


def test_attention_sync(device: torch.device, process_mesh_dict_2x2: dict) -> None:
    in_dim, embed_dim, out_dim = 256, 256, 256
    heads = 4
    model = attention.DistributedAttention(
        in_dim=in_dim,
        embed_dim=embed_dim,
        out_dim=out_dim,
        heads=heads,
        dropout_p=0.0,
        local_rank=process_mesh_dict_2x2["col"].rank(),
        device=device,
        spatial_group=process_mesh_dict_2x2["row"],
        channel_group=process_mesh_dict_2x2["col"],
        bayesian=False,
        qk_norm=False,
        ring_attention=False,
    )
    batch, local_seq_len = 10, 14
    local_channels = in_dim // process_mesh_dict_2x2["col"].size()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    for _ in range(10):
        x = torch.randn(batch, local_seq_len, local_channels, device=device)
        pred = model(x)
        y = torch.randn_like(pred)
        loss = loss_fn(y, pred)
        loss.backward()
        optimizer.step()

    weights = gather_1d_results_list(
        model.qkv.weight, process_mesh_dict=process_mesh_dict_2x2, pm_key="model"
    )
    assert torch.allclose(weights[0], weights[2])
    assert not torch.allclose(weights[0], weights[1])
