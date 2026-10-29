"""Ring attention keeps no chunks for backward when no backward will run."""

from typing import Any

import pytest
import torch

import jigsaw.modules.attention as attention
from jigsaw.modules.attention._config import RingAttentionConfig, SaveScope
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required, pytest.mark.dist(4)]

DIM, HEADS, BATCH, SEQ_LOCAL = 8, 2, 2, 4


@pytest.mark.parametrize("save_scope", [SaveScope.GLOBAL, SaveScope.LOCAL])
@pytest.mark.parametrize(
    "mode",
    [torch.no_grad, torch.inference_mode],
    ids=["no_grad", "inference_mode"],
)
def test_ring_saves_no_chunks_without_grad(
    device: torch.device,
    process_mesh_dict_4x1: dict,
    save_scope: SaveScope,
    mode: Any,
) -> None:
    """Output matches the grad-enabled forward, and nothing is saved."""
    mesh = process_mesh_dict_4x1
    torch.manual_seed(0)  # same weights on every rank
    model = attention.DistributedAttention(
        in_dim=DIM,
        embed_dim=DIM,
        out_dim=DIM,
        heads=HEADS,
        dropout_p=0.0,
        local_rank=mesh["col"].rank(),
        device=device,
        spatial_group=mesh["row"],
        channel_group=mesh["col"],
        ring=True,
        attn_backend="math",
        config=RingAttentionConfig(
            device=device, attn_backend="math", save_scope=save_scope
        ),
    )
    saved = 0
    save_fn = model.config.save_for_backward_fn

    def counted(*args: Any) -> Any:
        nonlocal saved
        saved += 1
        return save_fn(*args)

    model.config.save_for_backward_fn = counted

    torch.manual_seed(1 + mesh["row"].rank())
    x = torch.randn(BATCH, SEQ_LOCAL, DIM, device=device)
    y_grad = model(x)
    assert saved > 0, "the grad-enabled forward must save chunks"

    saved = 0
    with mode():
        y = model(x)
    assert saved == 0
    assert_close("output", y, y_grad.detach(), 1e-6, 1e-6, device, mesh["model"])
