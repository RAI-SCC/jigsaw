import pytest
import torch

import jigsaw.modules.attention as attention
from jigsaw.utils_aggregate import (
    gather_1d_results_list,
)
from tests.parity import all_ranks_ok, assert_close

pytestmark = [pytest.mark.dist_required]


def _assert_qkv_replication(
    weights: list[torch.Tensor], device: torch.device, model_group: object
) -> None:
    """qkv shards agree across the spatial (row) group and differ across the channel (col) group.

    ``weights`` is one shard per model-group rank; with a row-major 2x2 mesh rank
    ``r`` is ``(row, col) = divmod(r, 2)``, so index 0 and 2 share a channel column
    (must match after the spatial grad-sync) and 0 and 1 do not (must differ).
    Routed through the parity helpers so a mismatch on a subset of ranks fails the
    job instead of hanging the rest in the next collective.
    """
    assert_close(
        "qkv replicated across spatial group",
        weights[0],
        weights[2],
        1e-8,
        1e-5,
        device,
        model_group,
    )
    differ = not torch.allclose(weights[0], weights[1])
    assert all_ranks_ok(differ, device, model_group), (
        "qkv shards must differ across the channel group"
    )


class TestDistributedAttention:
    @pytest.mark.dist(4)
    def test_init_attention(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
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
        _assert_qkv_replication(weights, device, process_mesh_dict_2x2["model"])

    @pytest.mark.dist(4)
    def test_attention_sync(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Verify synchronized distributed attention.

        Parameters
        ----------
        device : torch.device
            Device on which tensors and communication are placed.
        process_mesh_dict_2x2 : dict
            Two-by-two process mesh used by the test.

        """
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
        _assert_qkv_replication(weights, device, process_mesh_dict_2x2["model"])
