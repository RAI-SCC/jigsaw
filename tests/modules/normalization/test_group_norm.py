"""``DistributedGroupNorm``: local forward, and resharding of affine / bias-free layers.

The layer normalises each channel shard on its own, which equals the matching slice of a
global ``GroupNorm`` as long as no group straddles two shards. ``test_reshard_convert_coverage``
only checks the reshard assertions, so the forward pass and the ``affine=False`` /
``bias=False`` reshard branches are covered here.
"""

import pytest
import torch

from jigsaw.modules import normalization as norm
from tests.parity import all_ranks_ok, assert_close

GROUPS, CHANNELS = 4, 8


def _reference(affine: bool, bias: bool, device: torch.device) -> torch.nn.GroupNorm:
    ref = torch.nn.GroupNorm(GROUPS, CHANNELS, affine=affine, device=device)
    if affine:
        with torch.no_grad():
            ref.weight.copy_(torch.linspace(0.5, 2.0, CHANNELS))
            ref.bias.copy_(torch.linspace(-1.0, 1.0, CHANNELS))
        if not bias:
            ref.register_parameter("bias", None)
    return ref


@pytest.mark.parametrize("affine, bias", [(True, True), (True, False), (False, True)])
def test_unsharded_forward_matches_torch(device, affine, bias):
    layer = norm.DistributedGroupNorm(
        GROUPS, CHANNELS, affine=affine, bias=bias, device=device
    )
    ref = _reference(affine, bias, device)
    if affine:
        layer.group_norm.weight.data = ref.weight.data.clone()
        assert (layer.group_norm.bias is None) == (not bias)
        if bias:
            layer.group_norm.bias.data = ref.bias.data.clone()
    x = torch.randn(2, CHANNELS, 5, 3, device=device)
    assert torch.allclose(layer(x), ref(x), atol=1e-6)


@pytest.mark.dist(4)
class TestReshardFourRanks:
    @pytest.mark.parametrize(
        "affine, bias", [(True, True), (True, False), (False, True)]
    )
    def test_reshard_col2_to_col4_keeps_the_channel_slice(
        self,
        device,
        process_mesh_dict_2x2: dict,
        process_mesh_dict_1x4: dict,
        affine,
        bias,
    ):
        src, dst = process_mesh_dict_2x2, process_mesh_dict_1x4
        layer = norm.DistributedGroupNorm(
            GROUPS // 2,
            CHANNELS // 2,
            affine=affine,
            bias=bias,
            device=device,
            channel_group=src["col"],
        )
        ref = _reference(affine, bias, device)
        if affine:
            c = CHANNELS // 2
            lo = src["col"].rank() * c
            layer.group_norm.weight.data = ref.weight.data[lo : lo + c].clone()
            if bias:
                layer.group_norm.bias.data = ref.bias.data[lo : lo + c].clone()

        layer.reshard_parameters(src, dst)

        c = CHANNELS // 4
        lo = dst["col"].rank() * c
        group = dst["model"]
        assert all_ranks_ok(layer.num_channels == c, device, group)
        assert all_ranks_ok(layer.num_local_groups == GROUPS // 4, device, group)
        if affine:
            assert_close(
                "weight",
                layer.group_norm.weight.data,
                ref.weight.data[lo : lo + c],
                0.0,
                0.0,
                device,
                group,
            )
            if bias:
                assert_close(
                    "bias",
                    layer.group_norm.bias.data,
                    ref.bias.data[lo : lo + c],
                    0.0,
                    0.0,
                    device,
                    group,
                )
            else:
                assert all_ranks_ok(layer.group_norm.bias is None, device, group)
        else:
            assert all_ranks_ok(layer.group_norm.weight is None, device, group)
