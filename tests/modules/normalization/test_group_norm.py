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
    # Same parameter names as torch: ``weight`` / ``bias`` (``None`` when absent).
    assert (layer.weight is None) == (not affine)
    assert (layer.bias is None) == (not (affine and bias))
    layer.load_state_dict(ref.state_dict())
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
            GROUPS,
            CHANNELS,
            affine=affine,
            bias=bias,
            device=device,
            channel_group=src["col"],
        )
        ref = _reference(affine, bias, device)
        if affine:
            c = CHANNELS // 2
            lo = src["col"].rank() * c
            layer.weight.data = ref.weight.data[lo : lo + c].clone()
            if bias:
                layer.bias.data = ref.bias.data[lo : lo + c].clone()

        layer.reshard_parameters(src, dst)

        c = CHANNELS // 4
        lo = dst["col"].rank() * c
        group = dst["model"]
        assert all_ranks_ok(layer.num_local_channels == c, device, group)
        assert all_ranks_ok(layer.num_local_groups == GROUPS // 4, device, group)
        if affine:
            assert_close(
                "weight",
                layer.weight.data,
                ref.weight.data[lo : lo + c],
                0.0,
                0.0,
                device,
                group,
            )
            if bias:
                assert_close(
                    "bias",
                    layer.bias.data,
                    ref.bias.data[lo : lo + c],
                    0.0,
                    0.0,
                    device,
                    group,
                )
            else:
                assert all_ranks_ok(layer.bias is None, device, group)
        else:
            assert all_ranks_ok(layer.weight is None, device, group)


# Mesh fixture -> (key of the channel group, keys of the spatial groups split
# along H and W; ``None`` for an axis that is not split).
_SPATIAL_MESHES = {
    "process_mesh_dict_2x2": ("col", (None, "row")),
    "spatial_mesh_2x2_hw": ("col", ("sh", "sw")),
}


@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_name", list(_SPATIAL_MESHES))
@pytest.mark.parametrize("affine, bias", [(True, True), (True, False), (False, True)])
def test_spatially_sharded_matches_torch(request, device, mesh_name, affine, bias):
    """Each rank holds a block of H x W: the statistics span every block."""
    mesh = request.getfixturevalue(mesh_name)
    col_key, spatial_keys = _SPATIAL_MESHES[mesh_name]
    spatial_groups = [mesh[k] if k is not None else None for k in spatial_keys]
    channel_group = mesh[col_key]

    ref = _reference(affine, bias, device)
    layer = norm.DistributedGroupNorm(
        GROUPS,
        CHANNELS,
        affine=affine,
        bias=bias,
        device=device,
        channel_group=channel_group,
        spatial_group=spatial_groups,
    )
    c = CHANNELS // channel_group.size()
    channels = slice(channel_group.rank() * c, (channel_group.rank() + 1) * c)
    if affine:
        layer.weight.data = ref.weight.data[channels].clone()
        if bias:
            layer.bias.data = ref.bias.data[channels].clone()

    torch.manual_seed(0)
    x = torch.randn(2, CHANNELS, 4, 6, device=device, requires_grad=True)
    grad_out = torch.randn(2, CHANNELS, 4, 6, device=device)
    y_ref = ref(x)
    y_ref.backward(grad_out)

    index = [slice(None), channels]
    for dim, g in zip((2, 3), spatial_groups):
        n = x.shape[dim] // (g.size() if g is not None else 1)
        r = g.rank() if g is not None else 0
        index.append(slice(r * n, (r + 1) * n))
    index = tuple(index)
    x_local = x.detach()[index].clone().requires_grad_(True)
    y = layer(x_local)
    y.backward(grad_out[index])

    group = mesh["model"]
    assert_close("forward", y, y_ref[index], 1e-5, 1e-5, device, group)
    assert_close("input grad", x_local.grad, x.grad[index], 1e-5, 1e-5, device, group)
    if affine:
        assert_close(
            "weight grad",
            layer.weight.grad,
            ref.weight.grad[channels],
            1e-5,
            1e-5,
            device,
            group,
        )
        if bias:
            assert_close(
                "bias grad",
                layer.bias.grad,
                ref.bias.grad[channels],
                1e-5,
                1e-5,
                device,
                group,
            )
