"""Whole-model parameter count of distributed modules against their torch reference.

Two counts are checked against the reference module's ``numel``:

* ``get_parameter_count`` -- each layer *computes* its global count from its global
  shapes, with no communication (summed over a model by
  :func:`tests.parity.global_parameter_count`).
* ``debug.count_parameters`` -- counts the actual shards over the mesh.

``debug.count_parameters(module, mesh)`` sums shards over the model group and
counts a :func:`jigsaw.parallel.mark_replicated` parameter once per replica
group. So it equals the reference module's count only if every replicated
parameter is marked and no sharded one is: a missing mark overcounts by the
replica-group size (that parameter's gradient would also never be synced), a
spurious one undercounts.
"""

from typing import Callable

import pytest
import torch

import jigsaw.modules.attention as attention
import jigsaw.modules.convolution as convolution
import jigsaw.modules.linear as linear
import jigsaw.modules.normalization as normalization
from jigsaw import debug
from tests.parity import all_ranks_ok, assert_close, global_parameter_count

Build = Callable[[dict, torch.device], tuple[torch.nn.Module, torch.nn.Module]]


def _layer_norm(pm: dict, device: torch.device) -> tuple:
    dist_mod = normalization.DistributedLayerNorm(
        8,
        local_rank=pm["model"].rank(),
        device=device,
        row_group=pm["row"],
        col_group=pm["col"],
    )
    return dist_mod, torch.nn.LayerNorm(8)


def _group_norm(pm: dict, device: torch.device) -> tuple:
    dist_mod = normalization.DistributedGroupNorm(
        num_groups=4,
        num_channels=8,
        device=device,
        channel_group=pm["col"],
        spatial_group=pm["row"],
    )
    return dist_mod, torch.nn.GroupNorm(4, 8)


def _conv(groups: int, bias: bool) -> Build:
    def build(pm: dict, device: torch.device) -> tuple:
        kw = dict(kernel_size=3, groups=groups, bias=bias)
        dist_mod = convolution.DistributedConv2d(
            in_channels=4,
            out_channels=8,
            device=device,
            spatial_group=pm["row"],
            channel_group=pm["col"],
            **kw,
        )
        return dist_mod, torch.nn.Conv2d(4, 8, **kw)

    return build


def _conv_transpose(bias: bool) -> Build:
    def build(pm: dict, device: torch.device) -> tuple:
        dist_mod = convolution.DistributedConvTranspose2d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            device=device,
            spatial_group=pm["row"],
            channel_group=pm["col"],
            bias=bias,
        )
        return dist_mod, torch.nn.ConvTranspose2d(4, 8, kernel_size=3, bias=bias)

    return build


class _RefAttention(torch.nn.Module):
    """Parameters of a non-distributed ``DistributedAttention``."""

    def __init__(self, dim: int, heads: int, qk_norm: bool) -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.head_concat = torch.nn.Linear(dim, dim)
        if qk_norm:
            self.q_norm = torch.nn.LayerNorm(dim // heads)
            self.k_norm = torch.nn.LayerNorm(dim // heads)


def _attention(qk_norm: bool, ring: bool = False) -> Build:
    def build(pm: dict, device: torch.device) -> tuple:
        dist_mod = attention.DistributedAttention(
            8,
            8,
            8,
            heads=2,
            local_rank=pm["col"].rank(),
            device=device,
            spatial_group=pm["row"],
            channel_group=pm["col"],
            qk_norm=qk_norm,
            ring=ring,
        )
        return dist_mod, _RefAttention(8, 2, qk_norm)

    return build


def _mlp(pm: dict, device: torch.device) -> tuple:
    """Two linears with weights split over row *and* col, as in the ViT example."""

    def dist_linear(in_dim: int, out_dim: int) -> torch.nn.Module:
        return linear.DistributedLinear(
            in_dim,
            out_dim,
            local_rank=pm["model"].rank(),
            device=device,
            model_group=pm["model"],
            row_group=pm["row"],
            col_group=pm["col"],
            bias=True,
        )

    dist_mod = torch.nn.Sequential(
        dist_linear(8, 16), torch.nn.GELU(), dist_linear(16, 8)
    )
    ref = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8)
    )
    return dist_mod, ref


def _assert_count_matches(build: Build, pm: dict, device: torch.device) -> None:
    dist_mod, ref = build(pm, device)
    expected = sum(p.numel() for p in ref.parameters())
    total = debug.count_parameters(dist_mod, pm)
    assert_close(
        f"{type(dist_mod).__name__} parameter count",
        torch.tensor(float(total), device=device),
        torch.tensor(float(expected), device=device),
        atol=0.0,
        rtol=0.0,
        device=device,
        group=pm["model"],
    )


@pytest.mark.dist(4)
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_layer_norm, id="layer_norm"),
        pytest.param(_group_norm, id="group_norm"),
        pytest.param(_conv(1, True), id="conv"),
        pytest.param(_conv(1, False), id="conv-nobias"),
        pytest.param(_conv(2, True), id="conv-groups2"),
        pytest.param(_conv(4, True), id="conv-groups4"),
        pytest.param(_conv_transpose(True), id="conv_transpose"),
        pytest.param(_conv_transpose(False), id="conv_transpose-nobias"),
        pytest.param(_attention(False), id="attention"),
        pytest.param(_attention(True), id="attention-qk_norm"),
        pytest.param(_attention(True, ring=True), id="attention-ring"),
        pytest.param(_mlp, id="mlp"),
    ],
)
def test_count_matches_reference_2x2(
    build: Build, device: torch.device, process_mesh_dict_2x2: dict
) -> None:
    """Row-replicated, column-sharded: every replica marked once over ``row``."""
    _assert_count_matches(build, process_mesh_dict_2x2, device)


@pytest.mark.dist(4)
def test_group_norm_count_matches_reference(
    device: torch.device, process_mesh_dict_1x4: dict
) -> None:
    """GroupNorm takes only a channel group, so it is checked on a channel-only mesh."""

    def build(pm: dict, device: torch.device) -> tuple:
        dist_mod = normalization.DistributedGroupNorm(
            num_groups=4, num_channels=8, device=device, channel_group=pm["col"]
        )
        return dist_mod, torch.nn.GroupNorm(4, 8)

    _assert_count_matches(build, process_mesh_dict_1x4, device)


@pytest.mark.dist(4)
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_conv(1, True), id="conv"),
        pytest.param(_conv(1, False), id="conv-nobias"),
        pytest.param(_attention(False), id="attention"),
        pytest.param(_attention(True), id="attention-qk_norm"),
        pytest.param(_attention(True, ring=True), id="attention-ring"),
        pytest.param(_mlp, id="mlp"),
    ],
)
def test_get_parameter_count_matches_reference_2x2(
    build: Build, device: torch.device, process_mesh_dict_2x2: dict
) -> None:
    """The computed global count is the reference count on every rank."""
    dist_mod, ref = build(process_mesh_dict_2x2, device)
    expected = sum(p.numel() for p in ref.parameters())
    got = global_parameter_count(dist_mod)
    assert all_ranks_ok(got == expected, device, process_mesh_dict_2x2["model"]), (
        f"{type(dist_mod).__name__}: get_parameter_count gives {got}, "
        f"reference has {expected}"
    )
