"""``extra_repr`` of the distributed layers.

Each layer shows the fields of its ``torch.nn`` (or non-distributed jigsaw)
counterpart, in the same format, followed by the sizes of the process groups it
is sharded over.
"""

import pytest
import torch
import torch.distributed as dist

from jigsaw.modules import (
    Attention,
    DistributedAttention,
    DistributedConv2d,
    DistributedConvTranspose2d,
    DistributedCrossEntropyLoss,
    DistributedGroupNorm,
    DistributedL1Loss,
    DistributedLayerNorm,
    DistributedLinear,
    DistributedMatMulXWT,
    DistributedMSELoss,
    DistributedNLLLoss,
    DistributedRoll,
    DistributedSegmentationMetrics,
    DistributedSoftmax,
    Roll,
)
from tests.parity import all_ranks_ok
from tests.process_mesh import _get_mesh

meshes = pytest.mark.parametrize(
    "mesh",
    [
        dict(row=1, col=1),
        dict(row=2, col=2),
        dict(row=1, col=4),
        dict(row=4, col=2),
    ],
    indirect=True,
)


def _with_bias(torch_repr: str, bias: bool) -> str:
    """Add the ``bias`` field that torch < 2.12 leaves out of the norm reprs."""
    return torch_repr if "bias=" in torch_repr else f"{torch_repr}, bias={bias}"


def _check(module: torch.nn.Module, expected: str, device: torch.device) -> None:
    got = module.extra_repr()
    assert all_ranks_ok(got == expected, device, dist.group.WORLD), (
        f"{got!r} != {expected!r}"
    )


@meshes
def test_linear(mesh: dict, device: torch.device) -> None:
    layer = DistributedLinear(
        16,
        32,
        local_rank=mesh["model"].rank(),
        device=device,
        model_group=mesh["model"],
        row_group=mesh["row"],
        col_group=mesh["col"],
        bias=False,
    )
    torch_repr = torch.nn.Linear(16, 32, bias=False).extra_repr()
    row, col = mesh["row"].size(), mesh["col"].size()
    _check(layer, f"{torch_repr}, row={row}, col={col}", device)


@meshes
def test_linear_row_is_the_weight_split(mesh: dict, device: torch.device) -> None:
    """With ``model_group == col_group`` the weight is replicated over rows."""
    layer = DistributedLinear(
        16,
        32,
        local_rank=mesh["col"].rank(),
        device=device,
        model_group=mesh["col"],
        row_group=mesh["row"],
        col_group=mesh["col"],
    )
    torch_repr = torch.nn.Linear(16, 32).extra_repr()
    _check(layer, f"{torch_repr}, row=1, col={mesh['col'].size()}", device)


@meshes
def test_matmul(mesh: dict, device: torch.device) -> None:
    module = DistributedMatMulXWT(mesh["model"].rank(), device, mesh["model"])
    _check(module, f"model={mesh['model'].size()}", device)


@meshes
def test_conv(mesh: dict, device: torch.device) -> None:
    layer = DistributedConv2d(
        4,
        8,
        3,
        device=device,
        spatial_group=mesh["row"],
        channel_group=mesh["col"],
        padding=(1,),
        bias=False,
    )
    torch_repr = torch.nn.Conv2d(4, 8, 3, padding=1, bias=False).extra_repr()
    row, col = mesh["row"].size(), mesh["col"].size()
    _check(layer, f"{torch_repr}, spatial=(1, {row}), channel={col}", device)


@meshes
def test_conv_transpose(mesh: dict, device: torch.device) -> None:
    layer = DistributedConvTranspose2d(
        4,
        8,
        2,
        device=device,
        spatial_group=[mesh["row"], None],
        channel_group=mesh["col"],
        stride=2,
    )
    torch_repr = torch.nn.ConvTranspose2d(4, 8, 2, stride=2).extra_repr()
    row, col = mesh["row"].size(), mesh["col"].size()
    _check(layer, f"{torch_repr}, spatial=({row}, 1), channel={col}", device)


@meshes
def test_layer_norm(mesh: dict, device: torch.device) -> None:
    layer = DistributedLayerNorm(
        16,
        local_rank=mesh["model"].rank(),
        device=device,
        row_group=mesh["row"],
        col_group=mesh["col"],
        eps=1e-6,
    )
    torch_repr = _with_bias(torch.nn.LayerNorm(16, eps=1e-6).extra_repr(), True)
    row, col = mesh["row"].size(), mesh["col"].size()
    _check(layer, f"{torch_repr}, row={row}, col={col}", device)


@meshes
def test_group_norm(mesh: dict, device: torch.device) -> None:
    layer = DistributedGroupNorm(
        4,
        16,
        device=device,
        channel_group=mesh["col"],
        spatial_group=[mesh["row"], mesh["row"]],
        affine=False,
    )
    torch_repr = _with_bias(torch.nn.GroupNorm(4, 16, affine=False).extra_repr(), False)
    row, col = mesh["row"].size(), mesh["col"].size()
    _check(layer, f"{torch_repr}, spatial={row * row}, channel={col}", device)


@meshes
def test_softmax(mesh: dict, device: torch.device) -> None:
    layer = DistributedSoftmax(mesh["col"], dim=1)
    torch_repr = torch.nn.Softmax(dim=1).extra_repr()
    _check(layer, f"{torch_repr}, group={mesh['col'].size()}", device)


@meshes
def test_roll(mesh: dict, device: torch.device) -> None:
    layer = DistributedRoll((1, 2), (1, 2), (2,), (mesh["col"],), device)
    roll_repr = Roll((1, 2), (1, 2)).extra_repr()
    expected = f"{roll_repr}, distributed_dims=(2,), spatial=({mesh['col'].size()},)"
    _check(layer, expected, device)


@meshes
def test_attention(mesh: dict, device: torch.device) -> None:
    layer = DistributedAttention(
        16,
        16,
        8,
        heads=4,
        local_rank=mesh["model"].rank(),
        device=device,
        spatial_group=mesh["row"],
        channel_group=mesh["col"],
        qk_norm=True,
    )
    attn_repr = Attention(16, 16, 8, 4, device, qk_norm=True).extra_repr()
    row, col = mesh["row"].size(), mesh["col"].size()
    _check(layer, f"{attn_repr}, ring=False, spatial={row}, channel={col}", device)


@meshes
def test_losses_and_metrics(mesh: dict, device: torch.device) -> None:
    row, col = mesh["row"].size(), mesh["col"].size()
    _check(DistributedL1Loss(), "reduce_groups=1", device)
    _check(DistributedNLLLoss(), "channel=1, reduce_groups=1", device)
    _check(
        DistributedMSELoss((mesh["row"], mesh["col"])),
        f"reduce_groups=({row}, {col})",
        device,
    )
    _check(
        DistributedCrossEntropyLoss(mesh["col"], mesh["row"]),
        f"channel={col}, reduce_groups={row}",
        device,
    )
    _check(
        DistributedSegmentationMetrics(3, mesh["col"], [mesh["row"], None]),
        f"channel={col}, reduce_groups=({row}, 1)",
        device,
    )


@pytest.mark.mesh(row=2, col=2)
def test_reshard_updates_the_shard_sizes(mesh: dict, device: torch.device) -> None:
    to_mesh = _get_mesh(device.type, dict(row=1, col=4))
    layer = DistributedLinear(
        16,
        32,
        local_rank=mesh["model"].rank(),
        device=device,
        model_group=mesh["model"],
        row_group=mesh["row"],
        col_group=mesh["col"],
    )
    layer.reshard_parameters(mesh, to_mesh)
    torch_repr = torch.nn.Linear(16, 32).extra_repr()
    _check(layer, f"{torch_repr}, row=1, col=4", device)
