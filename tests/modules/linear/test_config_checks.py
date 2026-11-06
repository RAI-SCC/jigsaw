"""Config checks of the linear layer and the matmul modules, on every mesh size.

Every check raises before any communication, so the same ``ValueError`` reaches
every rank and nothing is left in flight. The matmul parity test runs here too,
since ``test_matmul.py`` covers only the 2x2 mesh.
"""

import re
from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist

from jigsaw.modules.linear import (
    DistributedLinear,
    DistributedMatMulXTW,
    DistributedMatMulXW,
    DistributedMatMulXWT,
    LinearConfig,
)
from tests.parity import (
    ShardSpec,
    all_ranks_ok,
    assert_close,
    gather_tensor,
    split_tensor,
)

meshes = pytest.mark.parametrize(
    "mesh",
    [
        dict(row=1, col=1),
        dict(row=1, col=2),
        dict(row=2, col=2),
        dict(row=4, col=2),
    ],
    indirect=True,
)

#: Both matmul operands and the output are split rows over row, columns over col.
_BLOCKS = ShardSpec(row=0, col=1)
_M, _K, _N = 8, 16, 8
#: module, global operand shapes, reference op.
_MATMULS = {
    "xwt": (DistributedMatMulXWT, ((_M, _K), (_N, _K)), lambda x, w: x @ w.mT),
    "xw": (DistributedMatMulXW, ((_M, _K), (_K, _N)), lambda x, w: x @ w),
    "xtw": (DistributedMatMulXTW, ((_K, _M), (_K, _N)), lambda x, w: x.mT @ w),
}
_MATMUL_CLASSES = [op[0] for op in _MATMULS.values()]


class _Group:
    """Stand-in for a process group; ``LinearConfig`` only reads its size."""

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


def _raises(fn: Callable[[], object], match: str) -> bool:
    """Return whether ``fn`` raises a ``ValueError`` whose message matches."""
    try:
        fn()
    except ValueError as e:
        return re.search(match, str(e)) is not None
    return False


def _linear(mesh: dict, device: torch.device, **kwargs: object) -> DistributedLinear:
    return DistributedLinear(
        8,
        8,
        device=device,
        model_group=mesh["model"],
        row_group=mesh["row"],
        col_group=mesh["col"],
        **kwargs,  # type: ignore[arg-type]
    )


def _other_size(mesh: dict) -> int:
    """Return a supported model group size that differs from the mesh's."""
    return 2 if mesh["model"].size() != 2 else 4


@meshes
@pytest.mark.parametrize("op", _MATMULS.values(), ids=_MATMULS.keys())
def test_matmul_block_split_matches_torch(
    mesh: dict, device: torch.device, op: tuple
) -> None:
    """Both operands block-split over row and col, on every supported size."""
    module_cls, (x_shape, w_shape), reference = op
    torch.manual_seed(0)  # same global operands on every rank
    x = torch.randn(x_shape, device=device)
    w = torch.randn(w_shape, device=device)
    module = module_cls(device=device, model_group=mesh["model"])
    y = module(split_tensor(x, _BLOCKS, mesh), split_tensor(w, _BLOCKS, mesh))
    assert_close(
        "output",
        gather_tensor(y, _BLOCKS, mesh, device, "output"),
        reference(x, w),
        1e-5,
        1e-5,
        device,
        dist.group.WORLD,
    )


@meshes
def test_linear_rejects_config_for_another_size(
    mesh: dict, device: torch.device
) -> None:
    config = LinearConfig(_Group(_other_size(mesh)), device=device)
    assert all_ranks_ok(
        _raises(lambda: _linear(mesh, device, config=config), "model group size"),
        device,
        dist.group.WORLD,
    )


@meshes
def test_linear_rejects_matmul_config(mesh: dict, device: torch.device) -> None:
    config = LinearConfig(mesh["model"], linear_layer=False, device=device)
    assert all_ranks_ok(
        _raises(lambda: _linear(mesh, device, config=config), "linear_layer"),
        device,
        dist.group.WORLD,
    )


@meshes
@pytest.mark.parametrize("module_cls", _MATMUL_CLASSES, ids=_MATMULS.keys())
@pytest.mark.parametrize("wrong", ["size", "linear_layer"])
def test_matmul_rejects_mismatched_config(
    mesh: dict, device: torch.device, module_cls: type, wrong: str
) -> None:
    config = (
        LinearConfig(_Group(_other_size(mesh)), linear_layer=False, device=device)
        if wrong == "size"
        else LinearConfig(mesh["model"], linear_layer=True, device=device)
    )
    match = "model group size" if wrong == "size" else "linear_layer"
    assert all_ranks_ok(
        _raises(
            lambda: module_cls(device=device, model_group=mesh["model"], config=config),
            match,
        ),
        device,
        dist.group.WORLD,
    )


@meshes
def test_reshard_to_unsupported_size_leaves_layer_unchanged(
    mesh: dict, device: torch.device
) -> None:
    """The target config is built before ``merge_weights`` communicates."""
    layer = _linear(mesh, device)
    weight, config = layer.weight, layer.config
    target = {"row": _Group(1), "col": _Group(3), "model": _Group(3)}
    raised = _raises(
        lambda: layer.reshard_parameters(mesh, target), "Unsupported model group size"
    )
    unchanged = layer.weight is weight and layer.config is config
    assert all_ranks_ok(raised and unchanged, device, dist.group.WORLD)
