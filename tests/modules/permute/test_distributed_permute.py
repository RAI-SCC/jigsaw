"""Distributed transpose kernels against a serial ``.mT``.

Each kernel bakes in one ``row x col`` arrangement of the model ranks, the same
ones the XWT kernels use: ``x_transpose_twoway`` a 1x2 in-split,
``x_transpose_fourway`` a 2x2 mesh and ``x_transpose_eightway`` a 4x2 mesh, with
``rank = row * n_cols + col``. A rank holds block ``(row, col)`` of the last two
dimensions of the global matrix and gets back block ``(row, col)`` of its
transpose on the same mesh.

Every rank builds the same global matrix from a seed, so the reference shard is
sliced locally and no gather is needed.

The kernels exchange unequal blocks with ``dist.all_to_all``, which gloo does
not support, so the tests are skipped on gloo.
"""

from typing import Callable

import pytest
import torch
import torch.distributed as dist

from jigsaw.modules.buffers import clear_preallocated_buffers
from jigsaw.modules.permute.distributed_permute import (
    x_transpose_eightway,
    x_transpose_fourway,
    x_transpose_twoway,
)
from tests.parity import all_ranks_ok, assert_close

pytestmark = [pytest.mark.dist_required]

BATCH_SHAPES = [(), (3,), (2, 3)]
DTYPES = [torch.float32, torch.float64]


def _global(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, seed: int = 0
) -> torch.Tensor:
    """Return the same random global matrix on every rank."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=dtype, generator=g).to(device)


def _shard(t: torch.Tensor, n_rows: int, n_cols: int, rank: int) -> torch.Tensor:
    """Return the block of the last two dimensions that ``rank`` holds."""
    row, col = divmod(rank, n_cols)
    h, w = t.shape[-2] // n_rows, t.shape[-1] // n_cols
    return t[..., row * h : (row + 1) * h, col * w : (col + 1) * w].contiguous()


def _check_transpose(
    kernel: Callable[..., torch.Tensor],
    mesh: dict,
    device: torch.device,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> torch.Tensor:
    """Run ``kernel`` on this rank's shard and compare with the serial ``.mT``."""
    group = mesh["model"]
    n_rows, n_cols = mesh["row"].size(), mesh["col"].size()
    x_global = _global(shape, dtype, device, seed)
    x = _shard(x_global, n_rows, n_cols, group.rank())
    expected = _shard(x_global.mT, n_rows, n_cols, group.rank())

    # A kernel that raises on one rank only would leave the others waiting in
    # the all-reduce of ``assert_close``, so agree on the outcome first.
    error = None
    try:
        got = kernel(x, group.rank(), device, group)
    except Exception as exc:  # noqa: BLE001
        error = exc
    if not all_ranks_ok(error is None, device, group):
        raise AssertionError(f"{kernel.__name__} raised on some rank: {error!r}")

    dtype_ok = got.dtype == dtype and got.device.type == device.type
    if not all_ranks_ok(dtype_ok, device, group):
        raise AssertionError(
            f"{kernel.__name__} returned {got.dtype} on {got.device}, "
            f"expected {dtype} on {device}"
        )
    assert_close(kernel.__name__, got, expected, 0.0, 0.0, device, group)
    return got


@pytest.fixture(autouse=True)
def _fresh_buffers():
    """Skip on gloo and start every test without cached communication buffers."""
    if dist.get_backend() == "gloo":
        pytest.skip("the transpose kernels need a non-gloo backend (all_to_all)")
    clear_preallocated_buffers()
    yield
    clear_preallocated_buffers()


@pytest.mark.dist(2)
class TestTransposeTwoWay:
    """1x2 mesh: the ranks split the last dimension."""

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_matches_serial_transpose(
        self,
        device: torch.device,
        process_mesh_dict_1x2: dict,
        batch: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        _check_transpose(
            x_transpose_twoway, process_mesh_dict_1x2, device, (*batch, 4, 6), dtype
        )

    @pytest.mark.parametrize("shape", [(4, 4), (2, 8), (8, 2), (3, 6, 10)])
    def test_rectangular_shards(
        self, device: torch.device, process_mesh_dict_1x2: dict, shape: tuple[int, ...]
    ) -> None:
        _check_transpose(x_transpose_twoway, process_mesh_dict_1x2, device, shape)

    def test_transposing_twice_is_the_identity(
        self, device: torch.device, process_mesh_dict_1x2: dict
    ) -> None:
        mesh = process_mesh_dict_1x2
        group = mesh["model"]
        x_global = _global((2, 4, 6), torch.float32, device)
        x = _shard(x_global, 1, 2, group.rank())
        xt = x_transpose_twoway(x, group.rank(), device, group)
        back = x_transpose_twoway(xt, group.rank(), device, group)
        assert_close("twoway round trip", back, x, 0.0, 0.0, device, group)

    def test_repeated_calls_reuse_the_buffers(
        self, device: torch.device, process_mesh_dict_1x2: dict
    ) -> None:
        """A later, larger call must not see the data of an earlier one."""
        for seed, shape in enumerate([(4, 6), (2, 8, 6), (4, 6)]):
            _check_transpose(
                x_transpose_twoway, process_mesh_dict_1x2, device, shape, seed=seed
            )


@pytest.mark.dist(4)
class TestTransposeFourWay:
    """2x2 mesh: the diagonal ranks transpose locally, the others swap."""

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_matches_serial_transpose(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        batch: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        _check_transpose(
            x_transpose_fourway, process_mesh_dict_2x2, device, (*batch, 4, 6), dtype
        )

    @pytest.mark.parametrize("shape", [(4, 4), (2, 8), (8, 2), (3, 6, 10)])
    def test_rectangular_shards(
        self, device: torch.device, process_mesh_dict_2x2: dict, shape: tuple[int, ...]
    ) -> None:
        _check_transpose(x_transpose_fourway, process_mesh_dict_2x2, device, shape)

    def test_transposing_twice_is_the_identity(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        mesh = process_mesh_dict_2x2
        group = mesh["model"]
        x_global = _global((2, 4, 6), torch.float32, device)
        x = _shard(x_global, 2, 2, group.rank())
        xt = x_transpose_fourway(x, group.rank(), device, group)
        back = x_transpose_fourway(xt, group.rank(), device, group)
        assert_close("fourway round trip", back, x, 0.0, 0.0, device, group)

    def test_repeated_calls_reuse_the_buffers(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """A later, larger call must not see the data of an earlier one."""
        for seed, shape in enumerate([(4, 6), (2, 8, 6), (4, 6)]):
            _check_transpose(
                x_transpose_fourway, process_mesh_dict_2x2, device, shape, seed=seed
            )


@pytest.mark.dist(8)
class TestTransposeEightWay:
    """4x2 mesh: every rank exchanges the two halves of its last dimension."""

    @pytest.mark.parametrize("batch", BATCH_SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_matches_serial_transpose(
        self,
        device: torch.device,
        process_mesh_dict_4x2: dict,
        batch: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        _check_transpose(
            x_transpose_eightway, process_mesh_dict_4x2, device, (*batch, 8, 12), dtype
        )

    @pytest.mark.parametrize("shape", [(8, 8), (4, 16), (16, 4), (3, 12, 20)])
    def test_rectangular_shards(
        self, device: torch.device, process_mesh_dict_4x2: dict, shape: tuple[int, ...]
    ) -> None:
        _check_transpose(x_transpose_eightway, process_mesh_dict_4x2, device, shape)

    def test_transposing_twice_is_the_identity(
        self, device: torch.device, process_mesh_dict_4x2: dict
    ) -> None:
        mesh = process_mesh_dict_4x2
        group = mesh["model"]
        x_global = _global((2, 8, 8), torch.float32, device)
        x = _shard(x_global, 4, 2, group.rank())
        xt = x_transpose_eightway(x, group.rank(), device, group)
        back = x_transpose_eightway(xt, group.rank(), device, group)
        assert_close("eightway round trip", back, x, 0.0, 0.0, device, group)

    def test_repeated_calls_reuse_the_buffers(
        self, device: torch.device, process_mesh_dict_4x2: dict
    ) -> None:
        """A later, larger call must not see the data of an earlier one."""
        for seed, shape in enumerate([(8, 12), (2, 16, 12), (8, 12)]):
            _check_transpose(
                x_transpose_eightway, process_mesh_dict_4x2, device, shape, seed=seed
            )
