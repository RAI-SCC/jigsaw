"""Distributed transpose kernels against a serial permute operation."""

from typing import Callable

import pytest
import torch

from jigsaw._core.buffers import clear_preallocated_buffers
from jigsaw.nn.permute._distributed_permute import (
    distributed_permute,
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


def _shard_dims(
    t: torch.Tensor, n_rows: int, n_cols: int, rank: int, row_dim: int, col_dim: int
) -> torch.Tensor:
    """Return the block of ``row_dim``/``col_dim`` that ``rank`` holds."""
    row, col = divmod(rank, n_cols)
    h, w = t.shape[row_dim] // n_rows, t.shape[col_dim] // n_cols
    index: list[slice] = [slice(None)] * t.ndim
    index[row_dim] = slice(row * h, (row + 1) * h)
    index[col_dim] = slice(col * w, (col + 1) * w)
    return t[tuple(index)].contiguous()


def _shard(t: torch.Tensor, n_rows: int, n_cols: int, rank: int) -> torch.Tensor:
    """Return the block of the last two dimensions that ``rank`` holds."""
    return _shard_dims(t, n_rows, n_cols, rank, -2, -1)


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
    """Start every test without cached communication buffers."""
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


def _check_permute(
    mesh: dict,
    device: torch.device,
    shape: tuple[int, ...],
    dims: tuple[int, ...],
    row_dim: int,
    col_dim: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> torch.Tensor:
    """Run ``distributed_permute`` on this rank's shard and compare with a serial permute."""
    group = mesh["model"]
    n_rows, n_cols = mesh["row"].size(), mesh["col"].size()
    x_global = _global(shape, dtype, device, seed)
    x = _shard_dims(x_global, n_rows, n_cols, group.rank(), row_dim, col_dim)
    # mesh["row"]/mesh["col"] shard the *fixed* positions row_dim/col_dim of both the
    # input and the output -- not wherever the original row_dim/col_dim axis's data
    # ends up (that's `dims.index(row_dim)`, which only coincides with row_dim itself
    # when `dims` doesn't swap the tracked pair).
    expected = _shard_dims(
        x_global.permute(*dims), n_rows, n_cols, group.rank(), row_dim, col_dim
    )

    error = None
    try:
        got = distributed_permute(x, dims, row_dim, col_dim, group.rank(), device, mesh)
    except Exception as exc:  # noqa: BLE001
        error = exc
    if not all_ranks_ok(error is None, device, group):
        raise AssertionError(f"distributed_permute raised on some rank: {error!r}")

    assert_close("distributed_permute", got, expected, 0.0, 0.0, device, group)
    return got


@pytest.mark.dist(2)
class TestDistributedPermuteTrailingTwoWay:
    """row_dim/col_dim at the trailing two positions on a 1x2 mesh: the fast path."""

    def test_identity(self, device: torch.device, process_mesh_dict_1x2: dict) -> None:
        _check_permute(process_mesh_dict_1x2, device, (4, 6), (0, 1), 0, 1)

    def test_swap_with_batch_reorder(
        self, device: torch.device, process_mesh_dict_1x2: dict
    ) -> None:
        # (E, B, S, C) -> (B, E, C, S): batch axes reordered, trailing two swapped.
        _check_permute(process_mesh_dict_1x2, device, (2, 3, 4, 6), (1, 0, 3, 2), 2, 3)


@pytest.mark.dist(4)
class TestDistributedPermuteTrailingFourWay:
    """row_dim/col_dim at the trailing two positions on a 2x2 mesh: the fast path."""

    def test_swap_with_batch_reorder(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        _check_permute(process_mesh_dict_2x2, device, (2, 3, 4, 6), (1, 0, 3, 2), 2, 3)


@pytest.mark.dist(8)
class TestDistributedPermuteTrailingEightWay:
    """row_dim/col_dim at the trailing two positions on a 4x2 mesh: the fast path."""

    def test_swap_with_batch_reorder(
        self, device: torch.device, process_mesh_dict_4x2: dict
    ) -> None:
        _check_permute(process_mesh_dict_4x2, device, (2, 3, 8, 12), (1, 0, 3, 2), 2, 3)


@pytest.mark.dist(4)
class TestDistributedPermuteGeneral:
    """row_dim/col_dim *not* trailing: the all-gather based general path.

    Shape is ``(E, S, H, C)`` with ``S`` (dim 1) row-sharded and ``C`` (dim 3)
    col-sharded, ``H`` (dim 2) sitting between them untouched -- the shape of
    problem this module couldn't express before: the sharded axes aren't the
    trailing two, and the mesh shape (2x2) isn't tied to a specific kernel.
    """

    SHAPE = (2, 4, 3, 6)
    ROW_DIM, COL_DIM = 1, 3

    def test_other_axes_reordered_without_resharding(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """No swap of the tracked pair: a pure, comm-free local permute."""
        dims = (2, 1, 0, 3)
        _check_permute(
            process_mesh_dict_2x2, device, self.SHAPE, dims, self.ROW_DIM, self.COL_DIM
        )

    def test_matches_serial_permute_swap(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        dims = (0, 3, 2, 1)
        _check_permute(
            process_mesh_dict_2x2, device, self.SHAPE, dims, self.ROW_DIM, self.COL_DIM
        )

    def test_swap_repositions_a_second_batch_axis_too(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """A swap of the tracked pair also moves whatever `dims` does to the rest.

        Axis 0 and axis 2 aren't row_dim/col_dim, so `distributed_permute` never
        reads or reshards them directly here -- but `_check_permute`'s reference
        is computed from the same `x_global` every rank builds identically, which
        is the one invariant this whole module relies on: an axis that isn't
        row_dim/col_dim must be uniform across mesh["row"] x mesh["col"] (it may
        still be sharded by some other, orthogonal mesh axis held fixed for this
        call -- just not by row/col themselves).
        """
        dims = (2, 3, 0, 1)
        _check_permute(
            process_mesh_dict_2x2, device, self.SHAPE, dims, self.ROW_DIM, self.COL_DIM
        )

    def test_invalid_dims_raises(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Moving an untracked axis into the row/col slot is not supported."""
        mesh = process_mesh_dict_2x2
        group = mesh["model"]
        x = torch.zeros(self.SHAPE, device=device)
        with pytest.raises(ValueError):
            distributed_permute(
                x, (3, 1, 2, 0), self.ROW_DIM, self.COL_DIM, group.rank(), device, mesh
            )
