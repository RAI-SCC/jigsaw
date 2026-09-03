"""Unit tests for ``get_preallocated_buffer`` buffer caching / keying.

These are plain (non-distributed) tests: the function under test only needs a
``torch.device`` and exercises the module-level ``_prealloc_buffers`` cache.
"""

import pytest
import torch

from jigsaw.modules.buffers import (
    _prealloc_buffers,
    clear_preallocated_buffers,
    get_preallocated_buffer,
)

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _clear_buffer_cache():
    """Isolate every test from the process-global buffer cache."""
    clear_preallocated_buffers()
    yield
    clear_preallocated_buffers()


# --- helpers -----------------------------------------------------------------
# Calls routed through a helper all share a single call site (one source line),
# which is exactly what the "reuse across iterations" behaviour keys on.


def _alloc(shape=(8,), dtype=torch.float32, local_rank=0, device=CPU, index=0):
    return get_preallocated_buffer(torch.Size(shape), dtype, local_rank, device, index)


# --- basic contract --------------------------------------------------------


class TestPreallocatedBuffer:
    def test_returns_requested_shape_dtype_device_and_is_zeroed(self):
        buf = get_preallocated_buffer(torch.Size([2, 3]), torch.float64, 0, CPU)
        assert buf.shape == (2, 3)
        assert buf.dtype == torch.float64
        assert buf.device == CPU
        assert not buf.any()

    def test_multidim_shape_produces_a_writable_view(self):
        buf = get_preallocated_buffer(torch.Size([2, 3, 4]), torch.float32, 0, CPU)
        assert buf.shape == (2, 3, 4)
        assert buf.numel() == 24
        buf[1, 2, 3] = 5.0
        assert buf[1, 2, 3].item() == 5.0

    # --- the fix: distinct call sites must not alias --------------------------

    def test_distinct_call_sites_get_distinct_storage_despite_identical_args(self):
        a = get_preallocated_buffer(torch.Size([4]), torch.float32, 0, CPU)
        b = get_preallocated_buffer(torch.Size([4]), torch.float32, 0, CPU)

        assert a.data_ptr() != b.data_ptr()

        a.fill_(1.0)
        b.fill_(2.0)
        assert torch.all(a == 1.0)
        assert torch.all(b == 2.0)

    def test_concurrent_recv_buffers_do_not_overwrite_each_other(self):
        # Mirrors the real usage: several same-shape/dtype/rank buffers requested
        # from different lines and kept live at once as irecv targets.
        b0 = get_preallocated_buffer(torch.Size([3, 3]), torch.float32, 1, CPU)
        b1 = get_preallocated_buffer(torch.Size([3, 3]), torch.float32, 1, CPU)
        b2 = get_preallocated_buffer(torch.Size([3, 3]), torch.float32, 1, CPU)
        b3 = get_preallocated_buffer(torch.Size([3, 3]), torch.float32, 1, CPU)

        bufs = [b0, b1, b2, b3]
        for i, b in enumerate(bufs):
            b.fill_(i + 1)
        for i, b in enumerate(bufs):
            assert torch.all(b == i + 1), f"buffer {i} was clobbered by another buffer"

        assert len({b.data_ptr() for b in bufs}) == 4

    # --- reuse across calls from the same call site -------------------------

    def test_same_call_site_reuses_the_same_allocation(self):
        first = _alloc()
        ptr = first.data_ptr()
        first.fill_(7.0)

        second = _alloc()
        assert second.data_ptr() == ptr  # same backing storage
        assert not second.any()  # but handed back zeroed

    def test_repeated_allocation_is_stable_and_does_not_leak(self):
        ptrs = set()
        for _ in range(50):
            b = _alloc()
            b.fill_(1.0)
            ptrs.add(b.data_ptr())
        assert len(ptrs) == 1
        assert len(_prealloc_buffers) == 1

    def test_cache_holds_one_entry_per_call_site(self):
        assert len(_prealloc_buffers) == 0
        _alloc()
        assert len(_prealloc_buffers) == 1
        _alloc()  # same call site -> no new entry
        assert len(_prealloc_buffers) == 1

    # --- key components -----------------------------------------------------

    def test_index_disambiguates_buffers_from_one_call_site(self):
        a = _alloc(index=0)
        b = _alloc(index=1)
        assert a.data_ptr() != b.data_ptr()
        assert len(_prealloc_buffers) == 2

    def test_dtype_is_part_of_the_key(self):
        a = _alloc(dtype=torch.float32)
        b = _alloc(dtype=torch.float64)
        assert a.dtype == torch.float32
        assert b.dtype == torch.float64
        assert a.data_ptr() != b.data_ptr()
        assert len(_prealloc_buffers) == 2

    def test_local_rank_is_part_of_the_key(self):
        a = _alloc(local_rank=0)
        b = _alloc(local_rank=1)
        assert a.data_ptr() != b.data_ptr()
        assert len(_prealloc_buffers) == 2

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs a second device to compare"
    )
    def test_device_is_part_of_the_key(self):
        a = _alloc(device=torch.device("cpu"))
        b = _alloc(device=torch.device("cuda", 0))
        assert a.device.type == "cpu"
        assert b.device.type == "cuda"
        assert len(_prealloc_buffers) == 2

    # --- grow-on-demand, shape deliberately not in the key -----------------

    def test_buffer_grows_when_a_larger_shape_is_requested(self):
        small = _alloc(shape=(4,))
        small.fill_(9.0)

        big = _alloc(shape=(64,))
        assert big.shape == (64,)
        assert big.numel() == 64
        assert not big.any()  # freshly grown allocation is zeroed

    def test_smaller_request_reuses_the_existing_allocation_as_a_trimmed_view(self):
        big = _alloc(shape=(64,))
        ptr = big.data_ptr()

        small = _alloc(shape=(8,))
        assert small.shape == (8,)
        assert small.numel() == 8
        assert small.data_ptr() == ptr  # no reallocation on shrink
        assert len(_prealloc_buffers) == 1

    def test_varying_shape_at_one_call_site_keeps_a_single_cache_entry(self):
        for n in (2, 5, 32, 7, 64, 1):
            buf = _alloc(shape=(n,))
            assert buf.shape == (n,)
            assert not buf.any()
        assert len(_prealloc_buffers) == 1
