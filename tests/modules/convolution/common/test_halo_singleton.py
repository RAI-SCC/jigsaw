"""Halo helpers on a spatial group of one rank.

With a single rank there is nothing to exchange, so each helper takes a local shortcut.
Every shortcut has a serial meaning that is checked here against plain torch:

* ``halo_gather`` zero-pads the input the way a padded convolution does, and
  ``halo_scatter_add`` is its adjoint (the gradient of the padded slice folds back onto
  the owned positions);
* ``transpose_halo_reduce`` crops the raw transpose-convolution output to what a padded
  ``conv_transpose`` would return, and ``transpose_halo_expand`` is its adjoint.

The adjoint identity ``<A x, g> == <x, A^T g>`` needs no reference implementation.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from jigsaw.modules.convolution.conv.halo import halo_gather, halo_scatter_add
from jigsaw.modules.convolution.conv_transpose.halo import (
    transpose_halo_expand,
    transpose_halo_reduce,
)

# (kernel, stride, padding, dilation)
CONV_CASES = [(3, 1, 1, 1), (3, 1, 2, 1), (3, 2, 0, 1), (3, 2, 1, 1), (2, 1, 0, 2)]
LOCAL_N = 9


def _singleton_group() -> dist.ProcessGroup:
    """A process group containing only this rank, valid at any world size."""
    groups = [dist.new_group([r]) for r in range(dist.get_world_size())]
    return groups[dist.get_rank()]


@pytest.fixture(scope="module")
def solo() -> dist.ProcessGroup:
    return _singleton_group()


@pytest.mark.parametrize("kernel, stride, padding, dilation", CONV_CASES)
class TestConvHalo:
    def test_gather_matches_a_padded_convolution(
        self, solo, kernel, stride, padding, dilation
    ):
        torch.manual_seed(0)
        x = torch.randn(2, 3, LOCAL_N)
        w = torch.randn(4, 3, kernel)
        x_need, out_local = halo_gather(x, solo, kernel, stride, padding, dilation)
        ref = F.conv1d(x, w, stride=stride, padding=padding, dilation=dilation)
        got = F.conv1d(x_need, w, stride=stride, padding=0, dilation=dilation)
        assert out_local == ref.shape[-1]
        assert torch.allclose(got, ref, atol=1e-6)

    def test_scatter_add_is_the_adjoint_of_gather(
        self, solo, kernel, stride, padding, dilation
    ):
        torch.manual_seed(1)
        x = torch.randn(2, 3, LOCAL_N)
        x_need, _ = halo_gather(x, solo, kernel, stride, padding, dilation)
        g = torch.randn_like(x_need)
        g_x = halo_scatter_add(g, solo, kernel, stride, padding, dilation, LOCAL_N)
        assert g_x.shape == x.shape
        assert torch.allclose((x_need * g).sum(), (x * g_x).sum(), atol=1e-4)


# (kernel, stride, padding, dilation)
TRANSPOSE_CASES = [(3, 1, 0, 1), (3, 1, 1, 1), (3, 2, 1, 1), (4, 2, 1, 1), (2, 1, 0, 2)]


@pytest.mark.parametrize("kernel, stride, padding, dilation", TRANSPOSE_CASES)
class TestTransposeHalo:
    def test_reduce_matches_a_padded_transpose_convolution(
        self, solo, kernel, stride, padding, dilation
    ):
        torch.manual_seed(2)
        x = torch.randn(2, 3, LOCAL_N)
        w = torch.randn(3, 4, kernel)
        raw = F.conv_transpose1d(x, w, stride=stride, dilation=dilation)
        got = transpose_halo_reduce(
            raw, solo, LOCAL_N, kernel, stride, padding, dilation
        )
        ref = F.conv_transpose1d(
            x, w, stride=stride, padding=padding, dilation=dilation
        )
        assert torch.allclose(got, ref, atol=1e-5)

    def test_expand_is_the_adjoint_of_reduce(
        self, solo, kernel, stride, padding, dilation
    ):
        torch.manual_seed(3)
        raw = torch.randn(2, 4, (LOCAL_N - 1) * stride + dilation * (kernel - 1) + 1)
        reduced = transpose_halo_reduce(
            raw, solo, LOCAL_N, kernel, stride, padding, dilation
        )
        g = torch.randn_like(reduced)
        g_raw = transpose_halo_expand(
            g, solo, LOCAL_N, kernel, stride, padding, dilation
        )
        assert g_raw.shape == raw.shape
        assert torch.allclose((reduced * g).sum(), (raw * g_raw).sum(), atol=1e-4)
