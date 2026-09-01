"""Unit tests for ``_pair`` scalar/tuple normalization of conv arguments.

These do not need a distributed environment: ``_pair`` is the pure helper that
lets ``DistributedConv2D`` accept ``stride`` / ``padding`` / ``dilation`` as a
plain ``int`` instead of a ``tuple[int, int]``.
"""

import pytest

from jigsaw.modules.convolution.common.arg_norm import _ntuple

_pair = _ntuple(2)


class TestPair:
    def test_int_is_broadcast(self) -> None:
        """A scalar ``int`` becomes a symmetric 2-tuple."""
        assert _pair(4) == (4, 4)
        assert _pair(0) == (0, 0)
        assert _pair(1) == (1, 1)

    def test_single_element_tuple_is_broadcast(self) -> None:
        """A length-1 tuple (the module's default ``padding=(0,)``) is broadcast."""
        assert _pair((3,)) == (3, 3)
        assert _pair((0,)) == (0, 0)

    def test_two_element_tuple_is_passed_through(self) -> None:
        """An asymmetric 2-tuple is preserved as-is."""
        assert _pair((3, 5)) == (3, 5)

    def test_longer_tuple_is_truncated(self) -> None:
        """Tuples longer than 2 are silently truncated to their first two items."""
        assert _pair((1, 2, 3)) == (1, 2)

    def test_empty_tuple_raises(self) -> None:
        """An empty tuple has no elements to broadcast and raises ``IndexError``."""
        with pytest.raises(IndexError):
            _pair(())
