"""Unit tests for ``_ntuple`` scalar/tuple normalization of conv arguments.

These do not need a distributed environment: ``_ntuple`` is the pure helper in
:mod:`jigsaw.modules.convolution._common.arg_norm` shared by the conv and
conv-transpose modules; it lets them accept ``stride`` / ``padding`` /
``dilation`` as a plain ``int`` instead of a per-axis tuple. ``_pair`` here is
the 2-axis instance used by the 2D modules.
"""

import pytest

from jigsaw.modules.convolution._common.arg_norm import _ntuple, _resolve_padding_str

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

    def test_longer_tuple_raises(self) -> None:
        """A tuple with more entries than axes is rejected, not truncated."""
        with pytest.raises(ValueError, match="1 or 2 entries"):
            _pair((1, 2, 3))

    def test_empty_tuple_raises(self) -> None:
        """An empty tuple has no elements to broadcast and raises ``ValueError``."""
        with pytest.raises(ValueError, match="1 or 2 entries"):
            _pair(())


class TestResolvePaddingStr:
    def test_valid_is_zero(self) -> None:
        """``"valid"`` is no padding on every axis."""
        assert _resolve_padding_str("valid", (3, 4), (2, 2), (1, 1)) == (0, 0)

    def test_symmetric_same_is_half_the_span(self) -> None:
        """``"same"`` pads ``dilation * (kernel_size - 1) / 2`` on each side."""
        assert _resolve_padding_str("same", (3, 5), (1, 1), (2, 1)) == (2, 2)

    def test_asymmetric_same_stays_a_string(self) -> None:
        """An odd padding total cannot be split evenly and is left to torch."""
        assert _resolve_padding_str("same", (3, 4), (1, 1), (1, 1)) == "same"

    def test_strided_same_raises(self) -> None:
        """``"same"`` with a stride is rejected, as in torch."""
        with pytest.raises(ValueError, match="strided"):
            _resolve_padding_str("same", (3, 3), (2, 1), (1, 1))

    def test_unknown_string_raises(self) -> None:
        """Strings other than ``"valid"`` and ``"same"`` are rejected."""
        with pytest.raises(ValueError, match="Invalid padding string"):
            _resolve_padding_str("full", (3, 3), (1, 1), (1, 1))
