"""Unit tests for :mod:`jigsaw.modules.convolution._conv_transpose.channel_comm`.

``_bias_view`` is a pure reshape and needs no distributed environment. It has to
broadcast the raw bias parameter against the transpose-conv output for a plain
``(C_out,)`` bias *and* for the ``(S, C_out)`` bias handed in under ``vmap``
(stacked ensemble members); the streaming / all-gather drivers add the bias by hand
(it cannot go through ``conv_func`` in the halo regime, or overlapped positions
double-count it), so this reshape is the only thing keeping the channel axis
aligned.
"""

import pytest
import torch

from jigsaw.modules.convolution._conv_transpose.channel_comm import _bias_view


class TestBiasView:
    @pytest.mark.parametrize("n_spatial", [1, 2, 3])
    def test_plain_bias_broadcasts_over_batch_and_spatial(self, n_spatial: int) -> None:
        """A ``(C_out,)`` bias lands on the channel axis of ``(N, C_out, *spatial)``."""
        c_out = 5
        y = torch.zeros(2, c_out, *([4] * n_spatial))
        view = _bias_view(torch.arange(float(c_out)), n_spatial)

        assert view.shape == (1, c_out, *([1] * n_spatial))
        added = y + view
        assert added.shape == y.shape
        # every batch / spatial position now equals the per-channel bias
        idx = (0, slice(None)) + (0,) * n_spatial
        assert torch.equal(added[idx], torch.arange(float(c_out)))

    @pytest.mark.parametrize("n_spatial", [1, 2, 3])
    def test_vmapped_bias_keeps_the_ensemble_axis(self, n_spatial: int) -> None:
        """A ``(S, C_out)`` bias (the shape seen under ``vmap``) stays per-member.

        Regression: the old ``(-1, 1, ..., 1)`` view flattened ``(S, C_out)`` to
        ``(S * C_out, 1, ..., 1)`` and blew up against ``(S, N, C_out, *spatial)``.
        """
        s, c_out = 2, 512
        y = torch.zeros(s, 1, c_out, *([3] * n_spatial))
        b = torch.randn(s, c_out)
        view = _bias_view(b, n_spatial)

        assert view.shape == (s, 1, c_out, *([1] * n_spatial))
        added = y + view
        assert added.shape == y.shape
        idx = (slice(None), 0, slice(None)) + (0,) * n_spatial
        assert torch.equal(added[idx], b)

    def test_matches_the_failing_shapes_from_the_bug_report(self) -> None:
        """The exact shapes from the traceback: b (2, 512), y (2, 1, 512, 360, 180)."""
        b = torch.zeros(2, 512)
        y = torch.zeros(2, 1, 512, 360, 180)
        assert (y + _bias_view(b, 2)).shape == y.shape
