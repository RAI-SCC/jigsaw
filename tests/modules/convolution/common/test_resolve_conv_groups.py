"""Unit tests for ``resolve_conv_groups`` regime resolution.

Pure helper in :mod:`jigsaw.modules.convolution.common.channel_comm` shared by
the distributed conv and conv-transpose modules; no distributed environment
needed. It maps ``(groups, in_channels, out_channels, G)`` to one of three
regimes -- ``channel_local`` (rank owns whole conv-groups), ``channel_grouped``
(``1 < groups`` not a multiple of ``G``, a conv-group's channels straddle rank
boundaries) or the plain full-channel gather (``groups == 1``, ``G > 1``).
"""

import pytest

from jigsaw.modules.convolution.common.channel_comm import (
    conv_group_blocks,
    resolve_conv_groups,
)


class TestResolveConvGroups:
    @pytest.mark.parametrize(
        ("groups", "g_size", "expected"),
        [
            (1, 1, (True, False, 1)),  # no channel split
            (4, 1, (True, False, 4)),  # no channel split, grouped
            (1, 3, (False, False, 1)),  # gather regime
            (6, 3, (True, False, 2)),  # channel-local: groups a multiple of G
            (3, 3, (True, False, 1)),  # channel-local: groups == G
            (2, 3, (False, True, 1)),  # channel-grouped: G > groups
            (3, 2, (False, True, 1)),  # channel-grouped: groups > G, indivisible
            (2, 4, (False, True, 1)),  # channel-grouped: G a multiple of groups
        ],
    )
    def test_regimes(self, groups: int, g_size: int, expected: tuple) -> None:
        assert resolve_conv_groups(groups, 12, 12, g_size) == expected

    @pytest.mark.parametrize("g_size", [1, 2, 3, 4, 6, 12])
    def test_channel_local_and_grouped_are_exclusive(self, g_size: int) -> None:
        for groups in (1, 2, 3, 4, 6, 12):
            local, grouped, _ = resolve_conv_groups(groups, 12, 12, g_size)
            assert not (local and grouped)

    @pytest.mark.parametrize("groups", [5, 7, 8])
    def test_groups_not_dividing_channels_rejected(self, groups: int) -> None:
        with pytest.raises(ValueError):
            resolve_conv_groups(groups, 12, 12, 2)

    @pytest.mark.parametrize("g_size", [5, 7, 8])
    def test_channels_not_divisible_by_group_rejected(self, g_size: int) -> None:
        with pytest.raises(ValueError):
            resolve_conv_groups(2, 12, 12, g_size)

    def test_groups_below_one_rejected(self) -> None:
        with pytest.raises(ValueError):
            resolve_conv_groups(0, 12, 12, 1)


class TestConvGroupBlocks:
    def test_output_side_straddle(self) -> None:
        # C=12, groups=2 (Co_g=6), G=3 (Co_L=4): rank 1 straddles the group
        # boundary at output channel 6.
        assert conv_group_blocks(12, 12, 2, 0, 3) == [(0, 4, 0, 6)]
        assert conv_group_blocks(12, 12, 2, 1, 3) == [(0, 2, 0, 6), (2, 4, 6, 12)]
        assert conv_group_blocks(12, 12, 2, 2, 3) == [(0, 4, 6, 12)]

    def test_input_side_straddle_output_clean(self) -> None:
        # C=8, groups=2 (Cin_g=4), G=4 (Co_L=2): each rank owns one group's
        # output channels but only half of that group's 4 input channels.
        for r in range(4):
            (blk,) = conv_group_blocks(8, 8, 2, r, 4)
            out_lo, out_hi, in_lo, in_hi = blk
            assert (out_hi - out_lo) == 2
            assert (in_hi - in_lo) == 4  # the whole conv-group's inputs

    def test_blocks_tile_the_local_output_shard(self) -> None:
        for r in range(3):
            blocks = conv_group_blocks(12, 12, 2, r, 3)
            covered = 0
            for out_lo, out_hi, _, _ in blocks:
                assert out_lo == covered
                covered = out_hi
            assert covered == 12 // 3
