"""Tests for :mod:`jigsaw.init`, the shard-aware ``torch.nn.init`` counterpart.

Two properties matter here:

* With no ``global_shape`` / ``fan_in`` / ``fan_out`` given, every function must be a
  bit-for-bit drop-in for its ``torch.nn.init`` namesake -- these are plain,
  non-distributed calls and jigsaw never gets to change their behaviour.
* With a ``global_shape`` describing a bigger (unsharded) tensor, the fan -- and
  therefore the sampling bound / std -- must be computed from that shape, not from the
  tensor actually being filled. This is what lets a jigsaw layer initialize a local
  shard from the same distribution as the equivalent whole, non-distributed parameter.
"""

import math

import pytest
import torch

import jigsaw.init as init


def _seeded(fn, *args, seed: int = 0, **kwargs) -> torch.Tensor:
    """Run ``fn(*args, generator=Generator(seed), **kwargs)`` and return the tensor."""
    generator = torch.Generator().manual_seed(seed)
    return fn(*args, generator=generator, **kwargs)


class TestFanCalculation:
    def test_matches_torch_private_helper(self) -> None:
        t = torch.empty(12, 6, 3, 3)
        expected = torch.nn.init._calculate_fan_in_and_fan_out(t)
        assert init._calculate_fan_in_and_fan_out(t) == expected

    def test_accepts_a_plain_shape(self) -> None:
        assert init._calculate_fan_in_and_fan_out((12, 6, 3, 3)) == (6 * 9, 12 * 9)

    def test_accepts_a_torch_size(self) -> None:
        shape = torch.Size((12, 6, 3, 3))
        assert init._calculate_fan_in_and_fan_out(shape) == (6 * 9, 12 * 9)

    def test_rejects_fewer_than_two_dims(self) -> None:
        with pytest.raises(ValueError):
            init._calculate_fan_in_and_fan_out((6,))

    def test_2d_shape_has_no_receptive_field(self) -> None:
        assert init._calculate_fan_in_and_fan_out((16, 8)) == (8, 16)


class TestDropInEquivalence:
    """No ``global_shape`` -> identical to ``torch.nn.init``, seed for seed."""

    def test_kaiming_uniform(self) -> None:
        a = _seeded(torch.nn.init.kaiming_uniform_, torch.empty(8, 5), a=math.sqrt(5))
        b = _seeded(init.kaiming_uniform_, torch.empty(8, 5), a=math.sqrt(5))
        assert torch.equal(a, b)

    def test_kaiming_normal(self) -> None:
        a = _seeded(torch.nn.init.kaiming_normal_, torch.empty(8, 5, 3))
        b = _seeded(init.kaiming_normal_, torch.empty(8, 5, 3))
        assert torch.equal(a, b)

    def test_xavier_uniform(self) -> None:
        a = _seeded(torch.nn.init.xavier_uniform_, torch.empty(8, 5))
        b = _seeded(init.xavier_uniform_, torch.empty(8, 5))
        assert torch.equal(a, b)

    def test_xavier_normal(self) -> None:
        a = _seeded(torch.nn.init.xavier_normal_, torch.empty(8, 5))
        b = _seeded(init.xavier_normal_, torch.empty(8, 5))
        assert torch.equal(a, b)

    @pytest.mark.parametrize(
        "name", ["uniform_", "normal_", "constant_", "ones_", "zeros_", "trunc_normal_"]
    )
    def test_shape_independent_helpers_are_torchs_own(self, name: str) -> None:
        """These are re-exported verbatim -- same object, not just same behaviour."""
        assert getattr(init, name) is getattr(torch.nn.init, name)


class TestGlobalShapeOverride:
    """``global_shape`` keys the fan off a bigger, unsharded tensor."""

    def test_kaiming_uniform_bound_uses_global_fan_in(self) -> None:
        # A (4, 6) shard of a (4, 60) global weight must be bounded by the
        # *global* fan_in (60), not the local one (6).
        torch.manual_seed(0)
        shard = torch.empty(4, 6)
        init.kaiming_uniform_(shard, a=math.sqrt(5), global_shape=(4, 60))
        bound = 1.0 / math.sqrt(60)
        assert shard.abs().max().item() <= bound + 1e-6
        # sanity: the *local* fan_in bound would have been visibly looser.
        assert bound < 1.0 / math.sqrt(6)

    def test_xavier_uniform_bound_uses_global_fans(self) -> None:
        torch.manual_seed(0)
        shard = torch.empty(4, 6)
        init.xavier_uniform_(shard, global_shape=(4, 60))
        fan_in, fan_out = 60, 4
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        assert shard.abs().max().item() <= bound + 1e-6

    def test_explicit_fan_in_overrides_global_shape(self) -> None:
        torch.manual_seed(0)
        shard = torch.empty(1000)
        init.kaiming_uniform_(
            shard.unsqueeze(0), a=math.sqrt(5), global_shape=(1, 5), fan_in=1000
        )
        bound = 1.0 / math.sqrt(1000)
        assert shard.abs().max().item() <= bound + 1e-6

    def test_explicit_fan_out_overrides_global_shape(self) -> None:
        torch.manual_seed(0)
        shard = torch.empty(1, 1000)
        init.xavier_uniform_(shard, global_shape=(5, 5), fan_out=1000)
        # ``fan_in`` still comes from the global shape.
        bound = math.sqrt(6.0 / (5 + 1000))
        assert shard.abs().max().item() <= bound + 1e-6

    def test_sharded_shards_concatenate_to_the_global_distribution(self) -> None:
        """Filling two shards independently with the same global_shape reproduces
        (in distribution) a single non-sharded draw over the concatenation."""
        global_shape = (4, 4000)
        torch.manual_seed(0)
        whole = torch.empty(*global_shape)
        torch.nn.init.kaiming_uniform_(whole, a=math.sqrt(5))

        torch.manual_seed(0)
        left = torch.empty(4, 2000)
        init.kaiming_uniform_(left, a=math.sqrt(5), global_shape=global_shape)
        torch.manual_seed(1)
        right = torch.empty(4, 2000)
        init.kaiming_uniform_(right, a=math.sqrt(5), global_shape=global_shape)

        bound = 1.0 / math.sqrt(4000)
        for part in (whole, left, right):
            assert part.abs().max().item() <= bound + 1e-6
        # both halves were drawn from the same (global-fan) bound, so their
        # empirical maxima should land in the same ballpark -- this is exactly
        # the property that broke before global_shape existed (a channel- or
        # row-sharded rank silently sampled a *tighter* distribution).
        assert left.abs().max().item() > 0.85 * bound
        assert right.abs().max().item() > 0.85 * bound


class TestEmptyTensor:
    """Matches ``torch.nn.init``'s no-op guard for degenerate shards."""

    def test_kaiming_uniform_noop_on_empty(self) -> None:
        t = torch.empty(0, 4)
        out = init.kaiming_uniform_(t, a=math.sqrt(5), global_shape=(8, 4))
        assert out.numel() == 0

    def test_xavier_uniform_noop_on_empty(self) -> None:
        t = torch.empty(4, 0)
        out = init.xavier_uniform_(t, global_shape=(4, 8))
        assert out.numel() == 0

    def test_kaiming_normal_noop_on_empty(self) -> None:
        out = init.kaiming_normal_(torch.empty(0, 4), global_shape=(8, 4))
        assert out.numel() == 0

    def test_xavier_normal_noop_on_empty(self) -> None:
        out = init.xavier_normal_(torch.empty(4, 0), global_shape=(4, 8))
        assert out.numel() == 0


class TestModeValidation:
    @pytest.mark.parametrize("fn", [init.kaiming_uniform_, init.kaiming_normal_])
    def test_unknown_mode_is_rejected(self, fn) -> None:
        with pytest.raises(ValueError, match="Mode fan_avg not supported"):
            fn(torch.empty(4, 4), mode="fan_avg")

    def test_mode_is_case_insensitive(self) -> None:
        a = _seeded(init.kaiming_normal_, torch.empty(8, 5), mode="FAN_OUT")
        b = _seeded(init.kaiming_normal_, torch.empty(8, 5), mode="fan_out")
        assert torch.equal(a, b)

    def test_explicit_fans_override_the_tensor(self) -> None:
        got = _seeded(init.kaiming_normal_, torch.empty(8, 5), fan_in=100)
        want = _seeded(torch.nn.init.normal_, torch.empty(8, 5), std=math.sqrt(2 / 100))
        assert torch.allclose(got, want)
