"""Group-free checks of the roll modules, config and kernels.

With no distributed dimensions ``DistributedRoll`` is a plain ``torch.roll`` and issues
no communication, so these run in the single-process session and on every rank.
"""

import pytest
import torch

from jigsaw import comm
from jigsaw.modules import roll
from jigsaw.modules.roll import distributed_roll as dist_roll
from jigsaw.modules.roll.config import RollConfig

CPU = torch.device("cpu")


def _module(**kwargs) -> roll.DistributedRoll:
    args = dict(
        shifts=(2, -1),
        dims=(0, 2),
        distributed_dims=(),
        spatial_groups=(),
        device=CPU,
    )
    return roll.DistributedRoll(**(args | kwargs))


class TestConfig:
    def test_cpu_uses_p2p_halo(self):
        assert RollConfig(CPU).comm_halo_1d_fn is comm.comm_1d_p2p

    def test_cuda_uses_alltoall_halo(self):
        assert RollConfig(torch.device("cuda")).comm_halo_1d_fn is comm.comm_1d_alltoall

    def test_unknown_device_is_rejected(self):
        with pytest.raises(ValueError, match="No halo 1d function"):
            RollConfig(torch.device("meta"))


class TestTorchRollWrapper:
    def test_matches_torch_roll(self):
        x = torch.randn(3, 4, 5)
        module = roll.Roll(shifts=(1, -2), dims=(0, 2))
        assert torch.equal(module(x), torch.roll(x, (1, -2), (0, 2)))


class TestDistributedRollWithoutGroups:
    def test_forward_matches_torch_roll(self):
        x = torch.randn(4, 3, 5)
        assert torch.equal(_module()(x), torch.roll(x, (2, -1), (0, 2)))

    def test_backward_rolls_the_gradient_back(self):
        x = torch.randn(4, 3, 5, requires_grad=True)
        w = torch.randn(4, 3, 5)
        (_module()(x) * w).sum().backward()
        assert torch.equal(x.grad, torch.roll(w, (-2, 1), (0, 2)))

    def test_parameter_count_is_zero(self):
        assert _module().get_parameter_count() == 0

    def test_convert_to_torch(self):
        module = _module(dtype=torch.float64)
        assert module.convert_to_torch() == {
            "shifts": (2, -1),
            "dims": (0, 2),
            "dtype": torch.float64,
        }

    def test_supplied_config_is_used(self):
        config = RollConfig(CPU)
        assert _module(config=config).config is config

    def test_default_config_is_built_from_the_device(self):
        assert _module().config.comm_halo_1d_fn is comm.comm_1d_p2p

    def test_vmap_over_a_leading_batch_matches_a_loop(self):
        x = torch.randn(3, 4, 5, 6)
        module = _module(shifts=(1, -2), dims=(0, 2))
        got = torch.vmap(module)(x)
        want = torch.stack([torch.roll(t, (1, -2), (0, 2)) for t in x])
        assert torch.equal(got, want)


class TestValidation:
    def test_mismatched_group_count(self):
        with pytest.raises(ValueError, match="must align"):
            _module(distributed_dims=(0,), spatial_groups=())

    def test_distributed_dim_must_be_rolled(self):
        with pytest.raises(ValueError, match="must appear in dims"):
            _module(distributed_dims=(1,), spatial_groups=(None,))

    def test_reshard_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="mismatched lengths"):
            _module().reshard([0], [])

    def test_reshard_replaces_the_layout(self):
        module = _module()
        module.reshard([0, 2], ["g0", "g2"])
        assert module.distributed_dims == (0, 2)
        assert module.spatial_groups == ("g0", "g2")

    def test_functional_rejects_distributed_dim_outside_dims(self):
        with pytest.raises(ValueError, match="not in dims"):
            dist_roll.distributed_roll(
                torch.zeros(2, 2), (1,), (0,), [1], [None], torch.empty(0), None
            )


class TestExchangeHalo:
    def test_zero_window_is_a_noop(self):
        x = torch.randn(4, 3)

        def fail(*args, **kwargs):
            raise AssertionError("no communication expected")

        out = dist_roll.exchange_halo_1d(x, 0, 0, None, torch.empty(0), fail)
        assert out is x

    @pytest.mark.parametrize(
        "window, expected_slice", [(2, slice(0, 2)), (-2, slice(-2, None))]
    )
    def test_halo_slice_is_sent_and_replaced_in_place(self, window, expected_slice):
        x = torch.arange(12.0).reshape(6, 2)
        sent = {}

        def fake_comm(data, buffer, offset, **kwargs):
            sent["data"], sent["offset"] = data.clone(), offset
            buffer.fill_(-1.0)
            return buffer, None

        out = dist_roll.exchange_halo_1d(x, 0, window, None, torch.empty(0), fake_comm)
        reference = torch.arange(12.0).reshape(6, 2)
        assert torch.equal(sent["data"], reference[expected_slice])
        assert sent["offset"] == (1 if window > 0 else -1)
        assert out is x
        assert torch.all(x[expected_slice] == -1.0)
        untouched = torch.ones(6, dtype=torch.bool)
        untouched[expected_slice] = False
        assert torch.equal(x[untouched], reference[untouched])
