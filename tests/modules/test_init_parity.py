"""Distributed layers must init from the same distribution as their non-distributed torch counterpart.

``torch.nn.Linear`` / ``Conv{1,2,3}d`` / ``ConvTranspose{1,2,3}d`` draw ``weight``
from ``kaiming_uniform_(a=sqrt(5))`` and ``bias`` from ``U(-1/sqrt(fan_in),
1/sqrt(fan_in))``, both keyed off the layer's *global* fan. A distributed jigsaw
layer only ever samples a shard of that tensor, so the bound that shard is drawn
from must be computed from the same global fan, not from the local shard's own
(smaller) shape. If the row- or channel-group shard size leaks into the bound,
the concatenated global parameter is still uniform, just over a *narrower* range
than ``torch.nn`` would use: invisible to a plain sharded-forward-pass check, but
a real difference in the trained model's effective init scale.

These tests gather each layer's global weight/bias (via the module's own
``merge_weights``) and check its empirical bound against the closed-form
``torch.nn`` bound: at most the bound (``uniform_`` samples ``[-b, b)``) and,
since the gathered tensor holds thousands of i.i.d. uniform draws, above 85% of
it, catching an accidental extra ``1/sqrt(group size)`` factor either way.
"""

import math

import pytest
import torch

import jigsaw.init as init
import jigsaw.modules.convolution as convolution
import jigsaw.modules.linear as linear

pytestmark = [pytest.mark.dist_required]


def _assert_bound(
    tensor: torch.Tensor, bound: float, *, low_frac: float = 0.85
) -> None:
    """``tensor`` should be ~U(-bound, bound): its max magnitude is at most the bound
    and (with enough elements) above ``low_frac * bound``."""
    max_abs = tensor.abs().max().item()
    assert max_abs <= bound + 1e-6, f"{max_abs} exceeds the torch.nn bound {bound}"
    assert max_abs > low_frac * bound, (
        f"{max_abs} is well under the torch.nn bound {bound}; looks like a "
        "group-size factor leaked into the sampling bound"
    )


class TestLinearInitParity:
    @pytest.mark.dist(4)
    def test_weight_and_bias_match_torch_linear(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        mesh = process_mesh_dict_2x2
        in_dim, out_dim = 64, 128
        local_rank = torch.distributed.get_rank(group=mesh["model"])

        layer = linear.DistributedLinear(
            in_dim,
            out_dim,
            local_rank=local_rank,
            device=device,
            model_group=mesh["model"],
            col_group=mesh["col"],
            row_group=mesh["row"],
            bias=True,
        ).to(device)

        weight, bias = layer.merge_weights(mesh)
        assert weight.shape == (out_dim, in_dim)
        assert bias is not None and bias.shape == (out_dim,)

        bound = 1.0 / math.sqrt(in_dim)  # torch.nn.Linear's fan_in bound
        _assert_bound(weight, bound)
        _assert_bound(bias, bound)


_CONV_CLS = {
    1: convolution.DistributedConv1d,
    2: convolution.DistributedConv2d,
    3: convolution.DistributedConv3d,
}


class TestConvInitParity:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("n_spatial", [1, 2, 3])
    def test_weight_and_bias_match_torch_conv(
        self, device: torch.device, process_mesh_dict_2x2: dict, n_spatial: int
    ) -> None:
        mesh = process_mesh_dict_2x2
        in_channels, out_channels, kernel_size = 8, 32, 3

        conv = _CONV_CLS[n_spatial](
            in_channels,
            out_channels,
            kernel_size,
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            bias=True,
        ).to(device)

        weight, bias = conv.merge_weights(mesh, col_kw="col")
        assert weight.shape == (out_channels, in_channels, *([kernel_size] * n_spatial))
        assert bias is not None and bias.shape == (out_channels,)

        fan_in = in_channels * kernel_size**n_spatial
        bound = 1.0 / math.sqrt(fan_in)  # torch.nn.ConvNd's fan_in bound
        _assert_bound(weight, bound)
        _assert_bound(bias, bound)


_CONV_TRANSPOSE_CLS = {
    1: convolution.DistributedConvTranspose1d,
    2: convolution.DistributedConvTranspose2d,
    3: convolution.DistributedConvTranspose3d,
}


class TestConvTransposeInitParity:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("n_spatial", [1, 2, 3])
    def test_weight_and_bias_match_torch_conv_transpose(
        self, device: torch.device, process_mesh_dict_2x2: dict, n_spatial: int
    ) -> None:
        mesh = process_mesh_dict_2x2
        in_channels, out_channels, kernel_size = 8, 32, 3

        conv_t = _CONV_TRANSPOSE_CLS[n_spatial](
            in_channels,
            out_channels,
            kernel_size,
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            bias=True,
        ).to(device)

        weight, bias = conv_t.merge_weights(mesh, col_kw="col")
        assert weight.shape == (in_channels, out_channels, *([kernel_size] * n_spatial))
        assert bias is not None and bias.shape == (out_channels,)

        # ConvTranspose's weight leads with the input-channel axis, so torch
        # reads fan_in off axis 1: (C_out / groups) * prod(kernel).
        fan_in = out_channels * kernel_size**n_spatial
        bound = 1.0 / math.sqrt(fan_in)
        _assert_bound(weight, bound)
        _assert_bound(bias, bound)


class TestResetParametersIsOverridable:
    """A subclass overriding ``reset_parameters`` changes the init, jigsaw.init-style."""

    @pytest.mark.dist(4)
    def test_subclass_override_replaces_the_default_distribution(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        mesh = process_mesh_dict_2x2

        class ZerosConv2d(convolution.DistributedConv2d):
            def reset_parameters(self) -> None:
                init.zeros_(self.weight)
                if self.bias is not None:
                    init.constant_(self.bias, 0.5)

        conv = ZerosConv2d(
            8,
            16,
            3,
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            bias=True,
        ).to(device)

        weight, bias = conv.merge_weights(mesh, col_kw="col")
        assert torch.equal(weight, torch.zeros_like(weight))
        assert bias is not None
        assert torch.equal(bias, torch.full_like(bias, 0.5))
