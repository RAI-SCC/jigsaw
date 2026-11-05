"""Forward input shapes and passed-in configs of the (transpose) convolution layers.

* Unbatched input ``(C, *spatial)``, which torch accepts, runs as a batch of one.
  The parity cases cover the channel-gather path (``groups=1``) and the
  channel-local path (``groups=2``), with and without a halo exchange, on a
  2 (spatial) x 2 (channel) mesh.
* A config built for another layer is rejected, since it would select the wrong
  kernels (e.g. no bias gradient).
"""

import pytest
import torch
import torch.distributed as dist

import jigsaw.convert as convert
import jigsaw.modules.convolution as convolution
from tests.parity import REPLICATED, ShardSpec, check_parity

pytestmark = [pytest.mark.dist_required]

_TORCH_CLS = {
    ("conv", 1): torch.nn.Conv1d,
    ("conv", 2): torch.nn.Conv2d,
    ("transpose", 1): torch.nn.ConvTranspose1d,
    ("transpose", 2): torch.nn.ConvTranspose2d,
}
_JIGSAW_CLS = {
    ("conv", 1): convolution.DistributedConv1d,
    ("conv", 2): convolution.DistributedConv2d,
    ("transpose", 1): convolution.DistributedConvTranspose1d,
    ("transpose", 2): convolution.DistributedConvTranspose2d,
}
_CONFIG_CLS = {
    "conv": convolution.ConvNdConfig,
    "transpose": convolution.ConvTransposeNdConfig,
}

# (n_spatial, global spatial shape, kernel, stride, padding): no halo, then halo.
_CASES = [(1, (32,), 4, 4, 0), (2, (8, 16), 4, 2, 1)]
_IDS = [f"{n}d_k{k}_s{s}_p{p}" for (n, _, k, s, p) in _CASES]


class TestUnbatchedInput:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("kind", ["conv", "transpose"])
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize(
        ("n_spatial", "spatial", "kernel", "stride", "padding"), _CASES, ids=_IDS
    )
    def test_matches_torch(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        kind: str,
        groups: int,
        n_spatial: int,
        spatial: tuple[int, ...],
        kernel: int,
        stride: int,
        padding: int,
    ) -> None:
        """Forward, input grad and parameter grads match torch on unbatched input."""
        mesh = process_mesh_dict_2x2
        # Channels (dim 0) split over col, width (last dim) over row.
        data_spec = ShardSpec(col=0, row=n_spatial)
        # The transpose weight ``(C_in, C_out / groups, *k)`` is sharded on the
        # output axis, or on the input axis when each rank owns whole groups.
        weight_dim = 1 if kind == "transpose" and groups == 1 else 0
        param_rules = {
            "weight": ShardSpec(col=weight_dim, row=REPLICATED),
            "bias": ShardSpec(col=0, row=REPLICATED),
        }

        torch.manual_seed(0)  # identical serial reference on every rank
        serial = _TORCH_CLS[kind, n_spatial](
            4,
            8,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            device=device,
        )

        def build_distributed(module: torch.nn.Module) -> torch.nn.Module:
            converted = convert.convert_to_jigsaw(module, mesh, device)
            assert isinstance(converted, _JIGSAW_CLS[kind, n_spatial])
            return converted

        check_parity(
            serial_module=serial,
            build_distributed=build_distributed,
            input_shape=(4, *spatial),
            input_spec=data_spec,
            output_spec=data_spec,
            param_rules=param_rules,
            process_mesh_dict=mesh,
            device=device,
            atol=1e-4,
            rtol=1e-4,
        )

    @pytest.mark.parametrize("kind", ["conv", "transpose"])
    def test_wrong_rank_rejected(self, device: torch.device, kind: str) -> None:
        """An input that is neither ``(C, L)`` nor ``(N, C, L)`` raises ``ValueError``."""
        group = dist.group.WORLD
        layer = _JIGSAW_CLS[kind, 1](
            8, 8, 3, device=device, spatial_group=[None], channel_group=group
        )
        with pytest.raises(ValueError, match="2-D \\(unbatched\\) or 3-D"):
            layer(torch.zeros(2, 2, 2, 8, device=device))


class TestConfigForAnotherLayer:
    @pytest.mark.parametrize("kind", ["conv", "transpose"])
    @pytest.mark.parametrize(
        ("field", "built_for"),
        [
            ("n_spatial", {"kernel_size": (2,), "stride": (2,)}),
            ("bias", {"bias": None}),
            ("kernel_size == stride", {"stride": (1, 1)}),
            ("groups", {"groups": 2}),
        ],
        ids=["n_spatial", "bias", "halo", "groups"],
    )
    def test_rejected(
        self, device: torch.device, kind: str, field: str, built_for: dict
    ) -> None:
        """A config whose layer differs in one property raises ``ValueError``."""
        group = dist.group.WORLD
        layer = {
            "bias": torch.zeros(1),
            "kernel_size": (2, 2),
            "stride": (2, 2),
            "groups": 1,
        }
        layer.update(built_for)
        config = _CONFIG_CLS[kind](
            group,
            layer["bias"],
            layer["kernel_size"],
            layer["stride"],
            device,
            groups=layer["groups"],
        )
        with pytest.raises(ValueError, match=f"{field}="):
            _JIGSAW_CLS[kind, 2](
                8,
                8,
                2,
                device=device,
                spatial_group=group,
                channel_group=group,
                stride=2,
                config=config,
            )

    @pytest.mark.parametrize("kind", ["conv", "transpose"])
    def test_matching_config_accepted(self, device: torch.device, kind: str) -> None:
        """A config built for the layer's own properties is used as is."""
        group = dist.group.WORLD
        config = _CONFIG_CLS[kind](
            group, torch.zeros(1), (2, 2), (2, 2), device, channel_comm="all_gather"
        )
        layer = _JIGSAW_CLS[kind, 2](
            8,
            8,
            2,
            device=device,
            spatial_group=group,
            channel_group=group,
            stride=2,
            config=config,
        )
        assert layer.config is config
