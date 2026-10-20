"""Tests for the module-level API on :class:`_DistributedConvNd` (``conv/base.py``).

Covers the pieces the forward/backward reference suites never touch:

* ``spatial_world_size`` / ``get_parameter_count`` -- the effective (replica-
  divided) local parameter count.
* ``convert_to_torch`` -- rebuilding a plain ``torch.nn.Conv{1,2,3}d`` from the
  channel-sharded, spatially-replicated jigsaw layer and checking it reproduces
  the distributed forward.
"""

import pytest
import torch

import jigsaw.modules.convolution as convolution
from jigsaw.gather import gather_1d_results
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]

_CONV_CLS = {
    1: convolution.DistributedConv1d,
    2: convolution.DistributedConv2d,
    3: convolution.DistributedConv3d,
}
_REF_CLS = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}


class TestConvChannelSizes:
    @pytest.mark.dist(4)
    def test_out_channels_is_global_after_init_and_reshard(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        process_mesh_dict_1x4: dict,
    ) -> None:
        """``out_channels`` is the global size, like ``in_channels`` and torch;
        ``local_out_channels`` is this rank's shard."""
        conv = convolution.DistributedConv2d(
            8,
            12,
            2,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=2,
        ).to(device)
        assert (conv.in_channels, conv.out_channels) == (8, 12)
        assert conv.local_out_channels == 6 == conv.weight.shape[0]

        conv.reshard_parameters(process_mesh_dict_2x2, process_mesh_dict_1x4)
        assert (conv.in_channels, conv.out_channels) == (8, 12)
        assert conv.local_out_channels == 3 == conv.weight.shape[0]


class TestConvParameterCount:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("n_spatial", [1, 2, 3])
    def test_count_divides_by_spatial_replicas(
        self, device: torch.device, process_mesh_dict_2x2: dict, n_spatial: int
    ) -> None:
        """On a 2 (spatial) x 2 (channel) mesh the params are replicated over the
        two spatial ranks, so the effective count halves them."""
        mesh = process_mesh_dict_2x2
        conv = _CONV_CLS[n_spatial](
            8,
            12,
            2,
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            stride=2,
            bias=True,
        ).to(device)

        assert conv.spatial_world_size == mesh["row"].size() == 2
        replicas = conv.spatial_world_size
        expected = conv.weight.numel() // replicas + conv.bias.numel() // replicas
        assert conv.get_parameter_count() == expected

    @pytest.mark.dist(4)
    def test_pure_channel_split_has_no_spatial_replicas(
        self, device: torch.device, process_mesh_dict_1x4: dict
    ) -> None:
        """No split spatial axis -> ``spatial_world_size == 1`` and nothing is divided."""
        mesh = process_mesh_dict_1x4
        conv = convolution.DistributedConv2d(
            8,
            16,
            2,
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            stride=2,
            bias=True,
        ).to(device)

        assert conv.spatial_world_size == 1
        assert conv.get_parameter_count() == conv.weight.numel() + conv.bias.numel()

    @pytest.mark.dist(4)
    def test_bias_false_is_excluded_from_the_count(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """With ``bias=False`` only the weight contributes."""
        mesh = process_mesh_dict_2x2
        conv = convolution.DistributedConv2d(
            8,
            12,
            2,
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            stride=2,
            bias=False,
        ).to(device)

        assert conv.bias is None
        assert (
            conv.get_parameter_count() == conv.weight.numel() // conv.spatial_world_size
        )


class TestConvConvertToTorch:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("n_spatial", [1, 2, 3])
    def test_roundtrip_reproduces_the_distributed_forward(
        self, device: torch.device, process_mesh_dict_2x2: dict, n_spatial: int
    ) -> None:
        """``convert_to_torch`` on every rank yields a plain conv whose output
        matches the gathered distributed output for the same global input."""
        mesh = process_mesh_dict_2x2
        row, col = mesh["row"], mesh["col"]
        in_c, out_c, k = 4, 8, 2
        spatial = (16,) * n_spatial

        conv = _CONV_CLS[n_spatial](
            in_c,
            out_c,
            k,
            device=device,
            spatial_group=row,
            channel_group=col,
            stride=k,
            bias=True,
            channel_comm="all_gather",
        ).to(device)

        torch.manual_seed(0)  # identical global input on every rank
        x_global = torch.randn(2, in_c, *spatial, device=device)

        c_size = in_c // col.size()
        w_size = spatial[-1] // row.size()
        x_local = (
            x_global[:, col.rank() * c_size : (col.rank() + 1) * c_size]
            .narrow(-1, row.rank() * w_size, w_size)
            .contiguous()
        )

        y_local = conv(x_local)
        y_global = gather_1d_results(y_local.detach(), mesh, dim=1, pm_key="col")
        y_global = gather_1d_results(y_global, mesh, dim=-1, pm_key="row")

        torch_conv = conv.convert_to_torch(mesh)
        assert isinstance(torch_conv, _REF_CLS[n_spatial])
        assert tuple(torch_conv.weight.shape) == (out_c, in_c, *([k] * n_spatial))
        assert torch_conv.bias is not None

        y_ref = torch_conv(x_global)
        assert y_ref.shape == y_global.shape
        assert torch.allclose(y_global, y_ref, atol=1e-4, rtol=1e-4)


class TestConvSampledNoBias:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("n_spatial", [1, 2, 3])
    def test_sampled_weight_without_bias(
        self, device: torch.device, process_mesh_dict_4x1: dict, n_spatial: int
    ) -> None:
        """A bayesian layer with ``bias=False`` runs a per-sample weight stack.

        Patchify on a 4x1 mesh takes ``_forward_local``; the stacked weight
        selects the vmapped primitive, which must not be handed a ``None`` bias.
        """
        mesh = process_mesh_dict_4x1
        conv = _CONV_CLS[n_spatial](
            4,
            6,
            2,
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            stride=2,
            bias=False,
            bayesian=True,
        ).to(device)
        samples = 3
        weight = torch.randn(samples, *conv.weight.shape, device=device)
        x = torch.randn(samples, 2, 4, *([4] * n_spatial), device=device)

        y = torch.func.functional_call(conv, {"weight": weight}, (x,))

        native = conv.config.native_conv_fn
        y_ref = torch.stack([native(x[s], weight[s], stride=2) for s in range(samples)])
        assert_close("y", y, y_ref, 1e-5, 1e-5, device, mesh["row"])
