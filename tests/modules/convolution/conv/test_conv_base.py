"""Tests for the module-level API on :class:`_DistributedConvNd` (``conv/base.py``).

Covers the pieces the forward/backward reference suites never touch:

* ``spatial_world_size`` / ``get_parameter_count``: the global parameter
  count, computed from the global shapes, against the plain torch layer.
* ``convert_to_torch``: rebuilding a plain ``torch.nn.Conv{1,2,3}d`` from the
  channel-sharded, spatially-replicated jigsaw layer and checking it reproduces
  the distributed forward.
"""

import pytest
import torch
import torch.distributed as dist

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

    @pytest.mark.dist(4)
    def test_reshard_keeps_config_class_and_channel_comm(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        process_mesh_dict_1x4: dict,
    ) -> None:
        """A config passed to ``__init__`` survives a reshard."""

        class _Config(convolution.ConvNdConfig):
            pass

        col = process_mesh_dict_2x2["col"]
        config = _Config(col, None, (2, 2), (2, 2), device, channel_comm="isend")
        conv = convolution.DistributedConv2d(
            8,
            12,
            2,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=col,
            stride=2,
            bias=False,
            config=config,
        )
        conv.reshard_parameters(process_mesh_dict_2x2, process_mesh_dict_1x4)
        assert type(conv.config) is _Config
        assert conv.config.channel_comm == "isend"


class TestConvArgumentValidation:
    """Invalid constructor arguments raise on every rank before any collective."""

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("in_channels", "out_channels"), [(3, 4), (4, 3)], ids=["in", "out"]
    )
    def test_channels_indivisible_by_channel_group(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """A channel count the channel group (size 2) cannot shard is rejected."""
        with pytest.raises(ValueError, match="channel group size"):
            convolution.DistributedConv2d(
                in_channels,
                out_channels,
                3,
                device=device,
                spatial_group=process_mesh_dict_2x2["row"],
                channel_group=process_mesh_dict_2x2["col"],
            )

    @pytest.mark.parametrize("arg", ["kernel_size", "stride", "padding", "dilation"])
    def test_tuple_with_wrong_length(self, device: torch.device, arg: str) -> None:
        """A 3-tuple for a 2-D conv is rejected instead of truncated."""
        group = dist.group.WORLD
        kwargs = {"kernel_size": 3, arg: (1, 1, 1)}
        with pytest.raises(ValueError, match="1 or 2 entries"):
            convolution.DistributedConv2d(
                8,
                8,
                device=device,
                spatial_group=group,
                channel_group=group,
                **kwargs,
            )

    @pytest.mark.parametrize(
        ("padding", "stride"), [("full", 1), ("same", 2)], ids=["invalid", "strided"]
    )
    def test_padding_string_rejected(
        self, device: torch.device, padding: str, stride: int
    ) -> None:
        """Padding strings torch rejects raise ``ValueError`` here too."""
        group = dist.group.WORLD
        with pytest.raises(ValueError, match="padding"):
            convolution.DistributedConv2d(
                8,
                8,
                3,
                device=device,
                spatial_group=group,
                channel_group=group,
                stride=stride,
                padding=padding,
            )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [1, 2], ids=["channel-split", "channel-local"])
    def test_patchify_rejects_windows_across_shards(
        self, device: torch.device, process_mesh_dict_2x2: dict, groups: int
    ) -> None:
        """With ``kernel_size == stride`` a local size that is not a multiple of the
        stride would put a window across two ranks."""
        conv = convolution.DistributedConv1d(
            4,
            4,
            4,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=4,
            groups=groups,
        )
        with pytest.raises(ValueError, match="multiple of its stride"):
            conv(torch.randn(1, 2, 5, device=device))

    @pytest.mark.parametrize("padding_mode", ["reflect", "replicate", "circular"])
    def test_convert_rejects_non_zero_padding_mode(
        self, device: torch.device, padding_mode: str
    ) -> None:
        """Only zero padding converts; other modes would silently change results."""
        group = dist.group.WORLD
        model = torch.nn.Conv2d(8, 8, 3, padding=1, padding_mode=padding_mode)
        with pytest.raises(NotImplementedError, match="padding_mode"):
            convolution.DistributedConv2d.convert_to_jigsaw(
                model, {"row": group, "col": group, "model": group}, device
            )

    @pytest.mark.dist(4)
    def test_reshard_rejects_asymmetric_same_onto_split(
        self,
        device: torch.device,
        process_mesh_dict_1x4: dict,
        process_mesh_dict_2x2: dict,
    ) -> None:
        """Resharding asymmetric ``"same"`` onto a spatial split raises before
        the weights change."""
        conv = convolution.DistributedConv2d(
            8,
            12,
            4,
            device=device,
            spatial_group=process_mesh_dict_1x4["row"],
            channel_group=process_mesh_dict_1x4["col"],
            padding="same",
            groups=4,
        )
        weight = conv.weight
        with pytest.raises(NotImplementedError, match="asymmetrically"):
            conv.reshard_parameters(process_mesh_dict_1x4, process_mesh_dict_2x2)
        assert conv.weight is weight


def _reference_count(n_spatial: int, *args: int, bias: bool) -> int:
    ref = _REF_CLS[n_spatial](*args, bias=bias)
    return sum(p.numel() for p in ref.parameters())


class TestConvParameterCount:
    """``get_parameter_count`` is the global count, the same on every rank."""

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("n_spatial", [1, 2, 3])
    def test_count_is_global_with_spatial_replicas(
        self, device: torch.device, process_mesh_dict_2x2: dict, n_spatial: int
    ) -> None:
        """2 (spatial) x 2 (channel) mesh: replicated over the two spatial ranks,
        split over the two channel ranks; neither changes the count."""
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
        assert conv.get_parameter_count() == _reference_count(
            n_spatial, 8, 12, 2, bias=True
        )

    @pytest.mark.dist(4)
    def test_count_is_global_with_pure_channel_split(
        self, device: torch.device, process_mesh_dict_1x4: dict
    ) -> None:
        """No split spatial axis: ``spatial_world_size == 1``, the count is global."""
        mesh = process_mesh_dict_1x4
        conv = _CONV_CLS[2](
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
        assert conv.get_parameter_count() == _reference_count(2, 8, 16, 2, bias=True)

    @pytest.mark.dist(4)
    def test_bias_false_is_excluded_from_the_count(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """With ``bias=False`` only the weight contributes."""
        mesh = process_mesh_dict_2x2
        conv = _CONV_CLS[2](
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
        assert conv.get_parameter_count() == _reference_count(2, 8, 12, 2, bias=False)


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


class TestVmapMembersNoBias:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("n_spatial", [1, 2, 3])
    @pytest.mark.parametrize(
        ("mesh_name", "kernel", "stride", "padding"),
        [("4x1", 2, 2, 0), ("4x1", 3, 1, 1), ("2x2", 2, 2, 0)],
        ids=["local", "halo", "channel_gather"],
    )
    def test_vmap_over_weights_without_bias(
        self,
        request: pytest.FixtureRequest,
        device: torch.device,
        n_spatial: int,
        mesh_name: str,
        kernel: int,
        stride: int,
        padding: int,
    ) -> None:
        """``torch.vmap`` over a weight stack works for a layer with ``bias=False``.

        ``local`` runs ``_forward_local`` (the native batching rule), ``halo`` and
        ``channel_gather`` run ``_conv_apply`` (the ``conv_vmap`` rule with the
        members config). Each must match calling the layer once per member.
        """
        mesh = request.getfixturevalue(f"process_mesh_dict_{mesh_name}")
        conv = _CONV_CLS[n_spatial](
            4,
            6,
            kernel,
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            stride=stride,
            padding=padding,
            bias=False,
        ).to(device)
        members = 3
        weight = torch.randn(members, *conv.weight.shape, device=device)
        local_in = 4 // mesh["col"].size()
        x = torch.randn(members, 2, local_in, *([4] * n_spatial), device=device)

        def call(w: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
            return torch.func.functional_call(conv, {"weight": w}, (xi,))

        y = torch.vmap(call)(weight, x)
        y_ref = torch.stack([call(weight[m], x[m]) for m in range(members)])
        assert_close("y", y, y_ref, 1e-5, 1e-5, device, dist.group.WORLD)
