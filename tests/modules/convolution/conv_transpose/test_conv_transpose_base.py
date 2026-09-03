"""Module-level API on :class:`_DistributedConvTransposeNd` (``conv_transpose/base.py``).

The transpose mirror of ``tests/modules/convolution/conv/test_conv_base.py``:
covers ``spatial_world_size`` / ``get_parameter_count`` and the
``convert_to_torch`` round-trip, neither of which the forward/backward reference
suites touch.
"""

import pytest
import torch

import jigsaw.modules.convolution as convolution
from jigsaw.utils_aggregate import gather_1d_results

pytestmark = [pytest.mark.dist_required]

_CONV_CLS = {
    1: convolution.DistributedConv1DTranspose,
    2: convolution.DistributedConv2DTranspose,
    3: convolution.DistributedConv3DTranspose,
}
_REF_CLS = {
    1: torch.nn.ConvTranspose1d,
    2: torch.nn.ConvTranspose2d,
    3: torch.nn.ConvTranspose3d,
}


class TestConvTransposeParameterCount:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("n_spatial", [1, 2, 3])
    def test_count_divides_by_spatial_replicas(
        self, device: torch.device, process_mesh_dict_2x2: dict, n_spatial: int
    ) -> None:
        """2 (spatial) x 2 (channel) mesh: params are replicated over the two
        spatial ranks, so the effective count halves them."""
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
        expected = (
            conv.weight.numel() // replicas + conv.bias.numel() // replicas
        )
        assert conv.get_parameter_count() == expected

    @pytest.mark.dist(4)
    def test_pure_channel_split_has_no_spatial_replicas(
        self, device: torch.device, process_mesh_dict_1x4: dict
    ) -> None:
        """No split spatial axis -> ``spatial_world_size == 1`` and nothing is divided."""
        mesh = process_mesh_dict_1x4
        conv = convolution.DistributedConv2DTranspose(
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
        assert (
            conv.get_parameter_count()
            == conv.weight.numel() + conv.bias.numel()
        )

    @pytest.mark.dist(4)
    def test_bias_false_is_excluded_from_the_count(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """With ``bias=False`` only the weight contributes."""
        mesh = process_mesh_dict_2x2
        conv = convolution.DistributedConv2DTranspose(
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
            conv.get_parameter_count()
            == conv.weight.numel() // conv.spatial_world_size
        )


class TestConvTransposeConvertToTorch:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("n_spatial", [1, 2, 3])
    def test_roundtrip_reproduces_the_distributed_forward(
        self, device: torch.device, process_mesh_dict_2x2: dict, n_spatial: int
    ) -> None:
        """``convert_to_torch`` on every rank yields a plain transpose conv whose
        output matches the gathered distributed output for the same global input."""
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
        assert tuple(torch_conv.weight.shape) == (in_c, out_c, *([k] * n_spatial))
        assert torch_conv.bias is not None

        y_ref = torch_conv(x_global)
        assert y_ref.shape == y_global.shape
        assert torch.allclose(y_global, y_ref, atol=1e-4, rtol=1e-4)
