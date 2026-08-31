import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import jigsaw.modules.conv as conv
import jigsaw.modules.conv_transpose as conv_transpose
from jigsaw.utils_aggregate import gather_1d_results, gather_results_as_list
from tests.process_mesh import *

pytestmark = [pytest.mark.dist_required]

# Every channel-parallel exchange strategy; "all_to_all" needs a backend with
# all-to-all support and is skipped on gloo (see _skip_if_unsupported).
CHANNEL_COMMS = ["all_gather", "ring", "all_to_all"]


def _skip_if_unsupported(channel_comm: str) -> None:
    """Skip when the current backend cannot run ``channel_comm``."""
    if channel_comm == "all_to_all" and dist.get_backend() == "gloo":
        pytest.skip("all_to_all channel comm needs a non-gloo backend")


def _conv2d_reference_check(
    device: torch.device,
    mesh: dict,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    height: int,
    width: int,
    batch: int = 2,
    channel_comm: str = "all_gather",
) -> None:
    """Compare a distributed ``DistributedConv2D`` against a single-process conv2d.

    Works for any ``row.size()`` (spatial) x ``col.size()`` (channel) mesh and
    either channel-parallel exchange strategy (``"all_gather"`` / ``"ring"``).
    Rebuilds the equivalent non-distributed convolution from the sharded weight,
    bias and input, runs both forward and backward, and checks the output plus
    every gradient (input, weight, bias) after gathering the distributed shards.
    """
    row_group, col_group = mesh["row"], mesh["col"]
    n_spatial, n_channel = row_group.size(), col_group.size()
    model_group = mesh["model"]

    torch.manual_seed(model_group.rank())
    x_local = torch.randn(
        batch, in_channels // n_channel, height, width // n_spatial, device=device
    )

    conv2d = conv.DistributedConv2D(
        in_channels,
        out_channels,
        kernel_size,
        device=device,
        spatial_group=row_group,
        channel_group=col_group,
        stride=kernel_size,
        padding=(0,),
        dilation=1,
        groups=1,
        bias=True,
        parallelism=model_group.size(),
        bayesian=False,
        dtype=torch.float32,
        channel_comm=channel_comm,
    ).to(device)

    # --- reconstruct the equivalent global convolution -----------------------
    weight_global = gather_1d_results(
        conv2d.weight.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    bias_global = gather_1d_results(
        conv2d.bias.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    x_global = gather_1d_results(x_local, mesh, dim=-3, pm_key="col")
    x_global = gather_1d_results(x_global, mesh, dim=-1, pm_key="row").clone()

    x_global.requires_grad_(True)
    weight_global.requires_grad_(True)
    bias_global.requires_grad_(True)

    y_ref = F.conv2d(x_global, weight_global, bias_global, stride=kernel_size)

    torch.manual_seed(2024)
    grad_global = torch.randn(y_ref.shape, device=device)
    y_ref.backward(grad_global)

    # --- distributed forward / backward on this rank's shard ----------------
    x_dist = x_local.clone().requires_grad_(True)
    y = conv2d(x_dist)

    out_local = out_channels // n_channel
    ow_local = y_ref.shape[-1] // n_spatial
    grad_local = grad_global[
        ...,
        col_group.rank() * out_local : (col_group.rank() + 1) * out_local,
        :,
        row_group.rank() * ow_local : (row_group.rank() + 1) * ow_local,
    ].contiguous()
    y.backward(grad_local)

    # --- gather distributed results back to global tensors ------------------
    def _to_global(t: torch.Tensor, channel_dim: int) -> torch.Tensor:
        t = gather_1d_results(t, mesh, dim=channel_dim, pm_key="col")
        return gather_1d_results(t, mesh, dim=-1, pm_key="row")

    y_global = _to_global(y.detach(), -3)
    x_grad_global = _to_global(x_dist.grad, -3)
    w_grad_global = gather_1d_results(conv2d.weight.grad, mesh, dim=0, pm_key="col")
    b_grad_global = gather_1d_results(conv2d.bias.grad, mesh, dim=0, pm_key="col")

    assert torch.allclose(y_global, y_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(x_grad_global, x_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(w_grad_global, weight_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(b_grad_global, bias_global.grad, atol=1e-4, rtol=1e-4)


class TestConv2d:
    def test_param_init_conv2d(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Verify initial weights for Conv2d are correctly synchronized."""
        local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
        torch.manual_seed(local_rank)  # each process gets a different initial seed
        row_comm_size, col_comm_size = (
            process_mesh_dict_2x2["row"].size(),
            process_mesh_dict_2x2["col"].size(),
        )
        in_dim, out_dim = 5, 10
        kernel_size = 4
        conv2d = conv.DistributedConv2D(
            in_dim,
            out_dim,
            kernel_size,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=kernel_size,
            padding=(0,),
            dilation=1,
            groups=1,
            bias=True,
            parallelism=4,
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        weights_gathered_as_list = gather_results_as_list(
            conv2d.weight, process_mesh_dict_2x2
        )

        for i in range(process_mesh_dict_2x2["model"].size() // row_comm_size):
            assert torch.allclose(
                weights_gathered_as_list[i],
                weights_gathered_as_list[i + col_comm_size],
                atol=1e-5,
            )

    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_conv2d_forward(
        self, device: torch.device, process_mesh_dict_2x2: dict, channel_comm: str
    ) -> None:
        """2-way spatial x 2-way channel matches a single-process conv2d."""
        _skip_if_unsupported(channel_comm)
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            channel_comm=channel_comm,
        )

    @pytest.mark.parametrize("channel_comm", ["ring", "all_to_all"])
    def test_streaming_matches_all_gather(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        channel_comm: str,
    ) -> None:
        """The streaming exchanges match all-gather up to accumulation order.

        ``ring`` / ``all_to_all`` sum one ``conv`` per input-channel slice while
        all-gather runs a single fused ``conv``, so the results differ only by
        float reduction order.
        """
        _skip_if_unsupported(channel_comm)
        row_group, col_group = (
            process_mesh_dict_2x2["row"],
            process_mesh_dict_2x2["col"],
        )
        model_group = process_mesh_dict_2x2["model"]
        n_spatial, n_channel = row_group.size(), col_group.size()

        def _build(channel_comm: str) -> torch.nn.Module:
            torch.manual_seed(1234)  # identical weight init for both variants
            return conv.DistributedConv2D(
                8,
                12,
                4,
                device=device,
                spatial_group=row_group,
                channel_group=col_group,
                stride=4,
                padding=(0,),
                dilation=1,
                groups=1,
                bias=True,
                parallelism=model_group.size(),
                bayesian=False,
                dtype=torch.float32,
                channel_comm=channel_comm,
            ).to(device)

        conv_ag = _build("all_gather")
        conv_stream = _build(channel_comm)

        torch.manual_seed(model_group.rank())
        x = torch.randn(2, 8 // n_channel, 16, 16 // n_spatial, device=device)

        y_ag = conv_ag(x.clone().requires_grad_(True))
        y_stream = conv_stream(x.clone().requires_grad_(True))
        assert torch.allclose(y_ag, y_stream, atol=1e-5, rtol=1e-5)

        y_ag.backward(torch.ones_like(y_ag))
        y_stream.backward(torch.ones_like(y_stream))
        assert torch.allclose(
            conv_ag.weight.grad, conv_stream.weight.grad, atol=1e-5, rtol=1e-5
        )
        assert torch.allclose(
            conv_ag.bias.grad, conv_stream.bias.grad, atol=1e-5, rtol=1e-5
        )

    def test_unknown_channel_comm_rejected(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """An unrecognised ``channel_comm`` name fails fast at construction."""
        with pytest.raises(ValueError):
            conv.DistributedConv2D(
                8,
                12,
                4,
                device=device,
                spatial_group=process_mesh_dict_2x2["row"],
                channel_group=process_mesh_dict_2x2["col"],
                stride=4,
                channel_comm="alltoall",
            ).to(device)

    def test_alltoall_rejected_on_gloo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """``all_to_all`` refuses to run on gloo, which has no all-to-all."""
        if dist.get_backend() != "gloo":
            pytest.skip("only meaningful on the gloo backend")
        conv2d = conv.DistributedConv2D(
            8,
            12,
            4,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=4,
            padding=(0,),
            dilation=1,
            bias=True,
            parallelism=process_mesh_dict_2x2["model"].size(),
            channel_comm="all_to_all",
        ).to(device)
        x = torch.randn(2, 4, 16, 8, device=device)
        with pytest.raises(RuntimeError, match="all-to-all"):
            conv2d(x)

    def test_int_args_normalized_to_pairs(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """``stride`` / ``padding`` / ``dilation`` given as ``int`` become 2-tuples."""
        conv2d = conv.DistributedConv2D(
            8,
            12,
            4,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=4,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            parallelism=process_mesh_dict_2x2["model"].size(),
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        assert conv2d.kernel_size == (4, 4)
        assert conv2d.stride == (4, 4)
        assert conv2d.padding == (0, 0)
        assert conv2d.dilation == (1, 1)

    def test_int_args_match_tuple_args(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Passing ints produces the same forward/backward as the tuple form."""
        row_group, col_group = (
            process_mesh_dict_2x2["row"],
            process_mesh_dict_2x2["col"],
        )
        model_group = process_mesh_dict_2x2["model"]
        in_channels, out_channels, kernel_size = 8, 12, 4
        n_spatial, n_channel = row_group.size(), col_group.size()

        def _build(stride: object, padding: object, dilation: object) -> torch.nn.Module:
            torch.manual_seed(1234)  # identical weight init for both variants
            return conv.DistributedConv2D(
                in_channels,
                out_channels,
                kernel_size,
                device=device,
                spatial_group=row_group,
                channel_group=col_group,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                bias=True,
                parallelism=model_group.size(),
                bayesian=False,
                dtype=torch.float32,
            ).to(device)

        conv_int = _build(kernel_size, 0, 1)
        conv_tuple = _build((kernel_size, kernel_size), (0, 0), (1, 1))

        torch.manual_seed(model_group.rank())
        x = torch.randn(
            2, in_channels // n_channel, 16, 16 // n_spatial, device=device
        )

        y_int = conv_int(x.clone().requires_grad_(True))
        y_tuple = conv_tuple(x.clone().requires_grad_(True))
        assert torch.allclose(y_int, y_tuple, atol=1e-6)

        y_int.backward(torch.ones_like(y_int))
        y_tuple.backward(torch.ones_like(y_tuple))
        assert torch.allclose(
            conv_int.weight.grad, conv_tuple.weight.grad, atol=1e-6
        )
        assert torch.allclose(conv_int.bias.grad, conv_tuple.bias.grad, atol=1e-6)

    def test_int_args_match_reference_conv2d(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """A module built with int args still matches a single-process conv2d."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
        )

    def test_raw_autograd_fn_requires_tuples(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """The ``Conv2d`` autograd Function itself assumes tuples, not ints.

        Only ``DistributedConv2D`` normalizes via ``_pair``; calling the
        underlying function with an ``int`` stride trips ``zip(kernel_size,
        stride)`` in ``Conv2d.forward``.
        """
        conv2d = conv.DistributedConv2D(
            8,
            12,
            4,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=4,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            parallelism=process_mesh_dict_2x2["model"].size(),
            bayesian=False,
            dtype=torch.float32,
        ).to(device)
        x = torch.randn(2, 4, 16, 8, device=device)

        with pytest.raises(TypeError):
            conv2d.conv2d_apply(
                x,
                conv2d.weight,
                conv2d.bias,
                conv2d.kernel_size,
                4,  # int stride instead of (4, 4)
                conv2d.padding,
                conv2d.dilation,
                conv2d.in_channels,
                conv2d.spatial_group,
                conv2d.channel_group,
                conv2d.device,
                conv2d.parallelism,
                conv2d.empty,
                conv2d.bayesian,
                conv2d.config,
            )

    def test_requires_stride_equals_kernel(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Halo-exchange convolutions (stride != kernel_size) are rejected."""
        conv2d = conv.DistributedConv2D(
            8,
            12,
            4,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=2,
            padding=(0,),
            dilation=1,
            groups=1,
            bias=True,
            parallelism=process_mesh_dict_2x2["model"].size(),
            bayesian=False,
            dtype=torch.float32,
        ).to(device)
        x = torch.randn(2, 4, 16, 8, device=device)
        with pytest.raises(NotImplementedError):
            conv2d(x)


class TestConv2dParallel:
    """Distributed Conv2d across arbitrary spatial and channel group sizes."""

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_forward_backward_1x4(
        self, device: torch.device, process_mesh_dict_1x4: dict, channel_comm: str
    ) -> None:
        """Pure channel parallelism (4 channel ranks, no spatial split)."""
        _skip_if_unsupported(channel_comm)
        _conv2d_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_forward_backward_4x1(
        self, device: torch.device, process_mesh_dict_4x1: dict, channel_comm: str
    ) -> None:
        """Pure spatial parallelism (4 spatial ranks, no channel split)."""
        _skip_if_unsupported(channel_comm)
        _conv2d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_forward_backward_2x4(
        self,
        device: torch.device,
        process_mesh_dict_2x4_rowcol: dict,
        channel_comm: str,
    ) -> None:
        """Combined 2-way spatial and 4-way channel parallelism."""
        _skip_if_unsupported(channel_comm)
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            channel_comm=channel_comm,
        )


class TestConv2dTranspose:
    def test_param_init_conv2d_transpose(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Verify initial weights for Conv2d are correctly synchronized."""
        local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
        torch.manual_seed(local_rank)  # each process gets a different initial seed
        row_comm_size, col_comm_size = (
            process_mesh_dict_2x2["row"].size(),
            process_mesh_dict_2x2["col"].size(),
        )
        in_dim, out_dim = 6, 10
        kernel_size = 4
        conv2d = conv_transpose.DistributedConvTranspose2D(
            in_dim,
            out_dim,
            kernel_size,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=kernel_size,
            padding=(0,),
            dilation=1,
            groups=1,
            bias=True,
            parallelism=4,
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        weights_gathered_as_list = gather_results_as_list(
            conv2d.weight, process_mesh_dict_2x2
        )

        for i in range(process_mesh_dict_2x2["model"].size() // row_comm_size):
            assert torch.allclose(
                weights_gathered_as_list[i],
                weights_gathered_as_list[i + col_comm_size],
                atol=1e-5,
            )

    def test_conv2d_transpose_forwardbackward(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Verify initial weights for Conv2d are correctly synchronized."""
        local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
        torch.manual_seed(local_rank)  # each process gets a different initial seed
        row_comm_size, col_comm_size = (
            process_mesh_dict_2x2["row"].size(),
            process_mesh_dict_2x2["col"].size(),
        )
        in_dim, out_dim = 6, 10
        kernel_size = 4
        h, w = 16, 16

        conv2d = conv_transpose.DistributedConvTranspose2D(
            in_dim,
            out_dim,
            kernel_size,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=kernel_size,
            padding=(0,),
            dilation=1,
            groups=1,
            bias=True,
            parallelism=4,
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        weights_gathered_as_list = gather_results_as_list(
            conv2d.weight, process_mesh_dict_2x2
        )

        for i in range(process_mesh_dict_2x2["model"].size() // row_comm_size):
            assert torch.allclose(
                weights_gathered_as_list[i],
                weights_gathered_as_list[i + col_comm_size],
                atol=1e-5,
            )

        x = torch.randn(4, in_dim // 2, h, w // 2, device=device)
        y = conv2d(x)
        y.backward(torch.ones_like(y))
        assert isinstance(y, torch.Tensor)
