import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import jigsaw.modules.conv1d as conv1d
from jigsaw.utils_aggregate import gather_1d_results, gather_results_as_list
from tests.process_mesh import *

pytestmark = [pytest.mark.dist_required]

# Every channel-parallel exchange strategy; "all_to_all" needs a backend with
# all-to-all support and is skipped on gloo (see _skip_if_unsupported).
CHANNEL_COMMS = ["all_gather", "p2p", "all_to_all"]

Pair = int | tuple[int]


def _skip_if_unsupported(channel_comm: str) -> None:
    """Skip when the current backend cannot run ``channel_comm``."""
    if channel_comm == "all_to_all" and dist.get_backend() == "gloo":
        pytest.skip("all_to_all channel comm needs a non-gloo backend")


def _single(value: Pair) -> tuple[int]:
    """Normalise an ``int`` or length-1 tuple to a 1-element tuple."""
    return (value,) if isinstance(value, int) else (value[0],)


def _conv1d_reference_check(
    device: torch.device,
    mesh: dict,
    in_channels: int,
    out_channels: int,
    kernel_size: Pair,
    length: int,
    batch: int = 2,
    stride: Pair | None = None,
    padding: Pair = 0,
    dilation: Pair = 1,
    channel_comm: str = "all_gather",
) -> None:
    """Compare a distributed ``DistributedConv1D`` against a single-process conv1d.

    Works for any ``row.size()`` (spatial) x ``col.size()`` (channel) mesh and
    for both the no-halo regime (``stride == kernel_size``, the default) and the
    halo regime (any other stride). Rebuilds the equivalent non-distributed
    convolution from the sharded weight, bias and input, runs forward and
    backward on both, and checks the output plus every gradient (input, weight,
    bias) after gathering the distributed shards.
    """
    row_group, col_group = mesh["row"], mesh["col"]
    n_spatial, n_channel = row_group.size(), col_group.size()
    model_group = mesh["model"]
    kernel_size = _single(kernel_size)
    stride = kernel_size if stride is None else _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)

    torch.manual_seed(model_group.rank())
    x_local = torch.randn(
        batch, in_channels // n_channel, length // n_spatial, device=device
    )

    conv = conv1d.DistributedConv1D(
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
        channel_comm=channel_comm,
    ).to(device)

    # --- reconstruct the equivalent global convolution -----------------------
    weight_global = gather_1d_results(
        conv.weight.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    bias_global = gather_1d_results(
        conv.bias.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    x_global = gather_1d_results(x_local, mesh, dim=-2, pm_key="col")
    x_global = gather_1d_results(x_global, mesh, dim=-1, pm_key="row").clone()

    x_global.requires_grad_(True)
    weight_global.requires_grad_(True)
    bias_global.requires_grad_(True)

    y_ref = F.conv1d(
        x_global,
        weight_global,
        bias_global,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    torch.manual_seed(2024)
    grad_global = torch.randn(y_ref.shape, device=device)
    y_ref.backward(grad_global)

    # --- distributed forward / backward on this rank's shard ----------------
    x_dist = x_local.clone().requires_grad_(True)
    y = conv(x_dist)

    out_local = out_channels // n_channel
    assert y_ref.shape[-1] % n_spatial == 0, (
        f"global output length {y_ref.shape[-1]} not divisible by spatial "
        f"size {n_spatial}"
    )
    ol_local = y_ref.shape[-1] // n_spatial
    grad_local = grad_global[
        ...,
        col_group.rank() * out_local : (col_group.rank() + 1) * out_local,
        row_group.rank() * ol_local : (row_group.rank() + 1) * ol_local,
    ].contiguous()
    y.backward(grad_local)

    # --- gather distributed results back to global tensors ------------------
    def _to_global(t: torch.Tensor, channel_dim: int) -> torch.Tensor:
        t = gather_1d_results(t, mesh, dim=channel_dim, pm_key="col")
        return gather_1d_results(t, mesh, dim=-1, pm_key="row")

    y_global = _to_global(y.detach(), -2)
    x_grad_global = _to_global(x_dist.grad, -2)
    w_grad_global = gather_1d_results(conv.weight.grad, mesh, dim=0, pm_key="col")
    b_grad_global = gather_1d_results(conv.bias.grad, mesh, dim=0, pm_key="col")

    assert torch.allclose(y_global, y_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(x_grad_global, x_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(w_grad_global, weight_global.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(b_grad_global, bias_global.grad, atol=1e-4, rtol=1e-4)


class TestConv1d:
    # Every test here builds a 2x2 mesh, so it needs exactly 4 ranks. Without
    # this marker an 8-rank run would build a mesh over ranks 0-3 only and
    # deadlock the survivors in the first collective.
    pytestmark = pytest.mark.dist(4)

    def test_param_init_conv1d(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Initial weights are synchronized across model-parallel replicas."""
        local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
        torch.manual_seed(local_rank)
        row_comm_size, col_comm_size = (
            process_mesh_dict_2x2["row"].size(),
            process_mesh_dict_2x2["col"].size(),
        )
        conv = conv1d.DistributedConv1D(
            5,
            10,
            4,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=4,
            padding=(0,),
            dilation=1,
            groups=1,
            bias=True,
            parallelism=4,
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        weights_gathered = gather_results_as_list(conv.weight, process_mesh_dict_2x2)
        for i in range(process_mesh_dict_2x2["model"].size() // row_comm_size):
            assert torch.allclose(
                weights_gathered[i], weights_gathered[i + col_comm_size], atol=1e-5
            )

    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_conv1d_forward(
        self, device: torch.device, process_mesh_dict_2x2: dict, channel_comm: str
    ) -> None:
        """No-halo regime (stride == kernel) matches a single-process conv1d."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            length=32,
            channel_comm=channel_comm,
        )

    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_stride_lt_kernel_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict, channel_comm: str
    ) -> None:
        """Overlapping windows (stride < kernel_size) match a plain conv1d."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            length=32,
            stride=2,
            padding=1,
            channel_comm=channel_comm,
        )

    def test_stride_gt_kernel_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Strided-past-kernel windows (stride > kernel_size) match a plain conv1d."""
        _conv1d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=2,
            length=30,
            stride=4,
            padding=0,
        )

    def test_dilated_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Dilated 'same'-style conv (stride 1) matches a plain conv1d."""
        _conv1d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=3,
            length=32,
            stride=1,
            padding=2,
            dilation=2,
        )

    def test_string_padding_rejected_with_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """String padding is not supported once a halo exchange is needed."""
        conv = conv1d.DistributedConv1D(
            8,
            12,
            4,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=2,
            padding="same",
            dilation=1,
            groups=1,
            bias=True,
            parallelism=process_mesh_dict_2x2["model"].size(),
            bayesian=False,
            dtype=torch.float32,
        ).to(device)
        x = torch.randn(2, 4, 16, device=device)
        with pytest.raises(NotImplementedError):
            conv(x)

    def test_channel_comm_auto_defaults_by_device(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """``channel_comm='auto'`` resolves from the device on the config."""
        conv = conv1d.DistributedConv1D(
            8,
            12,
            4,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=4,
        ).to(device)
        expected = "all_to_all" if device.type == "cuda" else "p2p"
        assert conv.config.channel_comm == expected

    def test_unknown_channel_comm_rejected(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """An unrecognised ``channel_comm`` name fails fast at construction."""
        with pytest.raises(ValueError):
            conv1d.DistributedConv1D(
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
        conv = conv1d.DistributedConv1D(
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
        x = torch.randn(2, 4, 16, device=device)
        with pytest.raises(RuntimeError, match="all-to-all"):
            conv(x)


class TestConv1dParallel:
    """Distributed Conv1d across arbitrary spatial and channel group sizes."""

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_forward_backward_1x4(
        self, device: torch.device, process_mesh_dict_1x4: dict, channel_comm: str
    ) -> None:
        """Pure channel parallelism (4 channel ranks, no spatial split)."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            length=32,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_forward_backward_4x1(
        self, device: torch.device, process_mesh_dict_4x1: dict, channel_comm: str
    ) -> None:
        """Pure spatial parallelism (4 spatial ranks, no channel split)."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            length=32,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_halo_1x4(
        self, device: torch.device, process_mesh_dict_1x4: dict, channel_comm: str
    ) -> None:
        """Halo regime with no spatial split (channel-only, group size 1 path)."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            length=32,
            stride=2,
            padding=1,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_halo_4x1(
        self, device: torch.device, process_mesh_dict_4x1: dict, channel_comm: str
    ) -> None:
        """Halo regime with 4-way spatial split (real cross-rank exchange)."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            length=64,
            stride=2,
            padding=1,
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
        """Combined 2-way spatial and 4-way channel parallelism (no halo)."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            length=32,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_halo_2x4(
        self,
        device: torch.device,
        process_mesh_dict_2x4_rowcol: dict,
        channel_comm: str,
    ) -> None:
        """Halo regime combined with 4-way channel parallelism."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            length=32,
            stride=2,
            padding=1,
            channel_comm=channel_comm,
        )
