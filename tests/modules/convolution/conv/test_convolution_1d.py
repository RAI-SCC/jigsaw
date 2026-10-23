import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import jigsaw.modules.convolution as convolution
from jigsaw.gather import gather_1d_results, gather_results_as_list
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]

# Every channel-parallel exchange strategy; "all_to_all" needs a backend with
# all-to-all support and is skipped on gloo (see _skip_if_unsupported).
CHANNEL_COMMS = ["all_gather", "isend", "batch_isend_irecv", "all_to_all"]

Pair = int | tuple[int]


def _skip_if_unsupported(channel_comm: str) -> None:
    """Skip when the current backend cannot run ``channel_comm``."""
    if channel_comm == "all_to_all" and dist.get_backend() == "gloo":
        pytest.skip("all_to_all channel comm needs a non-gloo backend")


def _single(value: Pair) -> tuple[int]:
    """Normalize an ``int`` or length-1 tuple to a 1-element tuple."""
    return (value,) if isinstance(value, int) else (value[0],)


# (kernel, stride, padding, dilation) along the (split) length axis. Covers the
# no-halo tiling branch, the halo branch (stride != kernel) and the ``stride ==
# kernel`` branch with non-trivial padding / dilation that must still route
# through the halo path. Combos whose global output length is not divisible by
# an even spatial group are handled by the reference check (it asserts they
# raise ``NotImplementedError`` rather than returning a wrong result).
ARBITRARY_CASES = [
    (2, 2, 0, 1),
    (3, 2, 1, 1),
    (2, 3, 1, 1),
    (3, 1, 1, 1),
    (3, 1, 2, 2),
    (4, 2, 0, 1),
    (2, 2, 1, 1),
    (3, 3, 0, 2),
    (6, 6, 0, 1),
    (3, 2, 0, 1),
]
ARBITRARY_IDS = [f"k{k}_s{s}_p{p}_d{d}" for (k, s, p, d) in ARBITRARY_CASES]


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
    groups: int = 1,
    channel_comm: str = "all_gather",
    spatial_group_as_sequence: bool = False,
) -> None:
    """Compare a distributed ``DistributedConv1d`` against a single-process conv1d.

    Works for any ``row.size()`` (spatial) x ``col.size()`` (channel) mesh and
    for both the no-halo regime (``stride == kernel_size``, the default) and the
    halo regime (any other stride). Rebuilds the equivalent non-distributed
    convolution from the sharded weight, bias and input, runs forward and
    backward on both, and checks the output plus every gradient (input, weight,
    bias) after gathering the distributed shards.

    ``spatial_group_as_sequence`` passes the domain group as a 1-element list
    ``[row_group]`` instead of a bare group, exercising the plural
    ``spatial_groups`` code path (the two forms must be equivalent for 1-D).
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

    conv = convolution.DistributedConv1d(
        in_channels,
        out_channels,
        kernel_size,
        device=device,
        spatial_group=[row_group] if spatial_group_as_sequence else row_group,
        channel_group=col_group,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
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
        groups=groups,
    )

    # global output not divisible by the spatial ranks -> unsupported, cleanly.
    if n_spatial > 1 and y_ref.shape[-1] % n_spatial != 0:
        with pytest.raises(NotImplementedError):
            conv(x_local.clone().requires_grad_(True))
        return

    torch.manual_seed(2024)
    grad_global = torch.randn(y_ref.shape, device=device)
    y_ref.backward(grad_global)

    # --- distributed forward / backward on this rank's shard ----------------
    x_dist = x_local.clone().requires_grad_(True)
    y = conv(x_dist)

    out_local = out_channels // n_channel
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

    # Collective-safe: every rank reaches the same verdict, so a mismatch on a
    # subset of ranks fails the job instead of hanging the rest in the next gather.
    assert_close("output", y_global, y_ref, 1e-4, 1e-4, device, model_group)
    assert_close(
        "input grad", x_grad_global, x_global.grad, 1e-4, 1e-4, device, model_group
    )
    assert_close(
        "weight grad",
        w_grad_global,
        weight_global.grad,
        1e-4,
        1e-4,
        device,
        model_group,
    )
    assert_close(
        "bias grad", b_grad_global, bias_global.grad, 1e-4, 1e-4, device, model_group
    )


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
        conv = convolution.DistributedConv1d(
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

    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_arbitrary_params(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int,
        channel_comm: str,
    ) -> None:
        """2x2 mesh: arbitrary kernel/stride/padding/dilation x channel_comm."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            length=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
            channel_comm=channel_comm,
        )

    def test_string_padding_rejected_with_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """String padding is not supported once a halo exchange is needed."""
        conv = convolution.DistributedConv1d(
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
            dtype=torch.float32,
        ).to(device)
        x = torch.randn(2, 4, 16, device=device)
        with pytest.raises(NotImplementedError):
            conv(x)

    def test_channel_comm_auto_defaults_by_device(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """``channel_comm='auto'`` resolves from the device on the config."""
        conv = convolution.DistributedConv1d(
            8,
            12,
            4,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=4,
        ).to(device)
        expected = "all_to_all" if device.type == "cuda" else "batch_isend_irecv"
        assert conv.config.channel_comm == expected

    def test_unknown_channel_comm_rejected(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """An unrecognized ``channel_comm`` name fails fast at construction."""
        with pytest.raises(ValueError):
            convolution.DistributedConv1d(
                8,
                12,
                4,
                device=device,
                spatial_group=process_mesh_dict_2x2["row"],
                channel_group=process_mesh_dict_2x2["col"],
                stride=4,
                channel_comm="alltoall",
            ).to(device)

    def test_alltoall_rejected_off_cuda(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """``all_to_all`` needs NCCL, so a non-CUDA module refuses it at construction."""
        if device.type == "cuda":
            pytest.skip("only meaningful off CUDA")
        with pytest.raises(ValueError, match="needs CUDA"):
            convolution.DistributedConv1d(
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
                channel_comm="all_to_all",
            )


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

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_arbitrary_params_1x4(
        self,
        device: torch.device,
        process_mesh_dict_1x4: dict,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int,
        channel_comm: str,
    ) -> None:
        """Pure 4-way channel parallelism: every param combo (no spatial split)."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            length=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_arbitrary_params_4x1(
        self,
        device: torch.device,
        process_mesh_dict_4x1: dict,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int,
        channel_comm: str,
    ) -> None:
        """Pure 4-way spatial split with arbitrary params (channel group size 1)."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            length=48,
            stride=stride,
            padding=padding,
            dilation=dilation,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_arbitrary_params_2x4(
        self,
        device: torch.device,
        process_mesh_dict_2x4_rowcol: dict,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int,
        channel_comm: str,
    ) -> None:
        """2-way spatial x 4-way channel with arbitrary params."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            length=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
            channel_comm=channel_comm,
        )


class TestConv1dSpatialGroupsSequence:
    """The ``spatial_group=[group_l]`` sequence form must match the bare form.

    1-D has a single spatial axis, so the plural ``spatial_groups`` API only ever
    carries one entry; these cases pin that it behaves identically to passing a
    bare group across every mesh shape and kernel/stride/padding/dilation combo.
    """

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        "mesh_fx", ["process_mesh_dict_2x2", "process_mesh_dict_4x1"]
    )
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_sequence_form_matches_bare(
        self,
        request: pytest.FixtureRequest,
        device: torch.device,
        mesh_fx: str,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int,
        channel_comm: str,
    ) -> None:
        """2x2 (spatial+channel) and 4x1 (pure spatial): sequence form parity."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            request.getfixturevalue(mesh_fx),
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            length=48,
            stride=stride,
            padding=padding,
            dilation=dilation,
            channel_comm=channel_comm,
            spatial_group_as_sequence=True,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_sequence_form_pure_channel(
        self, device: torch.device, process_mesh_dict_1x4: dict, channel_comm: str
    ) -> None:
        """1x4: no spatial split, ``spatial_groups`` is ``(group_size_1,)``."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=8,
            out_channels=12,
            kernel_size=3,
            length=32,
            stride=2,
            padding=1,
            channel_comm=channel_comm,
            spatial_group_as_sequence=True,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_sequence_form_2x4(
        self,
        device: torch.device,
        process_mesh_dict_2x4_rowcol: dict,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int,
        channel_comm: str,
    ) -> None:
        """2-way spatial x 4-way channel, sequence form, arbitrary params."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            length=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
            channel_comm=channel_comm,
            spatial_group_as_sequence=True,
        )


class TestConv1dGroups:
    """Grouped 1-D convolution across spatial / channel group sizes.

    ``groups`` a multiple of ``col.size()`` -> every channel-parallel rank owns
    whole conv-groups and runs a plain grouped conv1d on its channel shard with
    no channel-group communication. ``groups == 1`` keeps the full-channel-gather
    path; ``1 < groups`` not divisible by ``col.size()`` is rejected.
    """

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [1, 2, 4])
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_2x2_no_halo(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        groups: int,
        channel_comm: str,
    ) -> None:
        """2-way spatial x 2-way channel, ``stride == kernel``."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            length=32,
            groups=groups,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [2, 4])
    def test_2x2_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict, groups: int
    ) -> None:
        """Channel-local grouped conv1d through the halo path (stride < kernel)."""
        _conv1d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            length=32,
            stride=2,
            padding=1,
            groups=groups,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [1, 2, 3, 4, 6, 12])
    def test_pure_spatial_4x1(
        self, device: torch.device, process_mesh_dict_4x1: dict, groups: int
    ) -> None:
        """No channel split: any ``groups`` dividing the channel counts."""
        _conv1d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=12,
            out_channels=12,
            kernel_size=4,
            length=32,
            groups=groups,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [1, 4, 12])
    def test_pure_channel_1x4(
        self, device: torch.device, process_mesh_dict_1x4: dict, groups: int
    ) -> None:
        """4-way channel split: ``groups`` 1 (gather) or a multiple of 4 (local)."""
        _conv1d_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=12,
            out_channels=12,
            kernel_size=4,
            length=16,
            groups=groups,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        [(2, 3, 1, 1), (3, 1, 2, 2)],
        ids=["stride_gt_kernel", "dilated"],
    )
    def test_2x2_arbitrary_params(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int,
    ) -> None:
        """Channel-local grouped conv1d, stride>kernel and dilated halo regimes."""
        _conv1d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            length=32,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=2,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [2, 4])
    def test_pure_spatial_4x1_halo(
        self, device: torch.device, process_mesh_dict_4x1: dict, groups: int
    ) -> None:
        """4-way spatial split, grouped conv1d, halo regime (real cross-rank exchange)."""
        _conv1d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            length=64,
            stride=2,
            padding=1,
            groups=groups,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("groups", [4, 8])
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_2x4(
        self,
        device: torch.device,
        process_mesh_dict_2x4_rowcol: dict,
        groups: int,
        channel_comm: str,
    ) -> None:
        """2-way spatial x 4-way channel, channel-local grouped conv1d."""
        _skip_if_unsupported(channel_comm)
        _conv1d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=16,
            kernel_size=4,
            length=32,
            groups=groups,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(4)
    def test_incompatible_groups_rejected(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """An odd ``groups`` on a real channel group fails fast at construction."""
        with pytest.raises(NotImplementedError):
            convolution.DistributedConv1d(
                18,
                18,
                4,
                device=device,
                spatial_group=process_mesh_dict_2x2["row"],
                channel_group=process_mesh_dict_2x2["col"],
                stride=4,
                groups=3,
            ).to(device)
