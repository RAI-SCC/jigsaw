import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import jigsaw.modules.convolution as convolution
from jigsaw.utils_aggregate import gather_1d_results, gather_results_as_list
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]

# Every channel-parallel exchange strategy; "all_to_all" needs a backend with
# all-to-all support and is skipped on gloo (see _skip_if_unsupported).
CHANNEL_COMMS = ["all_gather", "p2p", "all_to_all"]

Triple = int | tuple[int, int, int]


def _triple(value: Triple) -> tuple[int, int, int]:
    """Normalise an ``int`` or 3-tuple to a ``(d, h, w)`` tuple."""
    return (value, value, value) if isinstance(value, int) else tuple(value)


def _skip_if_unsupported(channel_comm: str) -> None:
    """Skip when the current backend cannot run ``channel_comm``."""
    if channel_comm == "all_to_all" and dist.get_backend() == "gloo":
        pytest.skip("all_to_all channel comm needs a non-gloo backend")


# (kernel, stride, padding, dilation) -- each entry is an int or a (d, h, w)
# tuple, so asymmetric kernels/strides/paddings/dilations are covered. The split
# axis is the width (last) axis. The last entries exercise the ``stride ==
# kernel`` config branch with non-trivial split-axis padding/dilation (must still
# route through the halo path) and a combo whose global output width is not
# divisible by an even spatial group (must raise ``NotImplementedError``).
ARBITRARY_CASES = [
    (2, 2, 0, 1),
    (3, 2, 1, 1),
    (2, 3, 1, 1),
    (3, 1, 1, 1),
    (3, 1, 2, 2),
    ((2, 3, 3), (2, 2, 1), (0, 1, 1), 1),
    ((3, 3, 4), (2, 2, 2), (1, 1, 2), 1),
    (2, 2, 1, 1),
    ((3, 3, 3), (3, 3, 3), (0, 0, 0), (1, 1, 2)),
    (3, 2, 0, 1),
]
ARBITRARY_IDS = [
    f"k{k}_s{s}_p{p}_d{d}".replace(" ", "") for (k, s, p, d) in ARBITRARY_CASES
]


def _conv3d_reference_check(
    device: torch.device,
    mesh: dict,
    in_channels: int,
    out_channels: int,
    kernel_size: Triple,
    depth: int,
    height: int,
    width: int,
    batch: int = 2,
    stride: Triple | None = None,
    padding: Triple = 0,
    dilation: Triple = 1,
    groups: int = 1,
    channel_comm: str = "all_gather",
) -> None:
    """Compare a distributed ``DistributedConv3D`` against a single-process conv3d.

    Works for any ``row.size()`` (spatial, splits the width axis) x ``col.size()``
    (channel) mesh and for both the no-halo regime (``stride == kernel_size``,
    the default) and the halo regime (any other stride, or ``stride ==
    kernel_size`` with split-axis padding/dilation). Rebuilds the equivalent
    non-distributed convolution from the sharded weight, bias and input, runs
    forward and backward on both, and checks the output plus every gradient
    (input, weight, bias) after gathering the distributed shards.

    When the global output width is not divisible by the spatial group size the
    halo exchange cannot shard the output evenly; this asserts it raises
    ``NotImplementedError`` instead of returning a wrong result.
    """
    row_group, col_group = mesh["row"], mesh["col"]
    n_spatial, n_channel = row_group.size(), col_group.size()
    model_group = mesh["model"]
    kernel_size = _triple(kernel_size)
    stride = kernel_size if stride is None else _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    torch.manual_seed(model_group.rank())
    x_local = torch.randn(
        batch,
        in_channels // n_channel,
        depth,
        height,
        width // n_spatial,
        device=device,
    )

    conv = convolution.DistributedConv3D(
        in_channels,
        out_channels,
        kernel_size,
        device=device,
        spatial_group=row_group,
        channel_group=col_group,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
        bayesian=False,
        dtype=torch.float32,
        channel_comm=channel_comm,
    ).to(device)

    # --- reconstruct the equivalent global convolution ---------------------
    # The channel group always shards the plain-conv weight's output-channel
    # axis (dim 0), grouped or not.
    weight_global = gather_1d_results(
        conv.weight.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    bias_global = gather_1d_results(
        conv.bias.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    x_global = gather_1d_results(x_local, mesh, dim=-4, pm_key="col")
    x_global = gather_1d_results(x_global, mesh, dim=-1, pm_key="row").clone()

    x_global.requires_grad_(True)
    weight_global.requires_grad_(True)
    bias_global.requires_grad_(True)

    y_ref = F.conv3d(
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

    # --- distributed forward / backward on this rank's shard --------------
    x_dist = x_local.clone().requires_grad_(True)
    y = conv(x_dist)

    out_local = out_channels // n_channel
    ow_local = y_ref.shape[-1] // n_spatial
    grad_local = grad_global[
        ...,
        col_group.rank() * out_local : (col_group.rank() + 1) * out_local,
        :,
        :,
        row_group.rank() * ow_local : (row_group.rank() + 1) * ow_local,
    ].contiguous()
    y.backward(grad_local)

    # --- gather distributed results back to global tensors ---------------
    def _to_global(t: torch.Tensor, channel_dim: int) -> torch.Tensor:
        t = gather_1d_results(t, mesh, dim=channel_dim, pm_key="col")
        return gather_1d_results(t, mesh, dim=-1, pm_key="row")

    y_global = _to_global(y.detach(), -4)
    x_grad_global = _to_global(x_dist.grad, -4)
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


class TestConv3d:
    # Every test here builds a 2x2 mesh, so it needs exactly 4 ranks. Without
    # this marker an 8-rank run would build a mesh over ranks 0-3 only and
    # deadlock the survivors in the first collective.
    pytestmark = pytest.mark.dist(4)

    def test_param_init(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Initial weights are synchronized across model-parallel replicas."""
        local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
        torch.manual_seed(local_rank)
        row_comm_size, col_comm_size = (
            process_mesh_dict_2x2["row"].size(),
            process_mesh_dict_2x2["col"].size(),
        )
        conv = convolution.DistributedConv3D(
            6,
            12,
            2,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=2,
            padding=(0,),
            dilation=1,
            groups=1,
            bias=True,
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        weights_gathered = gather_results_as_list(conv.weight, process_mesh_dict_2x2)
        for i in range(process_mesh_dict_2x2["model"].size() // row_comm_size):
            assert torch.allclose(
                weights_gathered[i], weights_gathered[i + col_comm_size], atol=1e-5
            )

    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_arbitrary_params(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
        channel_comm: str,
    ) -> None:
        """2x2 mesh: arbitrary kernel/stride/padding/dilation, incl. asymmetric."""
        _skip_if_unsupported(channel_comm)
        _conv3d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            depth=6,
            height=6,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
            channel_comm=channel_comm,
        )

    def test_halo_stride_eq_kernel_padding(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """stride == kernel with split-axis padding routes through the halo path."""
        _conv3d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=(2, 2, 2),
            depth=6,
            height=6,
            width=24,
            stride=(2, 2, 2),
            padding=(0, 0, 2),
            dilation=1,
        )

    def test_halo_stride_eq_kernel_dilation(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """stride == kernel with split-axis dilation routes through the halo path."""
        _conv3d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=(3, 3, 3),
            depth=6,
            height=6,
            width=22,
            stride=(3, 3, 3),
            padding=0,
            dilation=(1, 1, 2),
        )

    def test_string_padding_rejected_with_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """String padding is not supported once a halo exchange is needed."""
        conv = convolution.DistributedConv3D(
            8,
            12,
            3,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=2,
            padding="same",
            dilation=1,
            groups=1,
            bias=True,
            bayesian=False,
            dtype=torch.float32,
        ).to(device)
        x = torch.randn(2, 4, 6, 6, 8, device=device)
        with pytest.raises(NotImplementedError):
            conv(x)

    def test_channel_comm_auto_defaults_by_device(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """``channel_comm='auto'`` resolves from the device on the config."""
        conv = convolution.DistributedConv3D(
            8,
            12,
            2,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=2,
        ).to(device)
        expected = "all_to_all" if device.type == "cuda" else "p2p"
        assert conv.config.channel_comm == expected

    def test_unknown_channel_comm_rejected(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """An unrecognised ``channel_comm`` name fails fast at construction."""
        with pytest.raises(ValueError):
            convolution.DistributedConv3D(
                8,
                12,
                2,
                device=device,
                spatial_group=process_mesh_dict_2x2["row"],
                channel_group=process_mesh_dict_2x2["col"],
                stride=2,
                channel_comm="alltoall",
            ).to(device)

    def test_alltoall_rejected_on_gloo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """``all_to_all`` refuses to run on gloo, which has no all-to-all."""
        if dist.get_backend() != "gloo":
            pytest.skip("only meaningful on the gloo backend")
        conv = convolution.DistributedConv3D(
            8,
            12,
            2,
            device=device,
            spatial_group=process_mesh_dict_2x2["row"],
            channel_group=process_mesh_dict_2x2["col"],
            stride=2,
            padding=(0,),
            dilation=1,
            bias=True,
            channel_comm="all_to_all",
        ).to(device)
        x = torch.randn(2, 4, 6, 6, 8, device=device)
        with pytest.raises(RuntimeError, match="all-to-all"):
            conv(x)


class TestConv3dParallel:
    """Distributed Conv3d across arbitrary spatial and channel group sizes."""

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_arbitrary_params_1x4(
        self,
        device: torch.device,
        process_mesh_dict_1x4: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
        channel_comm: str,
    ) -> None:
        """Pure channel parallelism: every param combo works (no spatial split)."""
        _skip_if_unsupported(channel_comm)
        _conv3d_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            depth=6,
            height=6,
            width=24,
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
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
        channel_comm: str,
    ) -> None:
        """Pure 4-way spatial split with arbitrary params (real halo exchange)."""
        _skip_if_unsupported(channel_comm)
        _conv3d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            depth=6,
            height=6,
            width=48,
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
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
        channel_comm: str,
    ) -> None:
        """2-way spatial x 4-way channel with arbitrary params."""
        _skip_if_unsupported(channel_comm)
        _conv3d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            depth=6,
            height=6,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
            channel_comm=channel_comm,
        )


# ---------------------------------------------------------------------------
# multi-spatial-dim parallelism: the domain is split along depth / height /
# width simultaneously.
# ---------------------------------------------------------------------------
def _conv3d_multi_spatial_reference_check(
    device: torch.device,
    mesh: dict,
    in_channels: int,
    out_channels: int,
    kernel_size: Triple,
    depth: int,
    height: int,
    width: int,
    batch: int = 2,
    stride: Triple | None = None,
    padding: Triple = 0,
    dilation: Triple = 1,
    groups: int = 1,
) -> None:
    """Compare a ``DistributedConv3D`` split along D/H/W to a plain conv3d.

    ``mesh`` carries ``sd`` (depth), ``sh`` (height), ``sw`` (width) and ``col``
    (channel) groups plus a ``model`` group over every rank. Any spatial group
    may be size 1 (that axis is then not split). Covers the no-halo regime
    (``stride == kernel_size``) and the halo regime, and checks the output plus
    every gradient after gathering the sharded results.
    """
    d_group, h_group, w_group, c_group = (
        mesh["sd"],
        mesh["sh"],
        mesh["sw"],
        mesh["col"],
    )
    model_group = mesh["model"]
    nd, nh, nw, nc = (
        d_group.size(),
        h_group.size(),
        w_group.size(),
        c_group.size(),
    )
    kernel_size = _triple(kernel_size)
    stride = kernel_size if stride is None else _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    torch.manual_seed(model_group.rank())
    x_local = torch.randn(
        batch,
        in_channels // nc,
        depth // nd,
        height // nh,
        width // nw,
        device=device,
    )

    conv = convolution.DistributedConv3D(
        in_channels,
        out_channels,
        kernel_size,
        device=device,
        spatial_group=[d_group, h_group, w_group],
        channel_group=c_group,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
        bayesian=False,
        dtype=torch.float32,
    ).to(device)

    # When a halo exchange is needed, every split axis's global output must be
    # divisible by that axis's group size; otherwise the module rejects the call
    # cleanly rather than returning a wrong result. Predict that and check the
    # error path instead of the numeric one.
    split_idxs = [i for i, n in enumerate((nd, nh, nw)) if n > 1]
    groups_per_axis = (nd, nh, nw)
    global_sizes = (depth, height, width)
    halo = not all(k == s for k, s in zip(kernel_size, stride)) or any(
        padding[i] != 0 or dilation[i] != 1 for i in split_idxs
    )
    if halo:
        for i in split_idxs:
            ext = dilation[i] * (kernel_size[i] - 1) + 1
            g_out = (global_sizes[i] + 2 * padding[i] - ext) // stride[i] + 1
            if g_out % groups_per_axis[i] != 0:
                with pytest.raises(NotImplementedError):
                    conv(x_local.clone().requires_grad_(True))
                return

    def _gather_spatial(t: torch.Tensor) -> torch.Tensor:
        t = gather_1d_results(t, mesh, dim=-4, pm_key="col")
        t = gather_1d_results(t, mesh, dim=-3, pm_key="sd")
        t = gather_1d_results(t, mesh, dim=-2, pm_key="sh")
        return gather_1d_results(t, mesh, dim=-1, pm_key="sw")

    weight_global = gather_1d_results(
        conv.weight.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    bias_global = gather_1d_results(
        conv.bias.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    x_global = _gather_spatial(x_local).clone()

    x_global.requires_grad_(True)
    weight_global.requires_grad_(True)
    bias_global.requires_grad_(True)

    y_ref = F.conv3d(
        x_global,
        weight_global,
        bias_global,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    assert (
        y_ref.shape[-3] % nd == 0
        and y_ref.shape[-2] % nh == 0
        and y_ref.shape[-1] % nw == 0
    ), (
        f"global output {tuple(y_ref.shape[-3:])} not divisible by spatial "
        f"grid ({nd}, {nh}, {nw})"
    )

    torch.manual_seed(2024)
    grad_global = torch.randn(y_ref.shape, device=device)
    y_ref.backward(grad_global)

    x_dist = x_local.clone().requires_grad_(True)
    y = conv(x_dist)

    oc_local = out_channels // nc
    od_local = y_ref.shape[-3] // nd
    oh_local = y_ref.shape[-2] // nh
    ow_local = y_ref.shape[-1] // nw
    grad_local = grad_global[
        ...,
        c_group.rank() * oc_local : (c_group.rank() + 1) * oc_local,
        d_group.rank() * od_local : (d_group.rank() + 1) * od_local,
        h_group.rank() * oh_local : (h_group.rank() + 1) * oh_local,
        w_group.rank() * ow_local : (w_group.rank() + 1) * ow_local,
    ].contiguous()
    y.backward(grad_local)

    y_global = _gather_spatial(y.detach())
    x_grad_global = _gather_spatial(x_dist.grad)
    w_grad_global = gather_1d_results(conv.weight.grad, mesh, dim=0, pm_key="col")
    b_grad_global = gather_1d_results(conv.bias.grad, mesh, dim=0, pm_key="col")

    # Collective-safe: every rank reaches the same verdict, so a mismatch on a
    # subset of ranks fails the job instead of hanging the rest in the next gather.
    assert_close("output", y_global, y_ref, 1e-4, 1e-4, device, model_group)
    assert_close(
        "input grad", x_grad_global, x_global.grad, 1e-4, 1e-4, device, model_group
    )
    # The parameter grads reduce over every output position; across a big volume
    # the distributed partial sums land in a different float32 order than the
    # single-process reference, so allow ~1e-3 there (a real bug is orders of
    # magnitude larger -- cf. the exact ``y`` / ``g_x`` checks above).
    assert_close(
        "weight grad",
        w_grad_global,
        weight_global.grad,
        1e-3,
        1e-3,
        device,
        model_group,
    )
    assert_close(
        "bias grad", b_grad_global, bias_global.grad, 1e-3, 1e-3, device, model_group
    )


class TestConv3dGroups:
    """Grouped 3-D convolution across spatial / channel group sizes.

    ``groups`` a multiple of ``col.size()`` -> every channel-parallel rank owns
    whole conv-groups and runs a plain grouped conv3d on its channel shard with
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
        _conv3d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=2,
            depth=8,
            height=8,
            width=16,
            groups=groups,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [2, 4])
    def test_2x2_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict, groups: int
    ) -> None:
        """Channel-local grouped conv3d through the halo path (stride < kernel)."""
        _conv3d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=3,
            depth=8,
            height=8,
            width=16,
            stride=2,
            padding=1,
            groups=groups,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        [
            ((2, 3, 3), (2, 2, 1), (0, 1, 1), 1),
            (3, 1, 2, (1, 1, 2)),
        ],
        ids=["asymmetric", "dilated"],
    )
    def test_2x2_asymmetric_dilated(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """Channel-local grouped conv3d with asymmetric / dilated hyperparams."""
        _conv3d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            depth=8,
            height=8,
            width=16,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=2,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [1, 2, 3, 4, 6, 12])
    def test_pure_spatial_4x1(
        self, device: torch.device, process_mesh_dict_4x1: dict, groups: int
    ) -> None:
        """No channel split: any ``groups`` dividing the channel counts."""
        _conv3d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=12,
            out_channels=12,
            kernel_size=2,
            depth=8,
            height=8,
            width=16,
            groups=groups,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [1, 4, 12])
    def test_pure_channel_1x4(
        self, device: torch.device, process_mesh_dict_1x4: dict, groups: int
    ) -> None:
        """4-way channel split: ``groups`` 1 (gather) or a multiple of 4 (local)."""
        _conv3d_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=12,
            out_channels=12,
            kernel_size=2,
            depth=4,
            height=4,
            width=8,
            groups=groups,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("groups", [4, 8])
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_2x4_no_halo(
        self,
        device: torch.device,
        process_mesh_dict_2x4_rowcol: dict,
        groups: int,
        channel_comm: str,
    ) -> None:
        """2-way spatial x 4-way channel, channel-local grouped conv3d."""
        _skip_if_unsupported(channel_comm)
        _conv3d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=16,
            kernel_size=2,
            depth=8,
            height=8,
            width=16,
            groups=groups,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [1, 2])
    def test_dhw_split_no_halo(
        self, device: torch.device, spatial_mesh_2x2x1_dhw: dict, groups: int
    ) -> None:
        """D/H split, no channel split: grouped conv over a multi-axis domain."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw,
            in_channels=8,
            out_channels=16,
            kernel_size=2,
            depth=16,
            height=16,
            width=8,
            groups=groups,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("groups", [2, 4])
    def test_dhw_split_plus_channel_halo(
        self, device: torch.device, spatial_mesh_2x2x1_dhw_x2c: dict, groups: int
    ) -> None:
        """D/H split + 2-way channel split, channel-local grouped conv3d, halo."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=3,
            depth=16,
            height=16,
            width=8,
            stride=2,
            padding=1,
            groups=groups,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [3, 9])
    def test_incompatible_groups_rejected(
        self, device: torch.device, process_mesh_dict_2x2: dict, groups: int
    ) -> None:
        """``1 < groups`` not a multiple of ``col.size()`` fails at construction."""
        with pytest.raises(NotImplementedError):
            convolution.DistributedConv3D(
                18,
                18,
                2,
                device=device,
                spatial_group=process_mesh_dict_2x2["row"],
                channel_group=process_mesh_dict_2x2["col"],
                stride=2,
                groups=groups,
            ).to(device)


class TestConv3dMultiSpatial:
    """Distributed Conv3d with the domain split along several spatial axes."""

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"), ARBITRARY_CASES, ids=ARBITRARY_IDS
    )
    def test_dh_split_arbitrary_params(
        self,
        device: torch.device,
        spatial_mesh_2x2x1_dhw: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """2x2 (depth, height) split: every kernel/stride/padding/dilation combo.

        Each combo is either matched against a single-process conv3d or (when
        its global output is not divisible by the spatial group) asserted to be
        rejected with ``NotImplementedError`` -- never silently wrong.
        """
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            depth=12,
            height=12,
            width=6,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(4)
    def test_dh_split_no_halo(
        self, device: torch.device, spatial_mesh_2x2x1_dhw: dict
    ) -> None:
        """2 (depth) x 2 (height) split, stride == kernel (no communication)."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=2,
            depth=8,
            height=8,
            width=8,
        )

    @pytest.mark.dist(4)
    def test_dh_split_halo(
        self, device: torch.device, spatial_mesh_2x2x1_dhw: dict
    ) -> None:
        """2 (depth) x 2 (height) split, overlapping windows on both axes."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=3,
            depth=12,
            height=12,
            width=12,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(4)
    def test_dw_split_halo(
        self, device: torch.device, spatial_mesh_2x1x2_dhw: dict
    ) -> None:
        """Depth and width split, height not split, overlapping windows."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x1x2_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=3,
            depth=12,
            height=6,
            width=12,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(4)
    def test_dh_split_halo_asymmetric(
        self, device: torch.device, spatial_mesh_2x2x1_dhw: dict
    ) -> None:
        """2x2 (D, H) split, non-square kernel/stride/padding, stride != kernel."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=(2, 3, 3),
            depth=12,
            height=12,
            width=12,
            stride=(2, 2, 1),
            padding=(0, 1, 1),
        )

    @pytest.mark.dist(8)
    def test_dhw_split_no_halo(
        self, device: torch.device, spatial_mesh_2x2x2_dhw: dict
    ) -> None:
        """Full 2x2x2 depth/height/width split, stride == kernel."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x2_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=2,
            depth=8,
            height=8,
            width=8,
        )

    @pytest.mark.dist(8)
    def test_dhw_split_halo(
        self, device: torch.device, spatial_mesh_2x2x2_dhw: dict
    ) -> None:
        """Full 2x2x2 depth/height/width split, overlapping windows on all axes."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x2_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=3,
            depth=12,
            height=12,
            width=12,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(8)
    def test_dh_split_plus_channel_halo(
        self, device: torch.device, spatial_mesh_2x2x1_dhw_x2c: dict
    ) -> None:
        """2x2 (D, H) split combined with 2-way channel parallelism."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            depth=16,
            height=16,
            width=8,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(4)
    def test_wrong_length_sequence_rejected(
        self, device: torch.device, spatial_mesh_2x2x1_dhw: dict
    ) -> None:
        """A spatial_group sequence must have one entry per spatial axis."""
        with pytest.raises(ValueError):
            convolution.DistributedConv3D(
                6,
                12,
                2,
                device=device,
                spatial_group=[
                    spatial_mesh_2x2x1_dhw["sd"],
                    spatial_mesh_2x2x1_dhw["sh"],
                ],
                channel_group=spatial_mesh_2x2x1_dhw["col"],
                stride=2,
            ).to(device)


# (kernel, stride, padding, dilation) -- an int or a (d, h, w) tuple per entry.
# Spans exact tiling (no halo), same-conv, overlapping windows (stride < kernel),
# a gap between windows (stride > kernel), stride == kernel forced onto the halo
# path by padding, *different padding per axis*, per-axis dilation, and a fully
# asymmetric kernel/stride/padding. Every entry is sized to run real numerics on
# the 3-way-width grids; the helper still asserts a clean ``NotImplementedError``
# for any (grid, case) whose global output is not divisible by a split group.
GRID_SWEEP_CASES_3D = [
    (2, 2, 0, 1),
    (4, 4, 0, 1),
    (3, 1, 1, 1),
    (3, 2, 1, 1),
    (3, 4, 1, 1),
    (3, 1, (0, 2, 1), 1),
    (3, 1, (2, 0, 4), 1),
    (3, 1, (1, 1, 2), (1, 1, 2)),
    ((2, 3, 4), (2, 2, 2), (0, 1, 1), 1),
]
GRID_SWEEP_IDS_3D = [
    f"k{k}_s{s}_p{p}_d{d}".replace(" ", "") for (k, s, p, d) in GRID_SWEEP_CASES_3D
]


class TestConv3dSpatialGridSweep:
    """Arbitrary per-axis kernel/stride/padding/dilation over a range of grids.

    Covers ``(n_d, n_h, n_w)`` grids with a size-1 axis and with every axis
    split, both with no channel parallelism and with a 2-way channel split.
    Every (grid, case) pair is either matched against a single-process conv3d
    or -- when a split axis's global output is not divisible by that axis's
    group size -- asserted to be rejected with ``NotImplementedError``, never
    silently wrong. ``depth`` / ``height`` / ``width`` are multiples of the grid
    so the domain shards evenly on every axis.
    """

    @pytest.mark.dist(2)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        GRID_SWEEP_CASES_3D,
        ids=GRID_SWEEP_IDS_3D,
    )
    def test_split_2x1x1(
        self,
        device: torch.device,
        spatial_mesh_2x1x1_dhw: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """Only one of the three axes split: 2 (depth), height/width not split."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x1x1_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            depth=16,
            height=16,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        GRID_SWEEP_CASES_3D,
        ids=GRID_SWEEP_IDS_3D,
    )
    def test_split_2x2x1(
        self,
        device: torch.device,
        spatial_mesh_2x2x1_dhw: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """2 (depth) x 2 (height) grid, width not split, no channel split."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            depth=16,
            height=16,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(6)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        GRID_SWEEP_CASES_3D,
        ids=GRID_SWEEP_IDS_3D,
    )
    def test_split_2x1x3(
        self,
        device: torch.device,
        spatial_mesh_2x1x3_dhw: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """2 (depth) x 3 (width) grid, height not split, no channel split."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x1x3_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            depth=16,
            height=16,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(12)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        GRID_SWEEP_CASES_3D,
        ids=GRID_SWEEP_IDS_3D,
    )
    def test_split_2x2x3(
        self,
        device: torch.device,
        spatial_mesh_2x2x3_dhw: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """2 (depth) x 2 (height) x 3 (width) grid, no channel split."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x3_dhw,
            in_channels=6,
            out_channels=12,
            kernel_size=kernel,
            depth=16,
            height=16,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        GRID_SWEEP_CASES_3D,
        ids=GRID_SWEEP_IDS_3D,
    )
    def test_split_2x2x1_x2c(
        self,
        device: torch.device,
        spatial_mesh_2x2x1_dhw_x2c: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """2 (depth) x 2 (height) grid + 2-way channel split."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x1_dhw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            depth=16,
            height=16,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(12)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        GRID_SWEEP_CASES_3D,
        ids=GRID_SWEEP_IDS_3D,
    )
    def test_split_2x1x3_x2c(
        self,
        device: torch.device,
        spatial_mesh_2x1x3_dhw_x2c: dict,
        kernel: Triple,
        stride: Triple,
        padding: Triple,
        dilation: Triple,
    ) -> None:
        """2 (depth) x 3 (width) grid, height not split, + 2-way channel split."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x1x3_dhw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            depth=16,
            height=16,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @pytest.mark.dist(24)
    def test_split_2x2x3_x2c(
        self, device: torch.device, spatial_mesh_2x2x3_dhw_x2c: dict
    ) -> None:
        """2 x 2 x 3 spatial grid + 2-way channel split (world 24), mixed padding."""
        _conv3d_multi_spatial_reference_check(
            device,
            spatial_mesh_2x2x3_dhw_x2c,
            in_channels=8,
            out_channels=12,
            kernel_size=3,
            depth=16,
            height=16,
            width=24,
            stride=1,
            padding=(0, 2, 1),
            dilation=1,
        )
