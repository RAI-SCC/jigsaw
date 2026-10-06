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

Pair = int | tuple[int, int]


def _pair(value: Pair) -> tuple[int, int]:
    """Normalise an ``int`` or 2-tuple to a ``(h, w)`` tuple."""
    return (value, value) if isinstance(value, int) else tuple(value)


def _skip_if_unsupported(channel_comm: str) -> None:
    """Skip when the current backend cannot run ``channel_comm``."""
    if channel_comm == "all_to_all" and dist.get_backend() == "gloo":
        pytest.skip("all_to_all channel comm needs a non-gloo backend")


def _assert_weight_replicated_across_spatial(
    weight: torch.Tensor, mesh: dict, atol: float = 1e-5, rtol: float = 1e-5
) -> None:
    """Every spatial rank in a channel column must hold the same weight shard.

    Works for any ``row.size()`` x ``col.size()`` mesh. ``gather_results_as_list``
    returns one tensor per ``model``-group rank, and the row/col mesh lays ranks
    out row-major, so the shard for ``(spatial_row, channel_col)`` lives at index
    ``spatial_row * n_channel + channel_col``.

    Compares through ``tests.parity.assert_close`` rather than a bare ``allclose``:
    callers run further collectives after this returns (the multi-iteration test
    gathers again on the next loop pass), so a mismatch seen on one rank only must
    fail every rank or the others block in that next gather.
    """
    n_spatial, n_channel = mesh["row"].size(), mesh["col"].size()
    shards = gather_results_as_list(weight, mesh)
    for channel_col in range(n_channel):
        reference = shards[channel_col]  # spatial_row == 0
        for spatial_row in range(1, n_spatial):
            assert_close(
                f"weight shard at spatial_row={spatial_row}, "
                f"channel_col={channel_col} vs spatial_row=0",
                shards[spatial_row * n_channel + channel_col],
                reference,
                atol,
                rtol,
                weight.device,
                mesh["model"],
            )


def _conv2d_reference_check(
    device: torch.device,
    mesh: dict,
    in_channels: int,
    out_channels: int,
    kernel_size: Pair,
    height: int,
    width: int,
    batch: int = 2,
    stride: Pair | None = None,
    padding: Pair = 0,
    dilation: Pair = 1,
    groups: int = 1,
    channel_comm: str = "all_gather",
) -> None:
    """Compare a distributed ``DistributedConv2D`` against a single-process conv2d.

    Works for any ``row.size()`` (spatial) x ``col.size()`` (channel) mesh, any
    channel-parallel exchange strategy (``"all_gather"`` / ``"p2p"`` /
    ``"all_to_all"``), any ``groups`` compatible with the channel group (``1`` or
    a multiple of ``col.size()``) and for both the no-halo regime
    (``stride == kernel_size``, the default) and the halo regime (any other
    stride). ``kernel_size`` / ``stride`` / ``padding`` / ``dilation`` accept an
    ``int`` or a ``(h, w)`` tuple, so asymmetric kernels and strides are covered.
    Rebuilds the equivalent non-distributed convolution from the sharded weight,
    bias and input, runs both forward and backward, and checks the output plus
    every gradient (input, weight, bias) after gathering the distributed shards.
    """
    row_group, col_group = mesh["row"], mesh["col"]
    n_spatial, n_channel = row_group.size(), col_group.size()
    model_group = mesh["model"]
    kernel_size = _pair(kernel_size)
    stride = kernel_size if stride is None else _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    torch.manual_seed(model_group.rank())
    x_local = torch.randn(
        batch, in_channels // n_channel, height, width // n_spatial, device=device
    )

    conv = convolution.DistributedConv2D(
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

    # --- reconstruct the equivalent global convolution -----------------------
    # The channel group always shards the plain-conv weight's output-channel
    # axis (dim 0), grouped or not.
    weight_global = gather_1d_results(
        conv.weight.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    bias_global = gather_1d_results(
        conv.bias.detach(), mesh, dim=0, pm_key="col"
    ).clone()
    x_global = gather_1d_results(x_local, mesh, dim=-3, pm_key="col")
    x_global = gather_1d_results(x_global, mesh, dim=-1, pm_key="row").clone()

    x_global.requires_grad_(True)
    weight_global.requires_grad_(True)
    bias_global.requires_grad_(True)

    y_ref = F.conv2d(
        x_global,
        weight_global,
        bias_global,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    torch.manual_seed(2024)
    grad_global = torch.randn(y_ref.shape, device=device)
    y_ref.backward(grad_global)

    # --- distributed forward / backward on this rank's shard ----------------
    x_dist = x_local.clone().requires_grad_(True)
    y = conv(x_dist)

    out_local = out_channels // n_channel
    assert y_ref.shape[-1] % n_spatial == 0, (
        f"global output width {y_ref.shape[-1]} not divisible by spatial size {n_spatial}"
    )
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


# ``lr`` is kept small and ``iters`` short so float drift between the distributed
# (spatial-summed) gradient and the single-process one does not blow past the
# tolerances; momentum is the interesting case because it *accumulates* any
# drift, so a broken grad-sync hook diverges within a step or two.
_OPTIMIZERS = {
    "sgd": lambda params: torch.optim.SGD(params, lr=0.05),
    "sgd_momentum": lambda params: torch.optim.SGD(params, lr=0.05, momentum=0.9),
}


def _conv2d_optimizer_parity_check(
    device: torch.device,
    mesh: dict,
    in_channels: int,
    out_channels: int,
    kernel_size: Pair,
    height: int,
    width: int,
    *,
    optimizer: str = "sgd_momentum",
    iters: int = 4,
    batch: int = 2,
    stride: Pair | None = None,
    padding: Pair = 0,
    dilation: Pair = 1,
    groups: int = 1,
    channel_comm: str = "all_gather",
) -> None:
    """Run ``iters`` train steps of a ``DistributedConv2D`` and an equal
    single-process ``conv2d`` and assert they never drift apart.

    Both start from the same weights (the distributed shards, gathered) and the
    same optimizer. Every step feeds the same global input, sharded to this rank
    for the distributed module and whole for the reference, and the same
    ``grad_output``. After each step this checks, in order:

    * the weight / bias **gradients** gathered from the shards equal the
      reference's -- i.e. the ``register_post_accumulate_grad_hook`` really did
      sum the per-spatial-rank partial gradients (a missing or double hook shows
      up here as a 1/n or 2x gradient);
    * the forward output matches;
    * the weight / bias **values** match after ``optimizer.step()``; and
    * the weight shard is still replicated bit-for-bit across the spatial group,
      so the next iteration starts from a consistent state.
    """
    row_group, col_group = mesh["row"], mesh["col"]
    n_spatial, n_channel = row_group.size(), col_group.size()
    model_group = mesh["model"]

    kernel_size = _pair(kernel_size)
    stride = kernel_size if stride is None else _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    torch.manual_seed(model_group.rank())
    dist_conv = convolution.DistributedConv2D(
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

    # Single-process reference, seeded from the gathered initial shards.
    ref_weight = torch.nn.Parameter(
        gather_1d_results(dist_conv.weight.detach(), mesh, dim=0, pm_key="col").clone()
    )
    ref_bias = torch.nn.Parameter(
        gather_1d_results(dist_conv.bias.detach(), mesh, dim=0, pm_key="col").clone()
    )

    make_opt = _OPTIMIZERS[optimizer]
    opt_dist = make_opt(list(dist_conv.parameters()))
    opt_ref = make_opt([ref_weight, ref_bias])

    cr, rr = col_group.rank(), row_group.rank()
    cin_local = in_channels // n_channel
    cout_local = out_channels // n_channel
    win_local = width // n_spatial

    def _to_global(t: torch.Tensor, channel_dim: int) -> torch.Tensor:
        t = gather_1d_results(t, mesh, dim=channel_dim, pm_key="col")
        return gather_1d_results(t, mesh, dim=-1, pm_key="row")

    for it in range(iters):
        torch.manual_seed(4096 + it)
        x_global = torch.randn(batch, in_channels, height, width, device=device)

        opt_ref.zero_grad()
        x_ref = x_global.clone().requires_grad_(True)
        y_ref = F.conv2d(
            x_ref,
            ref_weight,
            ref_bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        torch.manual_seed(8192 + it)
        grad_global = torch.randn_like(y_ref)
        y_ref.backward(grad_global)

        ow_global = y_ref.shape[-1]
        assert ow_global % n_spatial == 0, (
            f"output width {ow_global} not divisible by spatial size {n_spatial}"
        )
        ow_local = ow_global // n_spatial

        x_local = (
            x_global[
                :,
                cr * cin_local : (cr + 1) * cin_local,
                :,
                rr * win_local : (rr + 1) * win_local,
            ]
            .clone()
            .requires_grad_(True)
        )
        grad_local = grad_global[
            :,
            cr * cout_local : (cr + 1) * cout_local,
            :,
            rr * ow_local : (rr + 1) * ow_local,
        ].contiguous()

        opt_dist.zero_grad()
        y = dist_conv(x_local)
        assert y.shape == (batch, cout_local, y_ref.shape[-2], ow_local)
        y.backward(grad_local)

        # (1) gradients: the spatial-group grad-sync hook must reconstruct the
        # single-process gradient exactly.
        w_grad = gather_1d_results(dist_conv.weight.grad, mesh, dim=0, pm_key="col")
        b_grad = gather_1d_results(dist_conv.bias.grad, mesh, dim=0, pm_key="col")
        assert_close(
            f"iter {it}: weight grad (grad-sync hook wrong?)",
            w_grad,
            ref_weight.grad,
            1e-4,
            1e-4,
            device,
            model_group,
        )
        assert_close(
            f"iter {it}: bias grad (grad-sync hook wrong?)",
            b_grad,
            ref_bias.grad,
            1e-4,
            1e-4,
            device,
            model_group,
        )

        # (2) forward output.
        y_global = _to_global(y.detach(), -3)
        assert_close(
            f"iter {it}: forward",
            y_global,
            y_ref.detach(),
            1e-4,
            1e-4,
            device,
            model_group,
        )

        opt_ref.step()
        opt_dist.step()

        # (3) parameters after the step, and (4) spatial replication is intact.
        _assert_weight_replicated_across_spatial(dist_conv.weight, mesh)
        w_now = gather_1d_results(dist_conv.weight.detach(), mesh, dim=0, pm_key="col")
        b_now = gather_1d_results(dist_conv.bias.detach(), mesh, dim=0, pm_key="col")
        assert_close(
            f"iter {it}: weight value after optimizer step",
            w_now,
            ref_weight.detach(),
            2e-4,
            2e-4,
            device,
            model_group,
        )
        assert_close(
            f"iter {it}: bias value after optimizer step",
            b_now,
            ref_bias.detach(),
            2e-4,
            2e-4,
            device,
            model_group,
        )


class TestConv2d:
    # Every test here builds a 2x2 mesh, so it needs exactly 4 ranks. Without
    # this marker an 8-rank run would build a mesh over ranks 0-3 only and
    # deadlock the survivors in the first collective.
    pytestmark = pytest.mark.dist(4)

    def test_param_init_conv2d(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Verify initial weights for Conv2d are correctly synchronized."""
        local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
        torch.manual_seed(local_rank)  # each process gets a different initial seed
        in_dim, out_dim = 5, 10
        kernel_size = 4
        conv = convolution.DistributedConv2D(
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
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        _assert_weight_replicated_across_spatial(conv.weight, process_mesh_dict_2x2)

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

    @pytest.mark.parametrize("channel_comm", ["p2p", "all_to_all"])
    def test_streaming_matches_all_gather(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        channel_comm: str,
    ) -> None:
        """The streaming exchanges match all-gather up to accumulation order.

        ``p2p`` / ``all_to_all`` sum one ``conv`` per input-channel slice while
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
            return convolution.DistributedConv2D(
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
        # Rank-local input, and two distributed backward passes follow: a bare
        # assert tripping on one rank would leave the others in the halo/grad
        # collectives inside those backwards.
        assert_close(
            "streaming vs all_gather forward",
            y_ag,
            y_stream,
            1e-5,
            1e-5,
            device,
            model_group,
        )

        y_ag.backward(torch.ones_like(y_ag))
        y_stream.backward(torch.ones_like(y_stream))
        assert torch.allclose(
            conv_ag.weight.grad, conv_stream.weight.grad, atol=1e-5, rtol=1e-5
        )
        assert torch.allclose(
            conv_ag.bias.grad, conv_stream.bias.grad, atol=1e-5, rtol=1e-5
        )

    def test_channel_comm_auto_defaults_by_device(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """``channel_comm='auto'`` resolves from the device on the config."""
        conv = convolution.DistributedConv2D(
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
            convolution.DistributedConv2D(
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
        conv = convolution.DistributedConv2D(
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
        ).to(device)
        x = torch.randn(2, 4, 16, 8, device=device)
        with pytest.raises(RuntimeError, match="all-to-all"):
            conv(x)

    def test_int_args_normalized_to_pairs(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """``stride`` / ``padding`` / ``dilation`` given as ``int`` become 2-tuples."""
        conv = convolution.DistributedConv2D(
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
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        assert conv.kernel_size == (4, 4)
        assert conv.stride == (4, 4)
        assert conv.padding == (0, 0)
        assert conv.dilation == (1, 1)

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

        def _build(
            stride: object, padding: object, dilation: object
        ) -> torch.nn.Module:
            torch.manual_seed(1234)  # identical weight init for both variants
            return convolution.DistributedConv2D(
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
                bayesian=False,
                dtype=torch.float32,
            ).to(device)

        conv_int = _build(kernel_size, 0, 1)
        conv_tuple = _build((kernel_size, kernel_size), (0, 0), (1, 1))

        torch.manual_seed(model_group.rank())
        x = torch.randn(2, in_channels // n_channel, 16, 16 // n_spatial, device=device)

        y_int = conv_int(x.clone().requires_grad_(True))
        y_tuple = conv_tuple(x.clone().requires_grad_(True))
        # Distributed backwards follow; see test_streaming_matches_all_gather.
        assert_close(
            "int args vs tuple args forward",
            y_int,
            y_tuple,
            1e-6,
            1e-5,
            device,
            model_group,
        )

        y_int.backward(torch.ones_like(y_int))
        y_tuple.backward(torch.ones_like(y_tuple))
        assert torch.allclose(conv_int.weight.grad, conv_tuple.weight.grad, atol=1e-6)
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

    def test_stride_lt_kernel_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Overlapping windows (stride < kernel_size) match a single-process conv2d."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
        )

    def test_stride_gt_kernel_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Strided-past-kernel windows (stride > kernel_size) match a plain conv2d."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=2,
            height=14,
            width=14,
            stride=4,
            padding=0,
        )

    def test_dilated_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Dilated 'same'-style conv (stride 1) matches a plain conv2d."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=3,
            height=16,
            width=16,
            stride=1,
            padding=2,
            dilation=2,
        )

    def test_asymmetric_kernel_stride_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """2x2 mesh, non-square kernel and stride, stride != kernel on both axes."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=(3, 4),
            height=16,
            width=16,
            stride=(2, 2),
            padding=(1, 1),
        )

    def test_string_padding_rejected_with_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """String padding is not supported once a halo exchange is needed."""
        conv = convolution.DistributedConv2D(
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
            bayesian=False,
            dtype=torch.float32,
        ).to(device)
        x = torch.randn(2, 4, 16, 8, device=device)
        with pytest.raises(NotImplementedError):
            conv(x)


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

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_forward_backward_4x2(
        self, device: torch.device, process_mesh_dict_4x2: dict, channel_comm: str
    ) -> None:
        """4-way spatial x 2-way channel parallelism."""
        _skip_if_unsupported(channel_comm)
        _conv2d_reference_check(
            device,
            process_mesh_dict_4x2,
            in_channels=12,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(6)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_forward_backward_2x3(
        self, device: torch.device, process_mesh_dict_2x3: dict, channel_comm: str
    ) -> None:
        """Rectangular non-power-of-two mesh: 2-way spatial x 3-way channel."""
        _skip_if_unsupported(channel_comm)
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x3,
            in_channels=12,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=24,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(6)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_forward_backward_3x2(
        self, device: torch.device, process_mesh_dict_3x2: dict, channel_comm: str
    ) -> None:
        """Rectangular non-power-of-two mesh: 3-way spatial x 2-way channel."""
        _skip_if_unsupported(channel_comm)
        _conv2d_reference_check(
            device,
            process_mesh_dict_3x2,
            in_channels=12,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=24,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(6)
    def test_param_init_replicated_over_3_spatial_ranks(
        self, device: torch.device, process_mesh_dict_3x2: dict
    ) -> None:
        """Weight init is broadcast across a spatial group larger than two."""
        torch.manual_seed(dist.get_rank(group=process_mesh_dict_3x2["model"]))
        conv = convolution.DistributedConv2D(
            12,
            12,
            4,
            device=device,
            spatial_group=process_mesh_dict_3x2["row"],
            channel_group=process_mesh_dict_3x2["col"],
            stride=4,
            padding=(0,),
            dilation=1,
            bias=True,
            bayesian=False,
            dtype=torch.float32,
        ).to(device)
        _assert_weight_replicated_across_spatial(conv.weight, process_mesh_dict_3x2)

    @pytest.mark.dist(4)
    def test_halo_1x4(self, device: torch.device, process_mesh_dict_1x4: dict) -> None:
        """Halo regime with no spatial split (channel-only, group size 1 path)."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(4)
    def test_halo_4x1(self, device: torch.device, process_mesh_dict_4x1: dict) -> None:
        """Halo regime with 4-way spatial split (real cross-rank exchange)."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=32,
            width=32,
            stride=2,
            padding=1,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("padding", [(3, 1), (1, 3), (3, 3), (0, 1)])
    def test_halo_4x1_padding_split_over_width_only(
        self,
        device: torch.device,
        process_mesh_dict_4x1: dict,
        padding: tuple[int, int],
    ) -> None:
        """4-way spatial (width) split with ``pad_h != pad_w``.

        The spatial split is along width only, so the local convolution keeps the
        *full* height padding on every rank while the width padding is
        materialised only at the two global edges: rank 0 gets ``pad_w`` zero
        columns on its left, rank 3 gets ``pad_w`` on its right, ranks 1 and 2
        get none. Parity with a single-process ``F.conv2d(..., padding=padding)``
        pins that distribution down -- a wrong local padding (``pad_w`` on every
        rank, or the edge columns dropped) breaks the output and the gradients.

        ``kernel_size=3, stride=1`` is used because a 4-way width split with
        ``stride=2`` forces the global output width to be divisible by 4, which
        rules out odd ``pad_w`` (``_needed_input_span`` raises).
        """
        _conv2d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=3,
            height=16,
            width=16,
            stride=1,
            padding=padding,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_halo_2x4(
        self,
        device: torch.device,
        process_mesh_dict_2x4_rowcol: dict,
        channel_comm: str,
    ) -> None:
        """Halo regime combined with 4-way channel parallelism, every channel comm.

        Exercises the halo path's channel-parallel exchange: ``"all_gather"``
        gathers then halo-gathers once, ``"p2p"`` / ``"all_to_all"`` stream and
        halo-gather one channel slab at a time.
        """
        _skip_if_unsupported(channel_comm)
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(8)
    def test_halo_2x4_asymmetric_kernel_stride(
        self, device: torch.device, process_mesh_dict_2x4_rowcol: dict
    ) -> None:
        """2x4 mesh, non-square kernel and stride, stride < kernel on both axes."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=(3, 4),
            height=17,
            width=16,
            stride=(2, 3),
            padding=(1, 2),
        )

    @pytest.mark.dist(8)
    def test_halo_2x4_asymmetric_stride_gt_kernel(
        self, device: torch.device, process_mesh_dict_2x4_rowcol: dict
    ) -> None:
        """2x4 mesh, non-square kernel and stride, width stride > kernel."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=(4, 2),
            height=16,
            width=18,
            stride=(2, 3),
            padding=(1, 0),
        )

    @pytest.mark.dist(8)
    def test_halo_2x4_asymmetric_dilation(
        self, device: torch.device, process_mesh_dict_2x4_rowcol: dict
    ) -> None:
        """2x4 mesh, non-square stride and dilation over the halo exchange."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=(3, 3),
            height=16,
            width=16,
            stride=(1, 2),
            padding=(2, 1),
            dilation=(2, 1),
        )


class TestConv2dChannelGroupOne:
    """``channel_group.size() == 1`` bypasses the custom kernel for native conv2d."""

    @pytest.mark.dist(4)
    def test_fully_local_allows_arbitrary_hyperparams(
        self, device: torch.device, process_mesh_dict_4x1x1: dict
    ) -> None:
        """No channel *and* no spatial split: any stride / padding / dilation.

        Previously the custom kernel rejected ``stride != kernel_size``
        unconditionally; with both groups size one it now runs plain autograd
        ``conv2d`` and must match a single-process reference in fwd and bwd.
        """
        mesh = process_mesh_dict_4x1x1
        row, col = mesh["row"], mesh["col"]
        assert row.size() == 1 and col.size() == 1

        in_ch, out_ch, k = 6, 8, 3
        conv = convolution.DistributedConv2D(
            in_ch,
            out_ch,
            k,
            device=device,
            spatial_group=row,
            channel_group=col,
            stride=2,
            padding=1,
            dilation=2,
            bias=True,
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        torch.manual_seed(dist.get_rank())
        x = torch.randn(2, in_ch, 15, 15, device=device)

        w = conv.weight.detach().clone().requires_grad_(True)
        b = conv.bias.detach().clone().requires_grad_(True)
        x_ref = x.clone().requires_grad_(True)
        y_ref = F.conv2d(x_ref, w, b, stride=2, padding=1, dilation=2)

        x_dist = x.clone().requires_grad_(True)
        y = conv(x_dist)
        assert torch.allclose(y, y_ref, atol=1e-5)

        y.backward(torch.ones_like(y))
        y_ref.backward(torch.ones_like(y_ref))
        assert torch.allclose(x_dist.grad, x_ref.grad, atol=1e-5)
        assert torch.allclose(conv.weight.grad, w.grad, atol=1e-5)
        assert torch.allclose(conv.bias.grad, b.grad, atol=1e-5)

    @pytest.mark.dist(4)
    def test_tiling_conv_matches_single_process(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        """Split spatial group, no channel split: the no-halo path still works."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
        )

    @pytest.mark.dist(4)
    def test_split_spatial_halo_no_channel_split(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        """No channel split, split spatial group, halo regime: routed through
        ``conv2d_apply`` / ``conv2d_halo_nway`` rather than the local fast path."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
        )


class TestConv2dGroups:
    """Grouped convolution (``groups > 1``) across spatial / channel group sizes.

    A grouped conv only mixes channels within a group, so once the
    channel-parallel group (``col``) owns whole conv-groups -- ``groups`` a
    multiple of ``col.size()`` -- every rank runs a plain grouped conv on its own
    channel shard with no channel-group communication. ``groups == 1`` keeps the
    historical full-channel-gather path; ``1 < groups`` not divisible by
    ``col.size()`` is rejected at construction.
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
        """2-way spatial x 2-way channel: groups 1 (gather) and 2, 4 (channel-local)."""
        _skip_if_unsupported(channel_comm)
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            groups=groups,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [2, 4])
    def test_2x2_halo(
        self, device: torch.device, process_mesh_dict_2x2: dict, groups: int
    ) -> None:
        """Channel-local grouped conv through the halo path (stride < kernel)."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
            groups=groups,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [1, 2, 3, 4, 6, 12])
    def test_pure_spatial_4x1(
        self, device: torch.device, process_mesh_dict_4x1: dict, groups: int
    ) -> None:
        """No channel split (``col.size() == 1``): any ``groups`` dividing channels."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=12,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            groups=groups,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [2, 4])
    def test_pure_spatial_4x1_halo(
        self, device: torch.device, process_mesh_dict_4x1: dict, groups: int
    ) -> None:
        """4-way spatial split, grouped conv, halo regime (real cross-rank exchange)."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=32,
            width=32,
            stride=2,
            padding=1,
            groups=groups,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [1, 4, 12])
    def test_pure_channel_1x4(
        self, device: torch.device, process_mesh_dict_1x4: dict, groups: int
    ) -> None:
        """4-way channel split: ``groups`` 1 (gather) or a multiple of 4 (local)."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_1x4,
            in_channels=12,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            groups=groups,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        ("kernel", "stride", "padding", "dilation"),
        [
            ((3, 5), (2, 3), (1, 1), 1),
            ((5, 3), (2, 2), (2, 1), 1),
            (3, 1, 2, (1, 2)),
        ],
        ids=["asym_stride_lt_k", "asym_stride", "dilated"],
    )
    def test_2x2_asymmetric_dilated(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        kernel: Pair,
        stride: Pair,
        padding: Pair,
        dilation: Pair,
    ) -> None:
        """Channel-local grouped conv with asymmetric / dilated hyperparams."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=kernel,
            height=24,
            width=24,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=2,
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
        """2-way spatial x 4-way channel, channel-local grouped conv."""
        _skip_if_unsupported(channel_comm)
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=16,
            kernel_size=4,
            height=16,
            width=16,
            groups=groups,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("groups", [4, 8])
    def test_2x4_halo(
        self, device: torch.device, process_mesh_dict_2x4_rowcol: dict, groups: int
    ) -> None:
        """2-way spatial x 4-way channel, channel-local grouped conv, halo regime."""
        _conv2d_reference_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=16,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
            groups=groups,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("groups", [3, 9])
    def test_incompatible_groups_rejected(
        self, device: torch.device, process_mesh_dict_2x2: dict, groups: int
    ) -> None:
        """``1 < groups`` not a multiple of ``col.size()`` fails fast at construction.

        ``col.size() == 2`` here, so an odd ``groups`` that still divides the
        channel counts (18 is 2 * 9, 9 = 3 * 3) can only mean several channel
        ranks share one conv-group -- unsupported.
        """
        with pytest.raises(NotImplementedError):
            convolution.DistributedConv2D(
                18,
                18,
                4,
                device=device,
                spatial_group=process_mesh_dict_2x2["row"],
                channel_group=process_mesh_dict_2x2["col"],
                stride=4,
                groups=groups,
            ).to(device)

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("optimizer", list(_OPTIMIZERS))
    def test_optimizer_parity_2x2(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        optimizer: str,
    ) -> None:
        """Several train steps of a channel-local grouped conv stay in lock-step
        with a single-process grouped ``conv2d`` (the spatial grad-sync hook still
        has to sum the per-spatial-rank partial weight gradients)."""
        _conv2d_optimizer_parity_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            groups=2,
            optimizer=optimizer,
        )


class TestConv2dOptimizerParity:
    """Multiple forward/backward/``optimizer.step`` iterations stay in lock-step
    with a single-process ``conv2d``.

    A single backward already checks the gradient (see ``TestConv2dParallel``);
    these run several *training steps* so a wrong
    ``register_post_accumulate_grad_hook`` -- one that fails to sum the
    per-spatial-rank partial gradients, sums them twice, or lets the optimizer
    step before the async all-reduce lands -- shows up as the weight shards
    drifting apart or away from the reference within a step or two (momentum
    makes any drift compound).
    """

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("optimizer", list(_OPTIMIZERS))
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_multi_iter_2x2(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        optimizer: str,
        channel_comm: str,
    ) -> None:
        """2-way spatial x 2-way channel, no-halo, every channel comm."""
        _skip_if_unsupported(channel_comm)
        _conv2d_optimizer_parity_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            optimizer=optimizer,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("optimizer", list(_OPTIMIZERS))
    def test_multi_iter_halo_2x2(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        optimizer: str,
    ) -> None:
        """Halo regime (stride < kernel): grads are summed over the spatial group
        from the halo-padded backward every step."""
        _conv2d_optimizer_parity_check(
            device,
            process_mesh_dict_2x2,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
            optimizer=optimizer,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("optimizer", list(_OPTIMIZERS))
    def test_multi_iter_pure_spatial_4x1(
        self,
        device: torch.device,
        process_mesh_dict_4x1: dict,
        optimizer: str,
    ) -> None:
        """4-way spatial, no channel split: the hook is the *only* thing summing
        weight/bias grads across the four spatial ranks, so a no-op hook drops
        the gathered gradient to ~1/4 of the reference on the first step."""
        _conv2d_optimizer_parity_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            optimizer=optimizer,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("optimizer", list(_OPTIMIZERS))
    def test_multi_iter_pure_spatial_halo_4x1(
        self,
        device: torch.device,
        process_mesh_dict_4x1: dict,
        optimizer: str,
    ) -> None:
        """4-way spatial, no channel split, halo regime (routed through
        ``conv2d_apply`` / ``conv2d_halo_nway``)."""
        _conv2d_optimizer_parity_check(
            device,
            process_mesh_dict_4x1,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=32,
            width=32,
            stride=2,
            padding=1,
            optimizer=optimizer,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_multi_iter_pure_channel_1x4(
        self,
        device: torch.device,
        process_mesh_dict_1x4: dict,
        channel_comm: str,
    ) -> None:
        """4-way channel, no spatial split: each rank owns a disjoint output-channel
        shard, so the hook must *not* touch (no double-reduce) the weight grad."""
        _skip_if_unsupported(channel_comm)
        _conv2d_optimizer_parity_check(
            device,
            process_mesh_dict_1x4,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            channel_comm=channel_comm,
        )

    @pytest.mark.dist(6)
    @pytest.mark.parametrize("optimizer", list(_OPTIMIZERS))
    def test_multi_iter_3x2(
        self,
        device: torch.device,
        process_mesh_dict_3x2: dict,
        optimizer: str,
    ) -> None:
        """3-way spatial x 2-way channel: spatial group larger than two."""
        _conv2d_optimizer_parity_check(
            device,
            process_mesh_dict_3x2,
            in_channels=12,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=24,
            optimizer=optimizer,
        )

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("channel_comm", CHANNEL_COMMS)
    def test_multi_iter_halo_2x4(
        self,
        device: torch.device,
        process_mesh_dict_2x4_rowcol: dict,
        channel_comm: str,
    ) -> None:
        """2-way spatial x 4-way channel, halo regime, every channel comm."""
        _skip_if_unsupported(channel_comm)
        _conv2d_optimizer_parity_check(
            device,
            process_mesh_dict_2x4_rowcol,
            in_channels=8,
            out_channels=12,
            kernel_size=4,
            height=16,
            width=16,
            stride=2,
            padding=1,
            channel_comm=channel_comm,
        )


class TestConv2dTranspose:
    # 2x2 mesh only -> needs exactly 4 ranks (see TestConv2d).
    pytestmark = pytest.mark.dist(4)

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
        conv2d = convolution.DistributedConv2DTranspose(
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
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        weights_gathered_as_list = gather_results_as_list(
            conv2d.weight, process_mesh_dict_2x2
        )

        for i in range(process_mesh_dict_2x2["model"].size() // row_comm_size):
            assert_close(
                f"weight replica {i} vs {i + col_comm_size}",
                weights_gathered_as_list[i],
                weights_gathered_as_list[i + col_comm_size],
                1e-5,
                1e-5,
                device,
                process_mesh_dict_2x2["model"],
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

        conv2d = convolution.DistributedConv2DTranspose(
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
            bayesian=False,
            dtype=torch.float32,
        ).to(device)

        weights_gathered_as_list = gather_results_as_list(
            conv2d.weight, process_mesh_dict_2x2
        )

        for i in range(process_mesh_dict_2x2["model"].size() // row_comm_size):
            assert_close(
                f"weight replica {i} vs {i + col_comm_size}",
                weights_gathered_as_list[i],
                weights_gathered_as_list[i + col_comm_size],
                1e-5,
                1e-5,
                device,
                process_mesh_dict_2x2["model"],
            )

        x = torch.randn(4, in_dim // 2, h, w // 2, device=device)
        y = conv2d(x)
        y.backward(torch.ones_like(y))
        assert isinstance(y, torch.Tensor)
