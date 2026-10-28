"""Unit tests for :mod:`jigsaw.modules.convolution._common.channel_comm`.

The pure helpers, ``resolve_channel_comm`` (``"auto"`` resolution + name
validation) and ``_in_channel_block`` (the weight input-channel slicer), need
no distributed environment. ``_resolve_channel_hop`` and ``gather_full_channels``
take a process group: their group-size-one paths run against a singleton group
built on the fly, and the real multi-rank gather is checked under
``@pytest.mark.dist``.
"""

import pytest
import torch
import torch.distributed as dist

from jigsaw._core.comm import (
    CommOp,
    comm_1d_alltoall,
    comm_1d_batch_isend_irecv,
    comm_1d_isend,
)
from jigsaw.modules.convolution._common.channel_comm import (
    CHANNEL_COMMS,
    _in_channel_block,
    _resolve_channel_hop,
    gather_full_channels,
    resolve_channel_comm,
    resolve_conv_groups,
)
from jigsaw.modules.convolution._common.halo import exchange_along_axis
from jigsaw.modules.convolution._common.spatial_split import _reduce_scatter_channels
from tests.parity import assert_close


class TestResolveChannelComm:
    def test_auto_resolves_to_batch_isend_irecv_on_cpu(self) -> None:
        """``"auto"`` -> ``"batch_isend_irecv"`` on a non-CUDA device."""
        assert resolve_channel_comm("auto", torch.device("cpu")) == "batch_isend_irecv"

    def test_auto_resolves_to_all_to_all_on_cuda(self) -> None:
        """``"auto"`` -> ``"all_to_all"`` on CUDA; only the device *type* matters."""
        assert resolve_channel_comm("auto", torch.device("cuda", 0)) == "all_to_all"

    @pytest.mark.parametrize("name", list(CHANNEL_COMMS))
    def test_explicit_names_pass_through(self, name: str) -> None:
        """A concrete, valid strategy name is returned unchanged."""
        assert resolve_channel_comm(name, torch.device("cuda")) == name

    def test_comm_op_is_accepted(self) -> None:
        """A :class:`CommOp` member resolves like its value."""
        assert resolve_channel_comm(CommOp.ISEND, torch.device("cpu")) == "isend"

    def test_all_to_all_off_cuda_raises(self) -> None:
        """All-to-all needs NCCL, so a CPU module refuses it at construction."""
        with pytest.raises(ValueError, match="needs CUDA"):
            resolve_channel_comm("all_to_all", torch.device("cpu"))

    @pytest.mark.parametrize("bad", ["alltoall", "all-gather", "", "p2p", "ring"])
    def test_unknown_name_raises(self, bad: str) -> None:
        """Anything that is neither ``"auto"`` nor a known name fails fast."""
        with pytest.raises(ValueError, match="channel_comm must be"):
            resolve_channel_comm(bad, torch.device("cpu"))


class TestInChannelBlock:
    @pytest.mark.parametrize("n_spatial", [1, 2, 3])
    def test_block_matches_manual_narrow(self, n_spatial: int) -> None:
        """Block ``b`` of width ``w`` is ``w[:, b*w : (b+1)*w]`` for any spatial rank."""
        torch.manual_seed(0)
        out_c, in_c, width = 6, 12, 4  # 12 / 4 -> 3 blocks
        w = torch.randn(out_c, in_c, *([3] * n_spatial))
        for block in range(in_c // width):
            got = _in_channel_block(w, channel_dim=1, block=block, width=width)
            expected = w[:, block * width : (block + 1) * width]
            assert got.shape == expected.shape
            assert torch.equal(got, expected)
            assert got.is_contiguous()

    def test_block_is_a_contiguous_copy(self) -> None:
        """The slice is materialized, so writing to it does not alias the weight."""
        w = torch.randn(4, 8, 3)
        block = _in_channel_block(w, channel_dim=1, block=0, width=4)
        block.add_(1.0)
        assert not torch.equal(w[:, :4], block)


def _singleton_group() -> dist.ProcessGroup:
    """A process group containing only this rank (size 1), valid at any world size."""
    world = dist.get_world_size()
    groups = [dist.new_group([r]) for r in range(world)]
    return groups[dist.get_rank()]


class TestResolveChannelHop:
    @pytest.mark.parametrize(
        ("name", "fn"),
        [("isend", comm_1d_isend), ("batch_isend_irecv", comm_1d_batch_isend_irecv)],
    )
    def test_p2p_names_return_their_transport(self, name: str, fn) -> None:
        """The point-to-point strategies map to their exchange on any backend."""
        assert _resolve_channel_hop(name, dist.group.WORLD) is fn

    def test_all_to_all_on_gloo_raises(self) -> None:
        """gloo has no all-to-all, so the hop lookup refuses it (loudly)."""
        if dist.get_backend() != "gloo":
            pytest.skip("only meaningful on the gloo backend")
        with pytest.raises(RuntimeError, match="all-to-all"):
            _resolve_channel_hop("all_to_all", dist.group.WORLD)

    def test_all_to_all_on_nccl_returns_the_alltoall_transport(self) -> None:
        """NCCL has a real all-to-all, so the hop lookup returns it."""
        if dist.get_backend() != "nccl":
            pytest.skip("only meaningful on the nccl backend")
        assert _resolve_channel_hop("all_to_all", dist.group.WORLD) is comm_1d_alltoall


class TestGatherFullChannels:
    def test_group_size_one_is_the_identity(self) -> None:
        """A channel group of one returns the input tensor unchanged (same object)."""
        x = torch.randn(2, 4, 8)
        out = gather_full_channels(x, _singleton_group(), channel_dim=1)
        assert out is x

    @pytest.mark.dist(4)
    def test_gather_concatenates_every_rank_slice_in_order(
        self, device: torch.device, process_mesh_dict_1x4: dict
    ) -> None:
        """The result is every rank's slice concatenated along ``channel_dim`` in
        rank order, and it stays wired into autograd for the local slice."""
        col = process_mesh_dict_1x4["col"]
        g_size, g_rank = col.size(), col.rank()
        torch.manual_seed(g_rank)  # a distinct slice per rank
        x = torch.randn(2, 3, 5, device=device)

        full = gather_full_channels(x, col, channel_dim=1)
        assert full.shape[1] == 3 * g_size

        buf = [torch.empty_like(x) for _ in range(g_size)]
        dist.all_gather(buf, x.contiguous(), group=col)
        # Each rank holds a different slice, and another gather_full_channels
        # all-gather (plus its backward) follows, so a one-sided mismatch has to
        # fail every rank rather than strand them in that collective. Exact
        # comparison: atol/rtol of 0 keeps the original torch.equal semantics.
        assert_close(
            "gathered channels",
            full,
            torch.cat(buf, dim=1),
            0.0,
            0.0,
            device,
            process_mesh_dict_1x4["model"],
        )

        x_grad = x.clone().requires_grad_(True)
        gather_full_channels(x_grad, col, channel_dim=1).pow(2).sum().backward()
        assert x_grad.grad is not None
        assert torch.equal(x_grad.grad, 2 * x_grad.detach())


class TestResolveConvGroups:
    def test_dense_convolution_is_not_channel_local(self) -> None:
        assert resolve_conv_groups(1, 8, 8, 4) == (False, 1)

    def test_groups_that_are_a_multiple_of_the_channel_group_are_channel_local(
        self,
    ) -> None:
        assert resolve_conv_groups(8, 16, 16, 4) == (True, 2)

    def test_rejects_non_positive_groups(self) -> None:
        with pytest.raises(ValueError, match="groups must be >= 1"):
            resolve_conv_groups(0, 8, 8, 2)

    @pytest.mark.parametrize("in_ch, out_ch", [(6, 8), (8, 6)])
    def test_rejects_channels_not_divisible_by_groups(
        self, in_ch: int, out_ch: int
    ) -> None:
        with pytest.raises(ValueError, match="divisible by groups"):
            resolve_conv_groups(4, in_ch, out_ch, 2)

    def test_rejects_groups_that_do_not_fit_the_channel_group(self) -> None:
        with pytest.raises(NotImplementedError, match="multiple"):
            resolve_conv_groups(3, 6, 6, 2)


class TestSingletonGroupPrimitives:
    def test_reduce_scatter_of_a_single_rank_is_the_identity(self) -> None:
        g_x = torch.randn(2, 4, 5)
        out, handle = _reduce_scatter_channels(g_x, _singleton_group(), n=1)
        assert handle is None and torch.equal(out, g_x)

    def test_exchange_with_nothing_to_send_returns_empty_slices(
        self, device: torch.device
    ) -> None:
        template = torch.randn(2, 3, 4, device=device)
        part = template.narrow(-1, 0, 0)
        (recv,) = exchange_along_axis(
            _singleton_group(), template, [part], [0], [0], axis=-1
        )
        assert recv.shape == (2, 3, 0)
