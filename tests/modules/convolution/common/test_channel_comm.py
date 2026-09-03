"""Unit tests for :mod:`jigsaw.modules.convolution.common.channel_comm`.

The pure helpers -- ``resolve_channel_comm`` (``"auto"`` resolution + name
validation) and ``_in_channel_block`` (the weight input-channel slicer) -- need
no distributed environment. ``_resolve_channel_hop`` and ``gather_full_channels``
take a process group: their group-size-one paths run against a singleton group
built on the fly, and the real multi-rank gather is checked under
``@pytest.mark.dist``.
"""

import pytest
import torch
import torch.distributed as dist

from jigsaw.comm import comm_1d_alltoall, comm_1d_p2p
from jigsaw.modules.convolution.common.channel_comm import (
    CHANNEL_COMMS,
    _in_channel_block,
    _resolve_channel_hop,
    gather_full_channels,
    resolve_channel_comm,
)


class TestResolveChannelComm:
    def test_auto_resolves_to_p2p_on_cpu(self) -> None:
        """``"auto"`` -> ``"p2p"`` on a non-CUDA device."""
        assert resolve_channel_comm("auto", torch.device("cpu")) == "p2p"

    def test_auto_resolves_to_all_to_all_on_cuda(self) -> None:
        """``"auto"`` -> ``"all_to_all"`` on CUDA; only the device *type* matters."""
        assert (
            resolve_channel_comm("auto", torch.device("cuda", 0)) == "all_to_all"
        )

    @pytest.mark.parametrize("name", list(CHANNEL_COMMS))
    def test_explicit_names_pass_through(self, name: str) -> None:
        """A concrete, valid strategy name is returned unchanged."""
        assert resolve_channel_comm(name, torch.device("cpu")) == name

    @pytest.mark.parametrize("bad", ["alltoall", "all-gather", "", "AUTO", "ring"])
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
        """The slice is materialised, so writing to it does not alias the weight."""
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
    def test_p2p_returns_the_p2p_transport(self) -> None:
        """``"p2p"`` maps to :func:`jigsaw.comm.comm_1d_p2p` on any backend."""
        assert _resolve_channel_hop("p2p", dist.group.WORLD) is comm_1d_p2p

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
        assert (
            _resolve_channel_hop("all_to_all", dist.group.WORLD) is comm_1d_alltoall
        )


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
        assert torch.equal(full, torch.cat(buf, dim=1))

        x_grad = x.clone().requires_grad_(True)
        gather_full_channels(x_grad, col, channel_dim=1).pow(2).sum().backward()
        assert x_grad.grad is not None
        assert torch.equal(x_grad.grad, 2 * x_grad.detach())
