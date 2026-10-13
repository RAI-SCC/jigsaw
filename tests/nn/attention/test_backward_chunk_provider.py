"""Transfer schedule of the hybrid-P2P backward chunk provider (no communication issued).

``_LocalChunkProviderHybridP2P.maybe_forward`` launches the transfers that deliver
the *next* step's chunk. Every launched transfer must be consumed by the following
``get``; one that is not gets its handle and receive buffer overwritten while still
in flight, which hangs Gloo (``test_ring_hybrid*[...GATHER_DKV...HYBRID_P2P...]``
deadlocked intermittently on CPU). The launches are stubbed out here, so the schedule
is checked without any ranks.
"""

from types import SimpleNamespace

import pytest
import torch

from jigsaw.nn.attention.config import SaveScope
from jigsaw.nn.attention.functional import forward as ring_forward
from jigsaw.nn.attention.functional.backward_chunk_provider import (
    _build_backward_chunk_provider,
    _GlobalChunkProvider,
    _LocalChunkProviderHybridP2P,
)
from jigsaw.nn.attention.functional.forward_chunk_provider import (
    _ForwardChunkProvider,
    _HybridP2PForwardChunkProvider,
    build_forward_chunk_provider,
)

# (n_nodes, gpus_per_node); the ring has n_nodes * gpus_per_node ranks.
TOPOLOGIES = [(2, 2), (2, 4), (3, 2), (4, 2), (2, 3), (2, 1), (3, 1)]


def _provider(n_nodes: int, gpn: int):
    """Build a provider that records its launches instead of communicating."""
    provider = object.__new__(_LocalChunkProviderHybridP2P)
    provider._gpn = gpn
    provider.topo = SimpleNamespace(n_nodes=n_nodes)
    provider._internode_recv_buf = torch.zeros(1)
    provider._recv_buf = torch.zeros(1)
    provider.intra, provider.inter = [], []
    provider._launch_intra_backward = lambda src, step: provider.intra.append(step)

    def launch_inter(data: torch.Tensor, step: int) -> None:
        # The real launch replaces the receive buffer with the next hop's empty one.
        provider.inter.append(step)
        provider._internode_recv_buf = torch.empty_like(data)

    provider._launch_inter_backward = launch_inter
    return provider


@pytest.mark.parametrize("n_nodes, gpn", TOPOLOGIES)
class TestMaybeForwardSchedule:
    def test_intra_node_sends_are_only_those_the_next_step_consumes(self, n_nodes, gpn):
        ring = n_nodes * gpn
        provider = _provider(n_nodes, gpn)
        for step in range(ring):
            provider.maybe_forward(step, ring)
        # Step ``s + 1`` reads the intra-node ring unless it starts a node round, in
        # which case the chunk comes from the inter-node link instead.
        expected = [s for s in range(1, ring - 1) if (s + 1) % gpn != 0]
        assert provider.intra == expected

    def test_inter_node_relay_stops_before_the_last_node_round(self, n_nodes, gpn):
        ring = n_nodes * gpn
        provider = _provider(n_nodes, gpn)
        for step in range(ring):
            provider.maybe_forward(step, ring)
        expected = [
            s for s in range(1, ring - 1) if s % gpn == 0 and s // gpn < n_nodes - 1
        ]
        assert provider.inter == expected

    def test_node_boundary_restarts_the_intra_ring_from_the_inter_node_chunk(
        self, n_nodes, gpn
    ):
        ring = n_nodes * gpn
        provider = _provider(n_nodes, gpn)
        for step in range(ring):
            arrived = torch.full((1,), float(step))
            provider._internode_recv_buf = arrived
            provider._recv_buf = torch.zeros(1)
            sent = []
            provider._launch_intra_backward = lambda src, step: sent.append(src)
            provider.maybe_forward(step, ring)
            if 0 < step < ring - 1 and step % gpn == 0 and gpn > 1:
                # Whatever the relay did to ``_internode_recv_buf``, the intra-node
                # ring must restart from the chunk that arrived, not an empty buffer.
                assert sent == [arrived]


@pytest.mark.parametrize("step", [0, 3])
def test_first_and_last_step_launch_nothing(step):
    provider = _provider(n_nodes=2, gpn=2)
    provider.maybe_forward(step, ring_group_size=4)
    assert provider.intra == [] and provider.inter == []


class TestBuilderErrors:
    """Configurations the builders do not know raise instead of picking a provider."""

    def _build_backward(self, **config):
        return _build_backward_chunk_provider(
            SimpleNamespace(**config), (), None, None, 1, 1, 1, 1, None, 0
        )

    def test_unknown_save_scope(self):
        with pytest.raises(NotImplementedError, match="Unknown save_scope"):
            self._build_backward(save_scope="bogus")

    def test_unknown_send_op_in_the_backward_ring(self):
        with pytest.raises(NotImplementedError, match="Unknown send_op"):
            self._build_backward(save_scope=SaveScope.LOCAL, send_op="bogus")

    def test_unknown_send_op_in_the_forward_ring(self):
        q = torch.zeros(1, 1, 2, 2)
        with pytest.raises(NotImplementedError, match="Unknown send_op"):
            build_forward_chunk_provider(
                SimpleNamespace(send_op="bogus"), q, q, q, q, None, None, 0, 1
            )

    def test_unknown_save_for_backward_in_the_global_provider(self):
        config = SimpleNamespace(offloader=None, save_for_backward="bogus")
        provider = _GlobalChunkProvider(
            config, (torch.zeros(2, 1, 1, 2, 2),), None, 1, 2, 1, 2
        )
        with pytest.raises(NotImplementedError):
            provider.get(0)


class TestForwardBase:
    def test_global_chunk_idx_must_be_overridden(self):
        stubs = {
            name: (lambda self, *a, **k: None)
            for name in (_ForwardChunkProvider.__abstractmethods__)
        }
        provider = object.__new__(type("Stub", (_ForwardChunkProvider,), stubs))
        with pytest.raises(NotImplementedError):
            provider.global_chunk_idx(0)


def test_dropout_base_seed_is_a_python_int_below_2_31():
    seed = ring_forward._draw_base_seed(torch.device("cpu"))
    assert isinstance(seed, int) and 0 <= seed < 2**31 - 1


class TestForwardInterNodeRelay:
    """With more than two nodes, later rounds relay the chunk that just arrived."""

    def _provider(self, send_x: bool):
        provider = object.__new__(_HybridP2PForwardChunkProvider)
        provider.send_x = send_x
        provider.x, provider.x_j = torch.zeros(2), torch.zeros(2)
        provider.kv, provider.kv_j = torch.ones(2), torch.ones(2)
        provider._inter_ranks = [1, 2]
        provider.group = None
        provider._inter_group = None
        provider._inter_recv_buf = None
        provider.sent = []

        def p2p(data, buffer, **kwargs):
            provider.sent.append((data, kwargs["tag"]))
            return buffer, "handle"

        provider.config = SimpleNamespace(p2p_fn=p2p)
        return provider

    @pytest.mark.parametrize("send_x", [True, False])
    def test_first_round_sends_the_local_chunk(self, send_x):
        provider = self._provider(send_x)
        provider._launch_inter_forward(0)
        ((data, tag),) = provider.sent
        assert tag == 0
        assert torch.equal(data, provider.x if send_x else provider.kv)
        assert data is not (provider.x if send_x else provider.kv)

    @pytest.mark.parametrize("send_x", [True, False])
    def test_later_rounds_send_the_chunk_that_arrived(self, send_x):
        """``_recv`` rebinds ``x_j``/``kv_j`` from the inter-node receive before
        the next ``_launch_inter_forward`` call, so simulate that directly."""
        provider = self._provider(send_x)
        arrived = torch.full((2,), 7.0)
        if send_x:
            provider.x_j = arrived
        else:
            provider.kv_j = arrived
        provider._launch_inter_forward(1)
        ((data, tag),) = provider.sent
        assert tag == 1
        assert data is arrived  # sent in place, no clone (see the method's docstring)
