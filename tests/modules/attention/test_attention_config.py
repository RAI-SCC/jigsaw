"""Policy -> callable resolution in ``RingAttentionConfig`` (no communication issued).

The selectors branch on ``dist.get_backend()`` and the device, so the NCCL/CUDA
branches are reached by monkeypatching the backend name; nothing is sent.
"""

import pytest
import torch
import torch.distributed as dist

import jigsaw.comm as comm
from jigsaw.modules.attention import config as cfg
from jigsaw.modules.attention.config import (
    AttnBackend,
    Offload,
    RingAttentionConfig,
    SaveBackward,
    SaveScope,
    SendForward,
    SendOp,
)
from jigsaw.modules.attention.functional.cores import math as math_core
from jigsaw.modules.attention.functional.cores import sdpa as sdpa_core

CPU = torch.device("cpu")


@pytest.fixture
def nccl(monkeypatch):
    monkeypatch.setattr(dist, "get_backend", lambda *a, **k: "nccl")


@pytest.fixture
def gloo(monkeypatch):
    monkeypatch.setattr(dist, "get_backend", lambda *a, **k: "gloo")


class TestCommSelection:
    @pytest.mark.parametrize("send_op", list(SendOp))
    def test_gloo_always_uses_p2p(self, gloo, send_op):
        assert cfg._select_p2p_fn(send_op) is comm.comm_1d_p2p
        assert cfg._select_send_op_fn(send_op) is comm.comm_1d_p2p

    @pytest.mark.parametrize(
        "send_op, expected",
        [
            (SendOp.ALLTOALL, comm.comm_1d_alltoall),
            (SendOp.BROADCAST, comm.comm_1d_alltoall),
            (SendOp.HYBRID_ALLTOALL, comm.comm_1d_alltoall),
            (SendOp.P2P, comm.comm_1d_p2p),
            (SendOp.HYBRID_P2P, comm.comm_1d_p2p),
            (SendOp.HYBRID_BROADCAST, comm.comm_1d_p2p),
        ],
    )
    def test_nccl_inter_node_hop(self, nccl, send_op, expected):
        assert cfg._select_p2p_fn(send_op) is expected

    @pytest.mark.parametrize(
        "send_op, expected",
        [
            (SendOp.BROADCAST, comm.comm_broadcast),
            (SendOp.ALLTOALL, comm.comm_1d_alltoall),
            (SendOp.HYBRID_P2P, comm.comm_1d_alltoall),
            (SendOp.HYBRID_ALLTOALL, comm.comm_1d_alltoall),
            (SendOp.P2P, comm.comm_1d_p2p),
            (SendOp.HYBRID_BROADCAST, comm.comm_1d_p2p),
        ],
    )
    def test_nccl_per_step_send(self, nccl, send_op, expected):
        assert cfg._select_send_op_fn(send_op) is expected


class TestCoreSelection:
    def test_math_backend_on_any_device(self):
        assert (
            cfg._select_attn_core_forward_fn(CPU, AttnBackend.MATH) is math_core.forward
        )
        assert (
            cfg._select_attn_core_backward_fn(CPU, AttnBackend.MATH)
            is math_core.backward
        )

    def test_sdpa_backend_uses_sdpa_on_cuda_and_math_on_cpu(self):
        cuda = torch.device("cuda")
        assert (
            cfg._select_attn_core_forward_fn(cuda, AttnBackend.SDPA)
            is sdpa_core.forward
        )
        assert (
            cfg._select_attn_core_backward_fn(cuda, AttnBackend.SDPA)
            is sdpa_core.backward
        )
        assert (
            cfg._select_attn_core_forward_fn(CPU, AttnBackend.SDPA) is math_core.forward
        )
        assert (
            cfg._select_attn_core_backward_fn(CPU, AttnBackend.SDPA)
            is math_core.backward
        )

    def test_string_backend_names_are_accepted(self):
        assert cfg._select_attn_core_forward_fn(CPU, "math") is math_core.forward

    def test_sdpa_on_unknown_device_is_rejected(self):
        meta = torch.device("meta")
        with pytest.raises(ValueError, match="no attention forward core"):
            cfg._select_attn_core_forward_fn(meta, AttnBackend.SDPA)
        with pytest.raises(ValueError, match="no attention backward core"):
            cfg._select_attn_core_backward_fn(meta, AttnBackend.SDPA)

    def test_te_backend_dispatches_to_te_core_when_available(self, monkeypatch):
        from jigsaw.modules.attention.functional.cores import te as te_core

        monkeypatch.setattr(te_core, "TE_CORE_AVAILABLE", True)
        assert cfg._select_attn_core_forward_fn(CPU, AttnBackend.TE) is te_core.forward
        assert (
            cfg._select_attn_core_backward_fn(CPU, AttnBackend.TE) is te_core.backward
        )

    def test_te_backend_raises_without_transformer_engine(self, monkeypatch):
        from jigsaw.modules.attention.functional.cores import te as te_core

        monkeypatch.setattr(te_core, "TE_CORE_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="transformer_engine"):
            cfg._select_attn_core_forward_fn(CPU, AttnBackend.TE)
        with pytest.raises(RuntimeError, match="transformer_engine"):
            cfg._select_attn_core_backward_fn(CPU, AttnBackend.TE)


class TestOffloaderSelection:
    def test_disabled_has_no_offloader(self):
        assert cfg._select_offloader(Offload.DISABLED, CPU) is None

    def test_cpu_offload_builds_an_offloader(self, monkeypatch):
        built = []
        monkeypatch.setattr(
            cfg, "NVLinkActivationOffloader", lambda dev: built.append(dev) or "off"
        )
        assert cfg._select_offloader(Offload.CPU, CPU) == "off" and built == [CPU]

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(ValueError, match="unhandled offload option"):
            cfg._select_offloader("bogus", CPU)  # type: ignore[arg-type]


class TestForwardSendRecv:
    def test_ring_send_forwards_x_or_kv_to_the_comm_fn(self):
        calls = []

        def fake(buf, recv, direction, **kw):
            calls.append((buf, direction, kw))
            return recv, "handle"

        x, kv = torch.ones(2), torch.zeros(3)
        for send_in_forward, expected in ((SendForward.X, x), (SendForward.KV, kv)):
            send = cfg._select_forward_send_fn(SendOp.P2P, send_in_forward, fake)
            recv, handle = send(x, kv, 1, "group", 7, group_size=4, extra="ignored")
            assert handle == "handle" and recv.shape == expected.shape
            buf, direction, kw = calls[-1]
            assert buf is expected and direction == 1
            assert kw == dict(group="group", async_op=True, tag=7, group_size=4)

    @pytest.mark.parametrize("send_kv", [False, True])
    def test_broadcast_send_reuses_the_buffer_on_the_source_rank(
        self, monkeypatch, send_kv
    ):
        seen = []
        monkeypatch.setattr(
            dist,
            "broadcast",
            lambda t, src, group, async_op: seen.append((t, src)) or "work",
        )
        x, kv = torch.ones(2), torch.zeros(3)
        send = cfg._select_forward_send_fn(
            SendOp.BROADCAST, SendForward.KV if send_kv else SendForward.X, None
        )
        buf = kv if send_kv else x

        recv, handle = send(x, kv, local_rank=2, group="g", src_rank=2)
        assert recv is buf and handle == "work"

        recv, _ = send(x, kv, local_rank=1, group="g", src_rank=2)
        assert recv is not buf and recv.shape == buf.shape
        assert seen[-1][1] == 2

    def test_recv_replaces_x_or_unpacks_kv(self):
        xj, kv_j, kj, vj = (torch.full((1,), float(i)) for i in range(4))
        buf = torch.stack([torch.full((1,), 8.0), torch.full((1,), 9.0)])

        assert cfg._select_forward_recv_fn(SendForward.X)(buf, xj, kv_j, kj, vj) == (
            buf,
            kv_j,
            kj,
            vj,
        )
        out = cfg._select_forward_recv_fn(SendForward.KV)(buf, xj, kv_j, kj, vj)
        assert out[0] is xj and out[1] is buf
        assert out[2].item() == 8.0 and out[3].item() == 9.0


class TestSaveForBackward:
    x, kv = torch.ones(2), torch.zeros(2)

    def test_none_saves_nothing(self):
        fn = cfg._select_save_for_backward_fn(None, SaveScope.LOCAL, None)  # type: ignore[arg-type]
        assert fn(self.x, self.kv) is None

    @pytest.mark.parametrize(
        "what, tensor", [(SaveBackward.X, x), (SaveBackward.KV, kv)]
    )
    def test_local_scope_aliases_and_global_scope_clones(self, what, tensor):
        local = cfg._select_save_for_backward_fn(what, SaveScope.LOCAL, None)(
            self.x, self.kv
        )
        glob = cfg._select_save_for_backward_fn(what, SaveScope.GLOBAL, None)(
            self.x, self.kv
        )
        assert local is tensor
        assert glob is not tensor and torch.equal(glob, tensor)

    def test_offloader_takes_precedence(self, monkeypatch):
        class FakeOffloader:
            def offload(self, chunk):
                return ("offloaded", chunk)

        monkeypatch.setattr(cfg, "NVLinkActivationOffloader", FakeOffloader)
        fn = cfg._select_save_for_backward_fn(
            SaveBackward.X, SaveScope.GLOBAL, FakeOffloader()
        )
        assert fn(self.x, self.kv) == ("offloaded", self.x)


class TestRingAttentionConfig:
    def test_rejects_non_positive_gpus_per_node(self):
        with pytest.raises(ValueError, match="gpus_per_node"):
            RingAttentionConfig(device=CPU, gpus_per_node=0)

    def test_wait_offloads_delegates_only_with_an_offloader(self):
        config = RingAttentionConfig(device=CPU)
        config.wait_offloads()  # no offloader: no-op

        waited = []
        config.offloader = type(
            "O", (), {"wait_offloads": lambda self: waited.append(1)}
        )()
        config.wait_offloads()
        assert waited == [1]

    def test_repr_lists_the_policies(self):
        text = repr(RingAttentionConfig(device=CPU, attn_backend="math"))
        for token in ("send_op=HYBRID_P2P", "save_scope=GLOBAL", "attn_backend=MATH"):
            assert token in text
