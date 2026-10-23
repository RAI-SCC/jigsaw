"""Policy -> callable resolution in ``RingAttentionConfig`` (no communication issued).

The selectors branch on the device only, so the CUDA branches are reached by
passing a CUDA device; nothing is sent.
"""

import pytest
import torch
import torch.distributed as dist

import jigsaw._core.comm as comm
from jigsaw.modules.attention import _config as cfg
from jigsaw.modules.attention._config import (
    AttnBackend,
    Offload,
    RingAttentionConfig,
    SaveBackward,
    SaveScope,
    SendBackward,
    SendForward,
    SendOp,
)
from jigsaw.modules.attention._functional.cores import math as math_core
from jigsaw.modules.attention._functional.cores import sdpa as sdpa_core

CPU, CUDA = torch.device("cpu"), torch.device("cuda")


class TestCommSelection:
    @pytest.mark.parametrize(
        "dev, expected",
        [(CPU, comm.comm_1d_batch_isend_irecv), (CUDA, comm.comm_1d_alltoall)],
        ids=["cpu", "cuda"],
    )
    def test_auto_picks_by_device(self, dev, expected):
        config = RingAttentionConfig(device=dev)
        assert config.send_op_fn is expected
        assert config.p2p_fn is expected

    @pytest.mark.parametrize(
        "op, expected",
        [
            ("isend", comm.comm_1d_isend),
            ("batch_isend_irecv", comm.comm_1d_batch_isend_irecv),
            ("all_to_all", comm.comm_1d_alltoall),
        ],
    )
    def test_comm_sets_both_hops(self, op, expected):
        config = RingAttentionConfig(device=CUDA, comm=op)
        assert config.send_op_fn is expected
        assert config.p2p_fn is expected

    def test_outer_comm_overrides_the_outer_hop(self):
        config = RingAttentionConfig(
            device=CUDA, send_op="hybrid", comm="all_to_all", outer_comm="isend"
        )
        assert config.send_op_fn is comm.comm_1d_alltoall
        assert config.p2p_fn is comm.comm_1d_isend

    def test_broadcast_broadcasts_each_step(self):
        config = RingAttentionConfig(device=CPU, send_op="broadcast")
        assert config.send_op_fn is comm.comm_broadcast
        assert config.p2p_fn is comm.comm_1d_batch_isend_irecv

    @pytest.mark.parametrize("arg", ["comm", "outer_comm"])
    def test_alltoall_off_cuda_is_rejected(self, arg):
        with pytest.raises(ValueError, match="needs CUDA"):
            RingAttentionConfig(device=CPU, **{arg: "all_to_all"})


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
        from jigsaw.modules.attention._functional.cores import te as te_core
        from jigsaw.modules.attention._functional.cores.te import cudnn

        monkeypatch.setattr(te_core, "TE_CORE_AVAILABLE", True)
        # ``select_core`` resolves non-CUDA devices to ``cudnn`` (see its docstring).
        assert cfg._select_attn_core_forward_fn(CPU, AttnBackend.TE) is cudnn.forward
        assert cfg._select_attn_core_backward_fn(CPU, AttnBackend.TE) is cudnn.backward

    def test_te_backend_raises_without_transformer_engine(self, monkeypatch):
        from jigsaw.modules.attention._functional.cores import te as te_core

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
            send = cfg._select_forward_send_fn(SendOp.RING, send_in_forward, fake)
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
            lambda t, group_src, group, async_op: seen.append((t, group_src)) or "work",
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
    def test_rejects_non_positive_intra_size(self):
        with pytest.raises(ValueError, match="intra_size"):
            RingAttentionConfig(device=CPU, intra_size=0)

    def test_wait_offloads_delegates_only_with_an_offloader(self):
        config = RingAttentionConfig(device=CPU)
        config.wait_offloads()  # no offloader: no-op

        waited = []
        config.offloader = type(
            "O", (), {"wait_offloads": lambda self: waited.append(1)}
        )()
        config.wait_offloads()
        assert waited == [1]

    @pytest.mark.parametrize(
        "arg, value, expected",
        [
            ("send_in_forward", "kv", SendForward.KV),
            ("send_op", "hybrid", SendOp.HYBRID),
            ("send_op", "Hybrid_Broadcast", SendOp.HYBRID_BROADCAST),
            ("send_op", "RING", SendOp.RING),
            ("save_for_backward", "x", SaveBackward.X),
            ("save_scope", "local", SaveScope.LOCAL),
            ("send_backward", "gather_dkv", SendBackward.GATHER_DKV),
            ("offload", "disabled", Offload.DISABLED),
            ("attn_backend", "MATH", AttnBackend.MATH),
            ("attn_backend", "math", AttnBackend.MATH),
        ],
    )
    def test_enum_names_are_accepted_in_any_case(self, arg, value, expected):
        config = RingAttentionConfig(device=CPU, **{arg: value})
        assert getattr(config, arg) is expected

    @pytest.mark.parametrize("member", [SendOp.RING, SendOp.BROADCAST])
    def test_enum_members_pass_through(self, member):
        assert RingAttentionConfig(device=CPU, send_op=member).send_op is member

    @pytest.mark.parametrize(
        "arg, value",
        [
            ("send_op", "p2p"),
            ("send_op", 3),
            ("save_scope", "everywhere"),
            ("attn_backend", "flash"),
            ("send_in_forward", None),
        ],
    )
    def test_unknown_values_are_rejected(self, arg, value):
        with pytest.raises(ValueError, match=rf"{arg}={value!r} is not a"):
            RingAttentionConfig(device=CPU, **{arg: value})

    def test_error_lists_the_valid_names(self):
        with pytest.raises(ValueError, match="'hybrid_broadcast'"):
            RingAttentionConfig(device=CPU, send_op="hybrid_p2p")

    def test_repr_lists_the_policies(self):
        text = repr(RingAttentionConfig(device=CPU, attn_backend="math"))
        for token in (
            "send_op=RING",
            "comm=batch_isend_irecv",
            "save_scope=GLOBAL",
            "attn_backend=MATH",
        ):
            assert token in text
