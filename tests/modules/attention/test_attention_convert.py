"""``Attention.convert_to_jigsaw`` on meshes that split the sequence, ring on and off.

The rows of the mesh split the sequence and the columns split the heads; the weights are
replicated over the rows and sharded over the columns. Every test converts a serial
``Attention`` and compares output, input gradient, parameter values and parameter
gradients with the serial module through ``check_parity``.

* **Ring** (``ring=True``) needs an unsharded channel axis by default, so it runs on a
  ``4x1`` mesh, and on ``2x2`` with a ``config`` that sends and saves the keys and values.
  The reference is the plain serial module on the whole sequence.
* **No ring** makes every row attend within its own chunk, so the reference attends over
  each row's window separately. That runs on ``4x1`` (spatial only) and on ``2x2``
  (spatial and channel).
* The converted weights must agree over the rows and their gradients must be summed over
  them, including the output bias, which ``DistributedXWT`` already syncs on its own.
"""

import pytest
import torch
import torch.distributed as dist

import jigsaw.modules.attention as attention
from jigsaw.modules.attention.config import (
    RingAttentionConfig,
    SaveBackward,
    SaveScope,
    SendForward,
    SendOp,
)
from tests.parity import REPLICATED, ShardSpec, assert_close, check_parity

IN_DIM, EMBED_DIM, OUT_DIM, HEADS = 16, 16, 16, 4
BATCH, SEQ = 2, 16


def _tols(device: torch.device) -> dict[str, float]:
    """Tolerances of a ``check_parity`` against the serial module.

    On CUDA the attention core runs in bf16 (``attn_core_dtype``), and the serial module
    uses the flash kernel where ``attn_backend="math"`` does not: the two differ by bf16
    rounding. That is ~1e-3 on the output and, summed over the tokens, ~1e-2 on the
    weight gradients (which are O(1) here); a wrong shard or weight layout is off by
    the size of the entries themselves.
    """
    if device.type == "cuda":
        return dict(atol=5e-3, rtol=5e-3, grad_atol=5e-2, grad_rtol=5e-3)
    return dict(atol=1e-4, rtol=1e-4)


def _serial(device: torch.device, out_bias: bool) -> attention.Attention:
    torch.manual_seed(0)  # identical reference on every rank
    serial = attention.Attention(IN_DIM, EMBED_DIM, OUT_DIM, HEADS, device)
    if out_bias:
        serial.head_concat = torch.nn.Linear(
            EMBED_DIM, OUT_DIM, bias=True, device=device
        )
    return serial


def _windowed(n_windows: int):
    """Reference forward where every window of the sequence attends on its own."""

    def forward(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([module(w) for w in x.chunk(n_windows, dim=1)], dim=1)

    return forward


def _unpermute_qkv(mesh: dict, device: torch.device):
    """Rebuild the serial ``qkv`` weight from a column-sharded, head-permuted one.

    Block ``i`` of the local weight holds the ``[3, heads // C, head_dim]`` rows of head
    shard ``i`` and the input columns of this channel rank. The weight must also agree
    over the rows.
    """
    col, row = mesh["col"], mesh["row"]

    def gather(local: torch.Tensor) -> torch.Tensor:
        replica = local.clone()
        dist.broadcast(replica, group_src=0, group=row)
        assert_close(
            "qkv row replicas", local, replica, 1e-6, 1e-6, device, mesh["model"]
        )

        shards = [torch.empty_like(local) for _ in range(col.size())]
        dist.all_gather(shards, local.contiguous(), group=col)
        head_dim = EMBED_DIM // HEADS
        blocks = [
            torch.cat([s.chunk(col.size(), dim=0)[i] for s in shards], dim=1).reshape(
                3, HEADS // col.size(), head_dim, IN_DIM
            )
            for i in range(col.size())
        ]
        return torch.cat(blocks, dim=1).reshape(3 * EMBED_DIM, IN_DIM)

    return gather


@pytest.mark.dist_required
class TestConvertWithRing:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("out_bias", [False, True])
    def test_matches_serial_attention(
        self, device: torch.device, process_mesh_dict_4x1: dict, out_bias: bool
    ) -> None:
        mesh = process_mesh_dict_4x1
        seen: list[attention.DistributedAttention] = []

        def build(module: torch.nn.Module) -> torch.nn.Module:
            model = attention.Attention.convert_to_jigsaw(
                module, mesh, device, ring=True, attn_backend="math"
            )
            seen.append(model)
            return model

        rules: dict = {
            "qkv.weight": ShardSpec(),
            "head_concat.weight": ShardSpec(),
        }
        if out_bias:
            rules["head_concat.bias"] = ShardSpec()
        check_parity(
            serial_module=_serial(device, out_bias),
            build_distributed=build,
            input_shape=(BATCH, SEQ, IN_DIM),
            input_spec=ShardSpec(row=1),
            output_spec=ShardSpec(row=1),
            param_rules=rules,
            process_mesh_dict=mesh,
            device=device,
            **_tols(device),
        )
        model = seen[0]
        assert model.ring
        assert model.attn_core == model._ring_attn_core

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("out_bias", [False, True])
    def test_matches_serial_attention_with_a_channel_split(
        self, device: torch.device, process_mesh_dict_2x2: dict, out_bias: bool
    ) -> None:
        """``config`` reaches the ring: sending and saving K/V allows a channel split."""
        mesh = process_mesh_dict_2x2
        config = RingAttentionConfig(
            device=device,
            send_in_forward=SendForward.KV,
            save_for_backward=SaveBackward.KV,
            send_op=SendOp.P2P,
            save_scope=SaveScope.LOCAL,
            attn_backend="math",
        )
        seen: list[attention.DistributedAttention] = []

        def build(module: torch.nn.Module) -> torch.nn.Module:
            model = attention.Attention.convert_to_jigsaw(
                module, mesh, device, ring=True, attn_backend="math", config=config
            )
            seen.append(model)
            return model

        rules: dict = {
            "qkv.weight": _unpermute_qkv(mesh, device),
            "head_concat.weight": ShardSpec(row=REPLICATED, col=1),
        }
        if out_bias:
            rules["head_concat.bias"] = ShardSpec(row=REPLICATED, col=0)
        check_parity(
            serial_module=_serial(device, out_bias),
            build_distributed=build,
            input_shape=(BATCH, SEQ, IN_DIM),
            input_spec=ShardSpec(row=1, col=2),
            output_spec=ShardSpec(row=1, col=2),
            param_rules=rules,
            process_mesh_dict=mesh,
            device=device,
            **_tols(device),
        )
        assert seen[0].ring
        assert seen[0].config is config

    @pytest.mark.dist(4)
    def test_auto_builds_a_kv_config_for_a_channel_split(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """No explicit config needed: a channel split auto-selects the KV policy."""
        serial = _serial(device, out_bias=False)
        model = attention.Attention.convert_to_jigsaw(
            serial, process_mesh_dict_2x2, device, ring=True
        )
        assert model.config.send_in_forward is SendForward.KV
        assert model.config.save_for_backward is SaveBackward.KV

    @pytest.mark.dist(4)
    def test_constructor_auto_builds_a_kv_config_for_a_channel_split(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        mesh = process_mesh_dict_2x2
        model = attention.DistributedAttention(
            in_dim=IN_DIM,
            embed_dim=EMBED_DIM,
            out_dim=OUT_DIM,
            heads=HEADS,
            dropout_p=0.0,
            local_rank=mesh["col"].rank(),
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            ring=True,
        )
        assert model.config.send_in_forward is SendForward.KV
        assert model.config.save_for_backward is SaveBackward.KV

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        "send, save, accepted",
        [
            (SendForward.KV, SaveBackward.KV, True),
            (SendForward.X, SaveBackward.KV, False),
            (SendForward.KV, SaveBackward.X, False),
        ],
        ids=["send_kv_save_kv", "send_x", "save_x"],
    )
    def test_channel_split_needs_a_ring_that_sends_and_saves_kv(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        send: SendForward,
        save: SaveBackward,
        accepted: bool,
    ) -> None:
        """A channel split is fine as long as no chunk's K/V is rebuilt from ``x``."""
        mesh = process_mesh_dict_2x2

        def build() -> attention.DistributedAttention:
            return attention.DistributedAttention(
                in_dim=IN_DIM,
                embed_dim=EMBED_DIM,
                out_dim=OUT_DIM,
                heads=HEADS,
                dropout_p=0.0,
                local_rank=mesh["col"].rank(),
                device=device,
                spatial_group=mesh["row"],
                channel_group=mesh["col"],
                ring=True,
                config=RingAttentionConfig(
                    device=device, send_in_forward=send, save_for_backward=save
                ),
            )

        if accepted:
            assert build().ring
        else:
            with pytest.raises(ValueError, match="channel/tensor parallelism"):
                build()


@pytest.mark.dist_required
class TestConvertWithoutRing:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("out_bias", [False, True])
    def test_spatial_only_mesh(
        self, device: torch.device, process_mesh_dict_4x1: dict, out_bias: bool
    ) -> None:
        mesh = process_mesh_dict_4x1
        rules: dict = {
            "qkv.weight": ShardSpec(),
            "head_concat.weight": ShardSpec(),
        }
        if out_bias:
            rules["head_concat.bias"] = ShardSpec()
        check_parity(
            serial_module=_serial(device, out_bias),
            build_distributed=lambda m: attention.Attention.convert_to_jigsaw(
                m, mesh, device, ring=False, attn_backend="math"
            ),
            input_shape=(BATCH, SEQ, IN_DIM),
            input_spec=ShardSpec(row=1),
            output_spec=ShardSpec(row=1),
            param_rules=rules,
            process_mesh_dict=mesh,
            device=device,
            **_tols(device),
            serial_forward=_windowed(mesh["row"].size()),
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("out_bias", [False, True])
    def test_spatial_and_channel_mesh(
        self, device: torch.device, process_mesh_dict_2x2: dict, out_bias: bool
    ) -> None:
        mesh = process_mesh_dict_2x2
        rules: dict = {
            "qkv.weight": _unpermute_qkv(mesh, device),
            "head_concat.weight": ShardSpec(row=REPLICATED, col=1),
        }
        if out_bias:
            rules["head_concat.bias"] = ShardSpec(row=REPLICATED, col=0)
        check_parity(
            serial_module=_serial(device, out_bias),
            build_distributed=lambda m: attention.Attention.convert_to_jigsaw(
                m, mesh, device, ring=False, attn_backend="math"
            ),
            input_shape=(BATCH, SEQ, IN_DIM),
            input_spec=ShardSpec(row=1, col=2),
            output_spec=ShardSpec(row=1, col=2),
            param_rules=rules,
            process_mesh_dict=mesh,
            device=device,
            **_tols(device),
            serial_forward=_windowed(mesh["row"].size()),
        )


@pytest.mark.dist_required
class TestConvertedWeightsAreReplicated:
    @pytest.mark.dist(4)
    def test_rows_take_the_first_rows_weights(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        """Serial weights that differ between the rows are overwritten by row 0's."""
        mesh = process_mesh_dict_4x1
        row = mesh["row"]
        torch.manual_seed(100 + row.rank())  # a different serial module per row
        serial = attention.Attention(IN_DIM, EMBED_DIM, OUT_DIM, HEADS, device)
        ref_qkv = serial.qkv.weight.detach().clone()
        dist.broadcast(ref_qkv, group_src=0, group=row)

        model = attention.Attention.convert_to_jigsaw(
            serial, mesh, device, ring=True, attn_backend="math"
        )
        assert_close(
            "qkv weight", model.qkv.weight.detach(), ref_qkv, 1e-6, 1e-6, device, row
        )
