"""Ulysses (all-to-all, head-parallel) sequence-parallel attention.

Covers the low-level all-to-all round-trip, a forward/backward parity check of
``DistributedAttention(ulysses=True)`` against a gathered-sequence dense
reference built from the module's own weights, composition with a
channel-sharded ``qkv``, and the construction guards.
"""

import pytest
import torch
import torch.distributed as dist
from einops import rearrange

import jigsaw.modules.attention as attention
import jigsaw.modules.attention._config as attention_config
from jigsaw.gather import gather_1d_results, gather_1d_results_list
from jigsaw.modules.attention._attention import attn_core_dtype
from jigsaw.modules.attention._functional.ulysses import (
    _a2a_heads_to_seq,
    _a2a_seq_to_heads,
)
from tests.parity import all_ranks_ok, assert_close, forced_sdpa_backend

pytestmark = [pytest.mark.dist_required]

_IN_DIM = _EMBED_DIM = _OUT_DIM = 256
_HEADS = 4


class TestUlyssesAllToAll:
    @pytest.mark.dist(4)
    def test_roundtrip_is_identity(self, device: torch.device) -> None:
        """``_a2a_seq_to_heads . _a2a_heads_to_seq == id`` and the gathered
        sequence matches a plain all-gather. Also checks the fused
        ``[3, b, s, h, d]`` stack path used for Q/K/V."""
        group = dist.group.WORLD
        assert group is not None
        p = dist.get_world_size(group)
        rank = dist.get_rank(group)

        b, s_loc, h, d = 2, 8, 2 * p, 5
        torch.manual_seed(rank)
        x = torch.randn(b, s_loc, h, d, device=device)

        gathered = _a2a_heads_to_seq(x, group)  # [b, s_loc * p, h // p, d]
        assert gathered.shape == (b, s_loc * p, h // p, d)

        # Sequence axis is source-rank-major: block r is rank r's local x, sliced
        # to this rank's head window: gather every rank's *full* local x first,
        # then slice each by this rank's window (the one every source sent here).
        head0 = rank * (h // p)
        full_blocks = gather_1d_results_list(x.contiguous(), {"g": group}, pm_key="g")
        ref_blocks = [block[:, :, head0 : head0 + h // p, :] for block in full_blocks]
        assert_close(
            "gather matches all_gather",
            gathered,
            torch.cat(ref_blocks, dim=1),
            1e-6,
            1e-6,
            device,
            group,
        )

        back = _a2a_seq_to_heads(gathered, group)  # [b, s_loc, h, d]
        assert_close("roundtrip is identity", back, x, 1e-6, 1e-6, device, group)

        # Fused Q/K/V: a [3, b, s, h, d] stack rides one all-to-all and each
        # slice matches the per-tensor path.
        qkv = torch.stack((x, x + 1, x + 2))
        fused = _a2a_heads_to_seq(qkv, group)
        assert fused.shape == (3, b, s_loc * p, h // p, d)
        for i in range(3):
            assert_close(
                f"fused slice {i}",
                fused[i],
                _a2a_heads_to_seq(qkv[i].contiguous(), group),
                1e-6,
                1e-6,
                device,
                group,
            )
        assert_close(
            "fused roundtrip",
            _a2a_seq_to_heads(fused, group),
            qkv,
            1e-6,
            1e-6,
            device,
            group,
        )


def _dense_reference(
    model: attention.DistributedAttention,
    x_full: torch.Tensor,
) -> torch.Tensor:
    """Replay ``DistributedAttention.forward`` on the full (un-sharded) sequence
    with a single dense SDPA call: the thing Ulysses must reproduce."""
    qkv = model.qkv(x_full)
    q, k, v = (
        rearrange(
            qkv, "... n s (three h d) -> three ... n h s d", three=3, h=model.heads
        )
        .to(attn_core_dtype(x_full))
        .unbind(0)
    )
    with torch.nn.attention.sdpa_kernel(
        [
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]
    ):
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False, dropout_p=0.0, scale=model.scale
        )
    o = rearrange(o, "... n h s d -> ... n s (h d)").to(x_full.dtype)
    return model.head_concat(o)


#: (attn_backend, variant) cases the flat Ulysses parity test below checks.
#: ``variant`` is ``None`` for the default dispatch, otherwise it pins a
#: kernel the default priority order would not otherwise reach on every GPU:
#:
#: * ``("te", "cudnn" | "flash")``: TE's two arms (cuDNN fused attention vs.
#:   the ``flash_attn`` package), selected via ``JIGSAW_TE_CORE`` rather than
#:   letting ``te_cores.select_core`` pick by GPU architecture; needs the
#:   container and a GPU, and each is skipped at runtime if that arm's
#:   package is not importable.
#: * ``("sdpa", "efficient")``: torch's own memory-efficient attention
#:   kernel (``SDPBackend.EFFICIENT_ATTENTION``), forced via
#:   :func:`~tests.parity.forced_sdpa_backend`; ``_dense_bshd_core``'s own
#:   list is ``[FLASH, MATH]``, so plain ``attn_backend="sdpa"`` never
#:   reaches it otherwise.
#:
#: The hierarchical test below checks the same three ``attn_backend``s
#: without pinning a specific kernel (that grid already has its own
#: ``inter_save_scope`` x ``inter_offload`` dimensions); the other Ulysses
#: coverage (``test_composes_with_channel_parallel``,
#: ``test_rejects_bad_config``) stays ``sdpa``-only, since it is wiring/guard
#: coverage, not a numeric parity check.
_ULYSSES_BACKEND_CASES = [
    ("sdpa", None),
    ("math", None),
    pytest.param(
        "sdpa", "efficient", id="sdpa-efficient", marks=[pytest.mark.gpu_required]
    ),
    pytest.param(
        "te",
        "cudnn",
        id="te-cudnn",
        marks=[pytest.mark.te_required, pytest.mark.gpu_required],
    ),
    pytest.param(
        "te",
        "flash",
        id="te-flash",
        marks=[pytest.mark.te_required, pytest.mark.gpu_required],
    ),
]

#: Plain ``attn_backend`` list (no explicit TE-core split) for the
#: hierarchical parity test, which already has its own ``inter_save_scope`` x
#: ``inter_offload`` grid.
_ULYSSES_BACKENDS = [
    "sdpa",
    "math",
    pytest.param("te", marks=[pytest.mark.te_required, pytest.mark.gpu_required]),
]


class TestUlyssesAttention:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("attn_backend,variant", _ULYSSES_BACKEND_CASES)
    def test_matches_dense_reference(
        self,
        device: torch.device,
        process_mesh_dict_4x1: dict,
        attn_backend: str,
        variant: str | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Forward and input-gradient parity vs a gathered-sequence dense run.

        ``spatial=4, channel=1``: ``qkv`` / ``head_concat`` are plain ``Linear``
        with spatially-replicated weights, so every rank can build the exact
        reference from ``model.<w>`` and the all-gathered input. The reference
        itself always runs plain, unforced SDPA (see ``_dense_reference``)
        regardless of ``attn_backend``/``variant``; that is the independent
        ground truth every backend's Ulysses path must reproduce.
        """
        if attn_backend == "te" and variant is not None:
            from jigsaw.modules.attention._functional.cores.te import cudnn, flash

            available = (
                cudnn.CUDNN_CORE_AVAILABLE
                if variant == "cudnn"
                else flash.FLASH_CORE_AVAILABLE
            )
            if not available:
                pytest.skip(f"te core {variant!r} not importable in this environment")
            monkeypatch.setenv("JIGSAW_TE_CORE", variant)

        mesh = process_mesh_dict_4x1
        row = mesh["row"]
        model_group = mesh["model"]
        dtype = torch.float32 if dist.get_backend() != "nccl" else torch.bfloat16
        atol = 1e-4 if dtype == torch.float32 else 2e-2

        torch.manual_seed(0)  # identical replicated weights on every rank
        model = attention.DistributedAttention(
            in_dim=_IN_DIM,
            embed_dim=_EMBED_DIM,
            out_dim=_OUT_DIM,
            heads=_HEADS,
            dropout_p=0.0,
            local_rank=mesh["col"].rank(),
            device=device,
            spatial_group=row,
            channel_group=mesh["col"],
            ring=False,
            ulysses=True,
            attn_backend=attn_backend,
            dtype=dtype,
        )

        b, s_loc = 2, 16
        s_full = s_loc * row.size()
        torch.manual_seed(100 + row.rank())
        x_local = torch.randn(
            b, s_loc, _IN_DIM, device=device, dtype=dtype, requires_grad=True
        )

        if attn_backend == "sdpa" and variant == "efficient":
            with forced_sdpa_backend(
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
            ) as forced:
                out_local = model(x_local)
            # Self-verifying: the forcing has to have actually been consulted,
            # not silently no-op'd (e.g. if a future refactor stopped looking
            # up ``sdpa_kernel`` fresh on ``torch.nn.attention`` per call, in
            # which case this would quietly go back to testing plain ``sdpa``).
            # Reduced, not a bare assert: more collectives follow (the gather
            # and grad sync below), so one rank raising alone would hang the rest.
            if not all_ranks_ok(forced.count > 0, device, model_group):
                raise AssertionError(
                    "forced_sdpa_backend intercepted no sdpa_kernel dispatch; "
                    "the efficient kernel was never pinned, so this ran the "
                    "default priority list"
                )
        else:
            out_local = model(x_local)
        assert out_local.shape == (b, s_loc, _OUT_DIM)

        # Reference on the gathered sequence, built *outside* any forcing
        # context, so it stays on the default, unforced priority list even
        # when ``out_local`` above was forced onto a specific kernel.
        x_full = gather_1d_results(x_local.detach(), mesh, dim=1, pm_key="row")
        x_full.requires_grad_(True)
        out_ref = _dense_reference(model, x_full)

        out_full = gather_1d_results(out_local, mesh, dim=1, pm_key="row")
        assert_close("output", out_full, out_ref, atol, 1e-5, device, model_group)

        # Input-gradient parity. Drive both with the same global upstream grad;
        # the local run gets this rank's sequence slice of it.
        torch.manual_seed(7)
        dout_full = torch.randn(b, s_full, _OUT_DIM, device=device, dtype=dtype)
        r0 = row.rank() * s_loc
        out_local.backward(dout_full[:, r0 : r0 + s_loc, :])
        out_ref.backward(dout_full)

        dx_full = gather_1d_results(x_local.grad, mesh, dim=1, pm_key="row")
        assert_close(
            "input grad", dx_full, x_full.grad, atol, 1e-5, device, model_group
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("attn_backend", _ULYSSES_BACKENDS)
    @pytest.mark.parametrize("inter_save_scope", ["LOCAL", "GLOBAL"])
    @pytest.mark.parametrize(
        "inter_offload",
        [attention_config.Offload.DISABLED, attention_config.Offload.CPU],
    )
    def test_hierarchical_matches_dense_reference(
        self,
        device: torch.device,
        process_mesh_dict_4x1: dict,
        inter_save_scope: str,
        inter_offload: attention_config.Offload,
        attn_backend: str,
    ) -> None:
        """Hierarchical Ulysses (intra-group a2a x inter-group K/V P2P ring)
        reproduces the same gathered-sequence dense run as the flat scheme.

        ``spatial=4, ulysses_intra_size=2``: a 2-rank intra-group all-to-all
        nested in a 2-group outer P2P ring, with both backward
        ``save_scope``s (``LOCAL`` re-rings K/V in the backward, ``GLOBAL``
        keeps it). The ring's per-step online-softmax merge adds a few ULP the
        flat single-kernel path does not. Every ``attn_backend`` (math, sdpa,
        te) feeds both the intra-group all-to-all's inner dense core
        (``_dense_bshd_core``) and the outer ring's inner core
        (``_inter_ring_core``); ``te`` picks its kernel by GPU
        architecture here rather than forcing a specific core, unlike
        ``test_matches_dense_reference``'s split cases, to keep this grid's
        size in check.

        ``inter_offload=CPU`` additionally routes the outer ring's saved
        K/V chunk through :class:`.offload.NVLinkActivationOffloader` (pinned
        host memory between forward and backward): same mechanism, same
        parity expectation, as the plain ring's own offload tests.
        """
        if inter_offload == attention_config.Offload.CPU and device.type != "cuda":
            pytest.skip("NVLinkActivationOffloader needs CUDA streams")

        mesh = process_mesh_dict_4x1
        row = mesh["row"]
        model_group = mesh["model"]
        dtype = torch.float32 if dist.get_backend() != "nccl" else torch.bfloat16
        atol = 1e-4 if dtype == torch.float32 else 2e-2

        torch.manual_seed(0)
        model = attention.DistributedAttention(
            in_dim=_IN_DIM,
            embed_dim=_EMBED_DIM,
            out_dim=_OUT_DIM,
            heads=_HEADS,
            dropout_p=0.0,
            local_rank=mesh["col"].rank(),
            device=device,
            spatial_group=row,
            channel_group=mesh["col"],
            ring=False,
            ulysses=True,
            ulysses_intra_size=2,
            ulysses_inter_save_scope=inter_save_scope,
            ulysses_inter_offload=inter_offload,
            attn_backend=attn_backend,
            dtype=dtype,
        )

        b, s_loc = 2, 16
        s_full = s_loc * row.size()
        torch.manual_seed(100 + row.rank())
        x_local = torch.randn(
            b, s_loc, _IN_DIM, device=device, dtype=dtype, requires_grad=True
        )
        out_local = model(x_local)
        assert out_local.shape == (b, s_loc, _OUT_DIM)

        x_full = gather_1d_results(x_local.detach(), mesh, dim=1, pm_key="row")
        x_full.requires_grad_(True)
        out_ref = _dense_reference(model, x_full)
        out_full = gather_1d_results(out_local, mesh, dim=1, pm_key="row")
        assert_close("output", out_full, out_ref, atol, 1e-4, device, model_group)

        torch.manual_seed(7)
        dout_full = torch.randn(b, s_full, _OUT_DIM, device=device, dtype=dtype)
        r0 = row.rank() * s_loc
        out_local.backward(dout_full[:, r0 : r0 + s_loc, :])
        out_ref.backward(dout_full)
        dx_full = gather_1d_results(x_local.grad, mesh, dim=1, pm_key="row")
        assert_close(
            "input grad", dx_full, x_full.grad, atol, 1e-4, device, model_group
        )

    @pytest.mark.dist(4)
    def test_composes_with_channel_parallel(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Ulysses over a 2-way spatial group with a channel-sharded ``qkv``.

        ``heads // channel = 2`` is divisible by ``spatial = 2``. Correctness vs
        a serial reference is left to :meth:`test_matches_dense_reference`; this
        guards the wiring forward and backward and the spatial param sync.
        """
        mesh = process_mesh_dict_2x2
        model_group = mesh["model"]
        dtype = torch.float32 if dist.get_backend() != "nccl" else torch.bfloat16

        model = attention.DistributedAttention(
            in_dim=_IN_DIM,
            embed_dim=_EMBED_DIM,
            out_dim=_OUT_DIM,
            heads=_HEADS,
            dropout_p=0.0,
            local_rank=mesh["col"].rank(),
            device=device,
            spatial_group=mesh["row"],
            channel_group=mesh["col"],
            ring=False,
            ulysses=True,
            attn_backend="sdpa",
            dtype=dtype,
        )

        local_channels = _IN_DIM // mesh["col"].size()
        x = torch.randn(
            2, 12, local_channels, device=device, dtype=dtype, requires_grad=True
        )
        out = model(x)
        # ``head_concat`` is channel-parallel too: like ``qkv``, its output stays
        # sharded along ``out_dim`` within ``channel_group`` (see
        # ``test_attention.py::test_converter_parity``, which gathers it back
        # explicitly) rather than being gathered inside the module.
        assert out.shape == (2, 12, local_channels)
        out.sum().backward()

        assert all_ranks_ok(bool(torch.isfinite(out).all()), device, model_group)
        assert all_ranks_ok(bool(torch.isfinite(x.grad).all()), device, model_group)

        # qkv weight is replicated across the spatial group (broadcast + grad
        # sync); shards 0 and 2 share a channel column on a row-major 2x2 mesh.
        shards = gather_1d_results_list(
            model.qkv.weight, process_mesh_dict=mesh, pm_key="model"
        )
        assert_close(
            "qkv weight across spatial",
            shards[0],
            shards[2],
            1e-8,
            1e-5,
            device,
            model_group,
        )

    @pytest.mark.dist(4)
    def test_rejects_bad_config(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        process_mesh_dict_1x4: dict,
    ) -> None:
        """Construction fails fast on spatial=1, an indivisible head count, or
        ``ring`` and ``ulysses`` both set."""
        common = dict(
            in_dim=_IN_DIM,
            embed_dim=_EMBED_DIM,
            out_dim=_OUT_DIM,
            heads=_HEADS,
            dropout_p=0.0,
            device=device,
            attn_backend="sdpa",
        )
        with pytest.raises(AssertionError, match="spatial size is 1"):
            attention.DistributedAttention(
                local_rank=process_mesh_dict_1x4["col"].rank(),
                spatial_group=process_mesh_dict_1x4["row"],
                channel_group=process_mesh_dict_1x4["col"],
                ulysses=True,
                **common,
            )
        # heads=2 over channel=2 leaves 1 head per channel shard, which the 2-way
        # spatial group cannot split.
        with pytest.raises(AssertionError, match="divisible by spatial group size"):
            attention.DistributedAttention(
                local_rank=process_mesh_dict_2x2["col"].rank(),
                spatial_group=process_mesh_dict_2x2["row"],
                channel_group=process_mesh_dict_2x2["col"],
                ulysses=True,
                **{**common, "heads": 2},
            )
        with pytest.raises(ValueError, match="mutually exclusive"):
            attention.DistributedAttention(
                local_rank=process_mesh_dict_2x2["col"].rank(),
                spatial_group=process_mesh_dict_2x2["row"],
                channel_group=process_mesh_dict_2x2["col"],
                ring=True,
                ulysses=True,
                **common,
            )
