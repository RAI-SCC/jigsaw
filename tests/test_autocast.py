"""Autocast coverage for the distributed custom-autograd layers.

``torch.autocast`` only ever targets fp16 or bf16. fp64 and fp8 are *not*
autocast dtypes -- pass either and PyTorch warns, disables autocast and stays in
fp32 -- so "autocast in fp8" is not a thing that exists; fp8 needs an explicit
scaled path (``torch._scaled_mm`` / transformer_engine) and a transport that can
carry it (Gloo cannot). Within the fp16/bf16 range, jigsaw's Functions cast by
hand and these tests pin the contract:

* the matmul family (``XWT``, attention, ring attention) runs forward in the
  autocast dtype and reconciles gradients to the fp32/fp64 parameter dtype in
  backward;
* LayerNorm and softmax follow stock-AMP fp32-list semantics -- computed and
  emitted in fp32 under autocast even when fed bf16/fp16;
* nothing goes non-finite, and every result tracks a high-precision reference.

Known quirks this file documents rather than hides:

* ``DistributedAttention.forward`` hard-codes ``.to(torch.bfloat16)`` on q/k/v
  and ``.to(self.dtype)`` on the core output, so attention ignores the autocast
  dtype (always bf16 core) and cannot run an fp64 parameter set. Attention tests
  therefore use fp32 params and a bf16-sized tolerance for both autocast dtypes.
* bias-carrying ``XWT`` returns the parameter dtype, not the autocast dtype,
  because ``XWT.forward`` casts x and w but not the bias.
* ring attention drives its forward under autocast (bf16/fp16 saved tensors) and
  ``ring_backward`` upcasts them to fp32 before the cores run.
"""

import contextlib

import pytest
import torch
import torch.distributed as dist

import jigsaw.modules.normalization.normalization as norm
from jigsaw.modules.attention import DistributedAttention
from jigsaw.modules.linear import DistributedXWT
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]

_AUTOCAST_DTYPES = [torch.bfloat16, torch.float16]
_BASE_DTYPES = [torch.float32, torch.float64]

#: (atol, rtol) for an autocast run vs a high-precision reference, by autocast dtype.
_TOL = {torch.bfloat16: (3e-2, 3e-2), torch.float16: (6e-3, 6e-3)}
#: attention always runs a bf16 core (hard-coded), so fp16 gets the bf16 tolerance.
_ATTN_TOL = (6e-2, 6e-2)


def _forward_backward(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    autocast_dtype: torch.dtype,
    *,
    autocast: bool,
    grad_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one forward/backward pass, returning ``(output, input_grad)`` detached.

    The upstream gradient is drawn from ``grad_seed`` against the output shape so
    the autocast run and the reference run see identical values. Backward runs
    outside the autocast region, the way training does.
    """
    model.zero_grad(set_to_none=True)
    x = x.clone().detach().requires_grad_(True)
    ctx: contextlib.AbstractContextManager = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast
        else contextlib.nullcontext()
    )
    with ctx:
        y = model(x)

    gen = torch.Generator(device=device).manual_seed(grad_seed)
    y.backward(torch.randn(y.shape, generator=gen, device=device).to(y.dtype))

    assert x.grad is not None, "backward produced no input gradient"
    return y.detach(), x.grad.detach()


@pytest.mark.parametrize("autocast_dtype", _AUTOCAST_DTYPES)
@pytest.mark.parametrize("base_dtype", _BASE_DTYPES)
class TestAutocastLinear:
    """XWT: forward emits the autocast dtype, backward the fp32/fp64 param dtype."""

    @pytest.mark.dist(4)
    def test_autocast_linear(
        self,
        process_mesh_dict_2x2: dict,
        device: torch.device,
        base_dtype: torch.dtype,
        autocast_dtype: torch.dtype,
    ) -> None:
        pm = process_mesh_dict_2x2
        in_dim, out_dim = 10, 20
        model = DistributedXWT(
            in_dim,
            out_dim,
            bias=False,
            local_rank=pm["model"].rank(),
            device=device,
            model_group=pm["model"],
            row_group=pm["row"],
            column_group=pm["col"],
        ).to(base_dtype)
        group = pm["model"]

        torch.manual_seed(0)
        x = torch.randn(1, 10, in_dim // 2, device=device, dtype=base_dtype).contiguous()

        y, dx = _forward_backward(model, x, device, autocast_dtype, autocast=True)
        y_ref, dx_ref = _forward_backward(model, x, device, autocast_dtype, autocast=False)

        assert y.dtype == autocast_dtype, f"forward output is {y.dtype}"
        assert dx.dtype == base_dtype, f"input grad is {dx.dtype}, not the param dtype"
        assert torch.isfinite(y).all(), "non-finite output"
        assert torch.isfinite(dx).all(), "non-finite input grad"

        atol, rtol = _TOL[autocast_dtype]
        assert_close("output", y.to(base_dtype), y_ref, atol, rtol, device, group)
        assert_close("input grad", dx, dx_ref, atol, rtol, device, group)


@pytest.mark.parametrize("autocast_dtype", _AUTOCAST_DTYPES)
class TestAutocastAttention:
    """DistributedAttention, plain and ring: autocast-safe, finite, fp32-close.

    fp64 params are unsupported (hard-coded bf16 casts in ``forward``); the
    output lands in the fp32 param dtype because the ``head_concat`` bias is not
    cast down.
    """

    def _run(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        device: torch.device,
        autocast_dtype: torch.dtype,
        group: dist.ProcessGroup,
    ) -> None:
        y, dx = _forward_backward(model, x, device, autocast_dtype, autocast=True)
        y_ref, dx_ref = _forward_backward(
            model, x, device, autocast_dtype, autocast=False
        )

        assert y.dtype == torch.float32, f"attention output is {y.dtype}"
        assert dx.dtype == torch.float32, f"input grad is {dx.dtype}"
        assert torch.isfinite(y).all(), "non-finite output"
        assert torch.isfinite(dx).all(), "non-finite input grad"

        atol, rtol = _ATTN_TOL
        assert_close("output", y, y_ref, atol, rtol, device, group)
        assert_close("input grad", dx, dx_ref, atol, rtol, device, group)

    @pytest.mark.dist(4)
    def test_autocast_attention(
        self,
        process_mesh_dict_2x2: dict,
        device: torch.device,
        autocast_dtype: torch.dtype,
    ) -> None:
        pm = process_mesh_dict_2x2
        in_dim = 8
        model = DistributedAttention(
            in_dim,
            embed_dim=8,
            out_dim=8,
            heads=4,
            dropout_p=0.0,
            local_rank=pm["model"].rank(),
            device=device,
            spatial_group=pm["row"],
            channel_group=pm["col"],
            ring=False,
        )
        torch.manual_seed(0)
        x = torch.randn(1, 2, in_dim // pm["col"].size(), device=device)
        self._run(model, x, device, autocast_dtype, pm["model"])

    @pytest.mark.dist(4)
    def test_autocast_ring_attention(
        self,
        process_mesh_dict_4x1: dict,
        device: torch.device,
        autocast_dtype: torch.dtype,
    ) -> None:
        # Drive the ring forward / backward directly (as tests/modules/attention
        # does): the forward runs under autocast -- its saved ``out`` / ``lse`` /
        # K-V chunks come back in the autocast dtype -- and ``ring_backward``
        # must reconcile them against the fp32 ``q`` without a dtype clash.
        from jigsaw.modules.attention import config as attn_config
        from jigsaw.modules.attention.functional import backward as ring_bwd
        from jigsaw.modules.attention.functional import forward as ring_fwd

        pm = process_mesh_dict_4x1
        ring_group = pm["row"]
        local_rank = ring_group.rank()
        world = ring_group.size()
        cfg = attn_config.RingAttentionConfig(device=device)
        ring_forward = ring_fwd.build_ring_forward(cfg)

        b, h, d = 2, 6, 16
        n_local = (world * 16) // world
        torch.manual_seed(2)
        w_qkv = torch.randn(3 * d * h, d, device=device) / d**0.5
        torch.manual_seed(local_rank)
        x0 = (torch.randn(b, n_local, d, device=device) / d**0.5).contiguous()
        scale = d**-0.5

        def run(autocast: bool) -> tuple[torch.Tensor, ...]:
            qkv = x0 @ w_qkv.mT
            q, k, v = (
                t.contiguous()
                for t in qkv.reshape(b, n_local, 3, h, d).permute(2, 0, 3, 1, 4)
            )
            ctx: contextlib.AbstractContextManager = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast
                else contextlib.nullcontext()
            )
            with ctx:
                out, lse, chunks = ring_forward(
                    q, k, v, x0, w_qkv, scale, 0.0, ring_group, local_rank, world, cfg
                )
            dq, dk, dv = ring_bwd.ring_backward(
                torch.ones_like(out),
                q,
                out,
                x0,
                lse,
                w_qkv,
                scale,
                ring_group,
                local_rank,
                world,
                chunks,
                ring_config=cfg,
            )
            return out.detach(), dq.detach(), dk.detach(), dv.detach()

        got = run(autocast=True)
        ref = run(autocast=False)

        atol, rtol = _ATTN_TOL
        for name, g, r in zip(("out", "dq", "dk", "dv"), got, ref):
            assert torch.isfinite(g).all(), f"non-finite ring {name}"
            assert_close(f"ring {name}", g.float(), r.float(), atol, rtol, device, ring_group)


@pytest.mark.parametrize("autocast_dtype", _AUTOCAST_DTYPES)
class TestAutocastFp32List:
    """LayerNorm and softmax: fp32 compute + fp32 output under autocast (stock AMP)."""

    def _check_fp32_list(
        self,
        model: torch.nn.Module,
        x_low: torch.Tensor,
        device: torch.device,
        autocast_dtype: torch.dtype,
        group: dist.ProcessGroup,
    ) -> None:
        y, dx = _forward_backward(model, x_low, device, autocast_dtype, autocast=True)
        y_ref, dx_ref = _forward_backward(
            model, x_low.float(), device, autocast_dtype, autocast=False
        )

        assert y.dtype == torch.float32, f"fp32-list op returned {y.dtype} under autocast"
        assert dx.dtype == autocast_dtype, f"input grad is {dx.dtype}, not the leaf dtype"
        assert torch.isfinite(y).all(), "non-finite output"
        assert torch.isfinite(dx).all(), "non-finite input grad"

        # Same values, fp32, no autocast: the fp32-list op must match it closely.
        assert_close("output", y, y_ref, 1e-4, 1e-4, device, group)
        atol, rtol = _TOL[autocast_dtype]
        assert_close("input grad", dx.float(), dx_ref.float(), atol, rtol, device, group)

    @pytest.mark.dist(4)
    def test_autocast_layernorm(
        self,
        process_mesh_dict_2x2: dict,
        device: torch.device,
        autocast_dtype: torch.dtype,
    ) -> None:
        pm = process_mesh_dict_2x2
        local_dim = 12
        model = norm.DistributedLayerNorm(
            local_dim=local_dim,
            local_rank=pm["model"].rank(),
            device=device,
            row_group=pm["row"],
            col_group=pm["col"],
        )
        torch.manual_seed(pm["col"].rank())
        model.weight.data = torch.randn(local_dim, device=device)
        model.bias.data = torch.randn(local_dim, device=device)

        torch.manual_seed(pm["model"].rank())
        # Low-precision input, as if straight out of an autocast matmul.
        x_low = torch.randn(4, 6, local_dim, device=device, dtype=autocast_dtype)
        self._check_fp32_list(model, x_low, device, autocast_dtype, pm["model"])

    @pytest.mark.dist(4)
    def test_autocast_noparam_layernorm(
        self,
        process_mesh_dict_2x2: dict,
        device: torch.device,
        autocast_dtype: torch.dtype,
    ) -> None:
        pm = process_mesh_dict_2x2
        local_dim = 12
        model = norm.DistributedNoParamLayerNorm(
            pm["model"].rank(), device, channel_group=pm["col"]
        )
        torch.manual_seed(pm["model"].rank())
        x_low = torch.randn(4, 6, local_dim, device=device, dtype=autocast_dtype)
        self._check_fp32_list(model, x_low, device, autocast_dtype, pm["model"])

    @pytest.mark.dist(4)
    def test_autocast_softmax(
        self,
        process_mesh_dict_2x2: dict,
        device: torch.device,
        autocast_dtype: torch.dtype,
    ) -> None:
        pm = process_mesh_dict_2x2
        local_dim = 12
        model = norm.DistributedSoftMax(pm["col"], dim=-1)
        torch.manual_seed(pm["model"].rank())
        x_low = torch.randn(4, 6, local_dim, device=device, dtype=autocast_dtype)
        self._check_fp32_list(model, x_low, device, autocast_dtype, pm["model"])
