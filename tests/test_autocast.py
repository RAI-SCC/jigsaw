"""Autocast coverage for the distributed custom-autograd layers.

``torch.autocast`` only ever targets fp16 or bf16. fp64 and fp8 are *not*
autocast dtypes -- pass either and PyTorch warns, disables autocast and stays in
fp32 -- so "autocast in fp8" is not a thing that exists; fp8 needs an explicit
scaled path (``torch._scaled_mm`` / transformer_engine) and a transport that can
carry it (Gloo cannot). Within the fp16/bf16 range, jigsaw's Functions cast by
hand and these tests pin the contract:

* the matmul family (``XWT`` / ``XW`` / ``XTW`` -- bias included -- attention,
  ring attention, ``ConvNd`` / ``ConvTransposeNd``) runs forward in the autocast
  dtype and reconciles gradients to the fp32 parameter dtype in backward;
* LayerNorm and softmax follow stock-AMP fp32-list semantics -- computed and
  emitted in fp32 under autocast even when fed bf16/fp16;
* nothing goes non-finite, and every result tracks a high-precision reference.

Known quirks this file documents rather than hides:

* the attention core follows :func:`jigsaw...attention.attn_core_dtype`: the
  autocast dtype under autocast, else bf16 on CUDA (flash rejects fp32), else
  the input dtype on CPU. ``DistributedAttention`` still can't run an fp64
  parameter set (it force-casts q/k/v), so attention tests use fp32 params.
* ring attention drives its forward under autocast (bf16/fp16 saved tensors) and
  ``ring_backward`` reconciles them to the ``w_qkv`` (master) dtype before the
  cores run -- the same backward-in-master-precision rule ``XWT`` follows.
"""

import contextlib

import pytest
import torch
import torch.distributed as dist

import jigsaw.modules.activation.activation as activation
import jigsaw.modules.convolution as convolution
import jigsaw.modules.normalization.normalization as norm
from jigsaw.modules.attention import DistributedAttention
from jigsaw.modules.linear import DistributedXWT
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]

_AUTOCAST_DTYPES = [torch.bfloat16, torch.float16]
_BASE_DTYPES = [torch.float32, torch.float64]

#: (atol, rtol) for an autocast run vs a high-precision reference, by autocast dtype.
_TOL = {torch.bfloat16: (3e-2, 3e-2), torch.float16: (6e-3, 6e-3)}
#: attention runs a bf16/fp16 core regardless of which autocast dtype, so both
#: get one generous tolerance against the fp32 (no-autocast) reference.
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
@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
class TestAutocastLinear:
    """XWT: forward emits the autocast dtype (bias included), backward the param dtype."""

    @pytest.mark.dist(4)
    def test_autocast_linear(
        self,
        process_mesh_dict_2x2: dict,
        device: torch.device,
        bias: bool,
        base_dtype: torch.dtype,
        autocast_dtype: torch.dtype,
    ) -> None:
        pm = process_mesh_dict_2x2
        in_dim, out_dim = 10, 20
        model = DistributedXWT(
            in_dim,
            out_dim,
            local_rank=pm["model"].rank(),
            device=device,
            model_group=pm["model"],
            column_group=pm["col"],
            row_group=pm["row"],
            bias=bias,
        ).to(base_dtype)
        group = pm["model"]

        torch.manual_seed(0)
        x = torch.randn(
            1, 10, in_dim // 2, device=device, dtype=base_dtype
        ).contiguous()

        y, dx = _forward_backward(model, x, device, autocast_dtype, autocast=True)
        y_ref, dx_ref = _forward_backward(
            model, x, device, autocast_dtype, autocast=False
        )

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

    The core runs in the autocast dtype (see ``attn_core_dtype``) and the module
    output carries it -- the ``head_concat`` bias is cast down too. fp64 params
    remain unsupported (q/k/v are force-cast).
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

        assert y.dtype == autocast_dtype, f"attention output is {y.dtype}"
        assert dx.dtype == torch.float32, f"input grad is {dx.dtype}"
        assert torch.isfinite(y).all(), "non-finite output"
        assert torch.isfinite(dx).all(), "non-finite input grad"

        atol, rtol = _ATTN_TOL
        assert_close("output", y.float(), y_ref.float(), atol, rtol, device, group)
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
                out, lse, chunks, base_seed, rng_states = ring_fwd.ring_forward(
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
            assert_close(
                f"ring {name}", g.float(), r.float(), atol, rtol, device, ring_group
            )


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

        assert y.dtype == torch.float32, (
            f"fp32-list op returned {y.dtype} under autocast"
        )
        assert dx.dtype == autocast_dtype, (
            f"input grad is {dx.dtype}, not the leaf dtype"
        )
        assert torch.isfinite(y).all(), "non-finite output"
        assert torch.isfinite(dx).all(), "non-finite input grad"

        # Same values, fp32, no autocast: the fp32-list op must match it closely.
        assert_close("output", y, y_ref, 1e-4, 1e-4, device, group)
        atol, rtol = _TOL[autocast_dtype]
        assert_close(
            "input grad", dx.float(), dx_ref.float(), atol, rtol, device, group
        )

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
        model = activation.DistributedSoftMax(pm["col"], dim=-1)
        torch.manual_seed(pm["model"].rank())
        x_low = torch.randn(4, 6, local_dim, device=device, dtype=autocast_dtype)
        self._check_fp32_list(model, x_low, device, autocast_dtype, pm["model"])


@pytest.mark.parametrize("autocast_dtype", _AUTOCAST_DTYPES)
@pytest.mark.parametrize(
    "kernel, stride, padding",
    [((2, 2), (2, 2), (0, 0)), ((3, 3), (1, 1), (1, 1))],
    ids=["no_halo", "halo"],
)
class TestAutocastConv:
    """ConvNd / ConvTransposeNd: forward emits the autocast dtype, backward fp32.

    Both regimes -- ``stride == kernel`` (no channel/halo comm in backward) and
    ``stride != kernel`` (halo exchange) -- are swept, both with a real channel
    split so the custom Function (not the ``_forward_local`` shortcut) runs.
    """

    def _check(
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

        assert y.dtype == autocast_dtype, f"conv output is {y.dtype}"
        assert dx.dtype == torch.float32, f"input grad is {dx.dtype}, not fp32"
        assert (
            model.weight.grad is not None and model.weight.grad.dtype == torch.float32
        )
        assert torch.isfinite(y).all(), "non-finite output"
        assert torch.isfinite(dx).all(), "non-finite input grad"

        atol, rtol = _TOL[autocast_dtype]
        assert_close("output", y.float(), y_ref, atol, rtol, device, group)
        assert_close("input grad", dx, dx_ref, atol, rtol, device, group)

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("transpose", [False, True], ids=["conv", "transpose"])
    def test_autocast_conv2d(
        self,
        process_mesh_dict_2x2: dict,
        device: torch.device,
        kernel: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        transpose: bool,
        autocast_dtype: torch.dtype,
    ) -> None:
        pm = process_mesh_dict_2x2
        nc, nsp = pm["col"].size(), pm["row"].size()
        in_ch, out_ch = 4, 4
        cls = (
            convolution.DistributedConv2DTranspose
            if transpose
            else convolution.DistributedConv2D
        )
        model = cls(
            in_ch,
            out_ch,
            kernel,
            device=device,
            spatial_group=pm["row"],
            channel_group=pm["col"],
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            bias=True,
            dtype=torch.float32,
        ).to(device)

        torch.manual_seed(pm["model"].rank())
        x = torch.randn(2, in_ch // nc, 8, 8 // nsp, device=device)
        self._check(model, x, device, autocast_dtype, pm["model"])
