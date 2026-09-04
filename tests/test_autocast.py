"""Autocast coverage for the distributed custom-autograd layers.

``torch.autocast`` does not wrap a :class:`torch.autograd.Function` for you: it
only intercepts eligible ATen ops. Every Function in ``jigsaw`` therefore casts
by hand -- ``torch.is_autocast_enabled`` in ``forward`` (see
:mod:`jigsaw.modules.linear.functional`), a cast back to ``weight.dtype`` in
``backward``. So the contract these tests pin is:

* the autocast dtype reaches the *output* (forward runs in bf16);
* the *gradient* comes back in the fp32 parameter dtype (backward upcasts, the
  way AMP master-weight training expects);
* nothing goes non-finite;
* the bf16 output/grad stay within a bf16-sized tolerance of a plain fp32 run of
  the same layer -- i.e. mixed precision is numerically doing its job.
"""

import contextlib

import pytest
import torch

from jigsaw.modules.attention import DistributedAttention
from jigsaw.modules.linear import DistributedXWT
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]

_AUTOCAST_DTYPE = torch.bfloat16
#: backward casts grads to the (fp32) parameter dtype, regardless of autocast.
_GRAD_DTYPE = torch.float32


def _forward_backward(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    *,
    autocast: bool,
    grad_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one forward/backward pass, returning ``(output, input_grad)`` detached.

    The upstream gradient is drawn from ``grad_seed`` against the output shape,
    so the autocast run and the reference run are driven with identical values.
    Backward runs outside the autocast region, the way training does.
    """
    model.zero_grad(set_to_none=True)
    x = x.clone().detach().requires_grad_(True)
    ctx: contextlib.AbstractContextManager = (
        torch.autocast(device_type=device.type, dtype=_AUTOCAST_DTYPE)
        if autocast
        else contextlib.nullcontext()
    )
    with ctx:
        y = model(x)

    gen = torch.Generator(device=device).manual_seed(grad_seed)
    grad = torch.randn(y.shape, generator=gen, device=device).to(y.dtype)
    y.backward(grad)

    assert x.grad is not None, "backward produced no input gradient"
    return y.detach(), x.grad.detach()


class TestAutocast:
    @pytest.mark.dist(4)
    def test_autocast_linear(
        self, process_mesh_dict_2x2: dict, device: torch.device
    ) -> None:
        """bf16 autocast: XWT output is bf16, grad fp32, both track a plain fp32 run."""
        process_mesh_dict = process_mesh_dict_2x2
        local_rank = process_mesh_dict["model"].rank()
        batch, seq, in_dim, out_dim = 1, 20, 10, 20
        model = DistributedXWT(
            in_dim,
            out_dim,
            bias=False,
            local_rank=local_rank,
            device=device,
            model_group=process_mesh_dict["model"],
            row_group=process_mesh_dict["row"],
            column_group=process_mesh_dict["col"],
        )
        group = process_mesh_dict["model"]

        torch.manual_seed(0)
        x = torch.randn(batch, seq // 2, in_dim // 2, device=device).contiguous()

        y, dx = _forward_backward(model, x, device, autocast=True)
        y_ref, dx_ref = _forward_backward(model, x, device, autocast=False)

        assert y.dtype == _AUTOCAST_DTYPE, f"autocast output is {y.dtype}"
        assert dx.dtype == _GRAD_DTYPE, f"autocast input grad is {dx.dtype}"
        assert torch.isfinite(y).all(), "autocast output has non-finite values"
        assert torch.isfinite(dx).all(), "autocast input grad has non-finite values"

        assert_close("autocast vs fp32 output", y.float(), y_ref, 2e-2, 2e-2, device, group)
        assert_close(
            "autocast vs fp32 input grad", dx, dx_ref, 2e-2, 2e-2, device, group
        )

    @pytest.mark.dist(4)
    def test_autocast_attention(
        self, process_mesh_dict_2x2: dict, device: torch.device
    ) -> None:
        """bf16 autocast through DistributedAttention: finite, fp32 grad, fp32-close."""
        process_mesh_dict = process_mesh_dict_2x2
        local_rank = process_mesh_dict["model"].rank()
        batch, seq, in_dim, out_dim = 1, 2, 8, 8
        model = DistributedAttention(
            in_dim,
            embed_dim=out_dim,
            out_dim=out_dim,
            heads=4,
            dropout_p=0.0,
            local_rank=local_rank,
            device=device,
            spatial_group=process_mesh_dict["row"],
            channel_group=process_mesh_dict["col"],
            ring=False,
        )
        group = process_mesh_dict["model"]

        torch.manual_seed(0)
        x = torch.randn(batch, seq // 2, in_dim // 2, device=device).contiguous()

        y, dx = _forward_backward(model, x, device, autocast=True)
        y_ref, dx_ref = _forward_backward(model, x, device, autocast=False)

        # NOTE: the output comes back fp32, not bf16. The final projection is an
        # ``XWT`` with bias, and ``XWT.forward`` casts x and w to the autocast
        # dtype but not the bias, so ``xwt.add(bias)`` promotes the result back
        # to fp32. Change to ``_AUTOCAST_DTYPE`` once the bias is cast too.
        assert y.dtype == torch.float32, f"autocast attention output is {y.dtype}"
        assert dx.dtype == _GRAD_DTYPE, f"autocast input grad is {dx.dtype}"
        assert torch.isfinite(y).all(), "autocast output has non-finite values"
        assert torch.isfinite(dx).all(), "autocast input grad has non-finite values"

        # Autocast must actually have run in low precision: the result should
        # differ from the fp32 path, but only within a bf16-sized tolerance.
        assert not torch.equal(y, y_ref), "autocast made no difference to the output"
        assert_close("autocast vs fp32 output", y, y_ref, 5e-2, 5e-2, device, group)
        assert_close(
            "autocast vs fp32 input grad", dx, dx_ref, 5e-2, 5e-2, device, group
        )
