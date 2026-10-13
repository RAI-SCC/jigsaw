"""Kernel selection in ``LinearConfig`` (pure dispatch logic, no communication)."""

import pytest
import torch

import jigsaw.nn.linear.distributed_matmul as dist_mm
import jigsaw.nn.linear.distributed_matmul_p2p as dist_mm_p2p
from jigsaw.nn.linear.config import LinearConfig

CPU, CUDA = torch.device("cpu"), torch.device("cuda")
WAYS = {2: "twoway", 4: "fourway", 8: "eightway"}
BIAS = torch.zeros(3)


def _config(parallelism=1, bias=BIAS, bayesian=False, linear_layer=True, device=CPU):
    return LinearConfig(parallelism, bias, bayesian, linear_layer, device)


class TestMatmulKernelSelection:
    @pytest.mark.parametrize(
        "dev, module", [(CPU, dist_mm_p2p), (CUDA, dist_mm)], ids=["cpu", "cuda"]
    )
    @pytest.mark.parametrize("parallelism, way", WAYS.items())
    def test_distributed_kernels_come_from_the_device_backend(
        self, dev, module, parallelism, way
    ):
        xwt, xw, xtw, fused = _config(parallelism, device=dev).mm_kernels
        assert xwt is getattr(module, f"xw_transpose_{way}")
        assert xw is getattr(module, f"xw_{way}")
        assert xtw is getattr(module, f"x_transpose_w_{way}")
        assert fused is getattr(module, f"fused_xw_xtw_{way}")

    @pytest.mark.parametrize("dev", [CPU, CUDA], ids=["cpu", "cuda"])
    def test_single_process_kernels_are_plain_matmuls(self, dev):
        xwt, xw, xtw, fused = _config(1, device=dev).mm_kernels
        x, w = torch.randn(2, 4, 5), torch.randn(3, 5)
        assert torch.equal(xwt(x, w, "ignored", key="ignored"), x @ w.mT)
        w = torch.randn(5, 3)
        assert torch.equal(xw(x, w), x @ w)
        x, w = torch.randn(2, 5, 4), torch.randn(2, 5, 3)
        assert torch.equal(xtw(x, w), x.mT @ w)
        assert fused is dist_mm.fused_xw_xtw_oneway

    @pytest.mark.parametrize("dev", [CPU, CUDA], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("parallelism", [0, 3, 16])
    def test_unsupported_parallelism_is_rejected(self, dev, parallelism):
        with pytest.raises(ValueError, match="Unsupported parallelism"):
            _config(parallelism, device=dev)

    @pytest.mark.parametrize("parallelism", [1, 2])
    def test_unsupported_device_is_rejected(self, parallelism):
        with pytest.raises(ValueError, match="Unsupported device meta"):
            _config(parallelism, device=torch.device("meta"))


class TestBiasSelection:
    def test_no_bias_means_identity_add_and_no_gradient(self):
        cfg = _config(bias=None)
        x = torch.randn(2, 3)
        assert cfg.add_bias(x, None) is x
        assert cfg.grad_bias(x) is None

    def test_add_bias_broadcasts(self):
        cfg = _config(bias=BIAS)
        x, b = torch.zeros(2, 4, 3), torch.arange(3.0)
        assert torch.equal(cfg.add_bias(x, b), b.expand(2, 4, 3))

    def test_bayesian_add_bias_inserts_member_axes(self):
        cfg = _config(bias=BIAS, bayesian=True)
        x, b = torch.zeros(2, 5, 4, 3), torch.arange(6.0).reshape(2, 3)
        out = cfg.add_bias(x, b)
        assert out.shape == x.shape
        assert torch.equal(out[1, 2, 3], b[1])

    def test_grad_bias_sums_batch_and_space_for_3d_and_up(self):
        cfg = _config(bias=BIAS)
        grad = torch.randn(2, 4, 3)
        assert torch.allclose(cfg.grad_bias(grad), grad.sum(dim=(-3, -2)))

    def test_grad_bias_sums_rows_for_2d(self):
        cfg = _config(bias=BIAS)
        grad = torch.randn(4, 3)
        assert torch.allclose(cfg.grad_bias(grad), grad.sum(dim=-2))


class TestWeightSelection:
    def test_reshape_only_for_bayesian_linear_layers(self):
        w = torch.randn(3, 4)
        assert _config(bayesian=True).reshape_weights(w).shape == (3, 1, 4)
        assert _config(bayesian=False).reshape_weights(w) is w
        assert _config(bayesian=True, linear_layer=False).reshape_weights(w) is w

    def test_grad_sum_reduces_batch_dim_of_linear_layers(self):
        grad = torch.randn(2, 3, 4)
        cfg = _config(linear_layer=True)
        assert torch.equal(cfg.grad_sum(grad, torch.tensor(0)), grad.sum(dim=0))

    def test_grad_sum_leaves_2d_gradients_alone(self):
        grad = torch.randn(3, 4)
        assert _config(linear_layer=True).grad_sum(grad, torch.tensor(0)) is grad

    def test_grad_sum_is_identity_for_raw_matrix_ops(self):
        grad = torch.randn(2, 3, 4)
        assert _config(linear_layer=False).grad_sum(grad, torch.tensor(0)) is grad


SELECTORS = [
    "_select_xw_transpose_kernel",
    "_select_xw_kernel",
    "_select_x_transpose_w_kernel",
    "_select_xw_xtw_fused_kernel",
]


class TestEachSelectorValidates:
    """``_mm_kernel_selection`` stops at the first bad selector, so hit each directly."""

    @pytest.mark.parametrize("selector", SELECTORS)
    @pytest.mark.parametrize("dev", [CPU, CUDA], ids=["cpu", "cuda"])
    def test_unsupported_parallelism(self, selector, dev):
        with pytest.raises(ValueError, match="Unsupported parallelism: 3"):
            getattr(_config(), selector)(3, dev)

    @pytest.mark.parametrize("selector", SELECTORS)
    def test_unsupported_device(self, selector):
        with pytest.raises(ValueError, match="Unsupported device meta"):
            getattr(_config(), selector)(2, torch.device("meta"))
